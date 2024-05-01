""" Sat 27 May 2023 03:19:49 PM CDT """

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
import sys
import numpy as np
import pyopencl as cl

from functools import partial

from dataclasses import dataclass, fields

from arraycontext import (
    dataclass_array_container, with_container_arithmetic,
    get_container_context_recursively
)

from meshmode.dof_array import DOFArray

from grudge import op
import grudge.geometry as geo
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, BoundaryDomainTag, VolumeDomainTag
)
from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.utils import force_evaluation, normalize_boundaries
from mirgecom.simutil import (
    check_naninf_local,
    generate_and_distribute_mesh,
    write_visfile,
    check_step,
    global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PressureOutflowBoundary,
    LinearizedOutflowBoundary,  
    DummyBoundary
)
from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient, make_conserved
)
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    make_operator_fluid_states
)
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage)
from mirgecom.navierstokes import (
    grad_t_operator,
    grad_cv_operator,
    ns_operator)
from mirgecom.fluid import ConservedVars

from mirgecom.eos import MixtureEOS, GasEOS, GasDependentVars

from mirgecom.wall_model import PorousWallTransport
from mirgecom.wall_model import PorousFlowModel

from typing import Optional, Union
from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array

#########################################################################

class _RadiationTag:
    pass

class _SampleOpStatesTag:
    pass

class _SampleGradCVTag:
    pass

class _SampleGradTempTag:
    pass

class _SampleGradPresTag:
    pass

class _SampleOperatorTag:
    pass


from mirgecom.materials.carbon_fiber import FiberEOS as OriginalFiberEOS
class FiberEOS(OriginalFiberEOS):
    """Inherits and modified the original carbon fiber."""

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the carbon fiber."""
        virgin = 1e-8
        char = 1.5125e-5
        return virgin*tau + char*(1.0 - tau)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:

    def __init__(self, *, x_min=None, x_max=None, x_thickness, amplitude):
        self._x_min = x_min
        self._x_max = x_max
        self._x_thickness = x_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec):
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos

        sponge_x = xpos*0.0

        if (self._x_max is not None):
          x0 = (self._x_max - self._x_thickness)
          dx = +((xpos - x0)/self._x_thickness)
          sponge_x = sponge_x + self._amplitude * actx.np.where(
              actx.np.greater(xpos, x0),
                  actx.np.where(actx.np.greater(xpos, self._x_max),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
                  0.0
          )

        if (self._x_min is not None):
          x0 = (self._x_min + self._x_thickness)
          dx = -((xpos - x0)/self._x_thickness)
          sponge_x = sponge_x + self._amplitude * actx.np.where(
              actx.np.less(xpos, x0),
                  actx.np.where(actx.np.less(xpos, self._x_min),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
              0.0
          )

        return sponge_x


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=False,
                           use_einsum_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 1000
    nrestart = 10000
    nhealth = 1
    nstatus = 100
    ngarbage = 10

    # default timestepping control
    maximum_sample_dt = 1.0e-6
    t_final = 1.0

    local_dt = False
    constant_cfl = False
    maximum_cfl = 0.2
    
    # discretization and model control
    order = 3
    use_overintegration = False

    my_material = "fiber"
    #mechanism_file = "uiuc_3sp_phenol"
    mechanism_file = "air_3sp"

    # wall stuff
    wall_time_scale = 1.0

    theta_factor = 1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    dim = 1

    from mirgecom.integrators.ssprk import ssprk43_step
    timestepper = ssprk43_step
    force_eval = True

    if rank == 0:
        print("\n #### Simulation control data: ####")
        print(f"\t nviz = {nviz}")
        print(f"\t nrestart = {nrestart}")
        print(f"\t nhealth = {nhealth}")
        print(f"\t nstatus = {nstatus}")
        print(f"\t maximum_sample_dt = {maximum_sample_dt}")
        print(f"\t t_final = {t_final}")
        print(f"\t order = {order}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    restart_step = None
    if restart_file is None:

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0

        nel_1d = 200

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={"right": ["+x"], "left": ["-x"]})

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        first_step = restart_step+0

        local_nelements = global_nelements

        assert comm.Get_size() == restart_data["num_parts"]

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD, DD_VOLUME_ALL
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else DISCR_TAG_BASE
    dd_vol_sample = DD_VOLUME_ALL

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    sample_nodes = actx.thaw(dcoll.nodes(dd_vol_sample))
    sample_zeros = actx.np.zeros_like(sample_nodes[0])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # {{{  Set up initial state using Cantera

    from mirgecom.transport import (
        SimpleTransport, MixtureAveragedTransport)
    import cantera
    from mirgecom.eos import PyrometheusMixture

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    x_cantera = np.zeros(nspecies)
    if my_material == "fiber":
        x_cantera[cantera_soln.species_index("O2")] = 0.21
        x_cantera[cantera_soln.species_index("N2")] = 0.79
    if my_material == "composite":
        x_cantera[cantera_soln.species_index("X2")] = 1.0
    cantera_soln.TPX = 900.0, 2000.0, x_cantera

    y_atmosphere = np.zeros(nspecies)
    _, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    cantera_soln.equilibrate("TP")
    temp_atmosphere, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    pres_atmosphere = cantera_soln.P
    mmw_atmosphere = cantera_soln.mean_molecular_weight

    x_products = np.zeros(nspecies)
    if my_material == "fiber":
        x_products[cantera_soln.species_index("CO2")] = 0.21
        x_products[cantera_soln.species_index("N2")] = 0.79

    cantera_soln.TPX = 900.0, 2000.0, x_products
    _, _, y_products = cantera_soln.TDY

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
                                cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 900.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    species_names = pyrometheus_mechanism.species_names

    print(f"Pyrometheus mechanism species names {species_names}")

    print(f"Atmosphere:")
    print(f"T = {temp_atmosphere}")
    print(f"D = {rho_atmosphere}")
    print(f"Y_atm = {y_atmosphere}")
    print(f"Y_pro = {y_products}")
    print(f"W = {mmw_atmosphere}")

    # }}}
    
    # {{{ Initialize transport model

    visc = 4.0e-5
    kappa = 0.06
    diff = 6.0e-3
    base_transport = SimpleTransport(viscosity=visc, thermal_conductivity=kappa,
                                     species_diffusivity=diff*np.ones(nspecies,))
    sample_transport = PorousWallTransport(base_transport=base_transport)

    # }}}

    # ~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    if my_material == "fiber":
        import mirgecom.materials.carbon_fiber as material_sample
        material = FiberEOS(dim=dim, char_mass=0.0, virgin_mass=1680.0/1e4*0.12, 
                            anisotropic_direction=0)
        decomposition = material_sample.Y3_Oxidation_Model(wall_material=material,
                                                           arrhenius=1e4,
                                                           activation_energy=-120000.0)

    # }}}

    # ~~~~~~~~~~~~~~

    gas_model_sample = PorousFlowModel(eos=eos, transport=sample_transport,
                                       wall_eos=material)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter

    from grudge.discretization import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_MODAL
    from meshmode.transform_metadata import FirstAxisIsElementsTag

    def _limit_sample_cv(cv, wv, temperature_seed, gas_model, dd=None):
        temperature = gas_model.get_temperature(cv=cv, wv=wv, tseed=temperature_seed)
        pressure = gas_model.get_pressure(cv, wv, temperature)
        return make_obj_array([cv, pressure, temperature])

#        spec_lim = make_obj_array([cv.mass*0.0, cv.mass*0.0, cv.mass*0.0+1.0])

#        mass_lim = wv.void_fraction*gas_model_sample.eos.get_density(
#            pressure=pressure,
#            temperature=temperature, species_mass_fractions=spec_lim)

#        velocity = cv.velocity
#        mom_lim = mass_lim*velocity

#        # recompute energy
#        energy_gas = mass_lim*(gas_model_sample.eos.get_internal_energy(
#            temperature, species_mass_fractions=spec_lim)
#            + 0.5*np.dot(velocity, velocity)
#        )

#        energy_solid = \
#            wv.density*gas_model_sample.wall_eos.enthalpy(temperature, wv.tau)

#        energy = energy_gas + energy_solid

#        # make a new CV with the limited variables
#        return make_conserved(dim=dim, mass=mass_lim, energy=energy,
#            momentum=mom_lim, species_mass=mass_lim*spec_lim)

    def _get_sample_state(cv, wv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_sample,
            material_densities=wv, temperature_seed=temp_seed,
            limiter_func=_limit_sample_cv, limiter_dd=dd_vol_sample
        )

    get_sample_state = actx.compile(_get_sample_state)

##############################################################################

    idx_CO2 = cantera_soln.species_index("CO2")
    idx_O2 = cantera_soln.species_index("O2")
    idx_N2 = cantera_soln.species_index("N2")

    if my_material == "fiber":

        sigma = 0.002
        H = 0.02
        porosity = 0.88
        x = sample_nodes[0]
        plug = 0.5*(actx.np.tanh(1.0/sigma*(x+H)) - actx.np.tanh(1.0/sigma*(x-H)))
        eps_gas = 1.0 - (1.0-porosity)*plug

        interface_location = -3.0*H
        species_smooth = 0.5*(1.0 + actx.np.tanh((x-interface_location)/(sigma*3)))

        x_initial = make_obj_array([sample_zeros, sample_zeros, sample_zeros])
        x_initial[idx_CO2] = 0.21*species_smooth
        x_initial[idx_O2] = x_cantera[idx_O2]*(1.0 - species_smooth)
        x_initial[idx_N2] = 1.0 - x_initial[idx_CO2] - x_initial[idx_O2]

        y_initial = eos.get_mass_fractions(x_initial)

        # soln setup and init
        _sample_density = 1680./1e4*0.12 + sample_zeros

    pressure = 2000.0 + sample_zeros
    temperature = 900.0 + sample_zeros

##############################################################################

    def initializer(dim, gas_model, material_densities, temperature,
                    epsilon_gas, gas_density=None, pressure=None):
        """Initialize material state."""
        if gas_density is None and pressure is None:
            raise ValueError("Must specify one of 'gas_density' or 'pressure'")

        if not isinstance(temperature, DOFArray):
            raise TypeError("Temperature does not have the proper shape")

        actx = temperature.array_context

        tau = gas_model.decomposition_progress(material_densities)*plug
        eps_rho_solid = material_densities*plug

        zeros = actx.np.zeros_like(tau)

        gas_const = gas_model.eos.gas_const(cv=None, temperature=temperature,
                                            species_mass_fractions=y_initial)

        if gas_density is None:
            eps_rho_gas = epsilon_gas*pressure/(gas_const*temperature)
       
        velocity = actx.np.zeros_like(sample_nodes)
        velocity[0] = 0.1/epsilon_gas
        momentum = eps_rho_gas*velocity

        gas_energy = eps_rho_gas * (
            gas_model.eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y_initial)
            + 0.5 * np.dot(velocity, velocity))
        solid_energy = eps_rho_solid*gas_model.wall_eos.enthalpy(temperature, tau)

        bulk_energy = gas_energy + solid_energy

#        print(np.min(actx.to_numpy(eps_rho_gas)))
#        print(np.max(actx.to_numpy(eps_rho_gas)))
#        print(np.min(actx.to_numpy(eps_rho_gas))/np.max(actx.to_numpy(eps_rho_gas)))
#        print(np.min(actx.to_numpy(velocity[0])))
#        print(np.max(actx.to_numpy(velocity[0])))
#        print(np.min(actx.to_numpy(momentum[0])))
#        print(np.max(actx.to_numpy(momentum[0])))
#        print(np.min(actx.to_numpy(temperature)))
#        print(np.max(actx.to_numpy(temperature)))
#        print(np.min(actx.to_numpy(pressure)))
#        print(np.max(actx.to_numpy(pressure)))
#        print(np.min(actx.to_numpy(eps_rho_solid)))
#        print(np.max(actx.to_numpy(eps_rho_solid)))
#        print("")

        cv = ConservedVars(mass=eps_rho_gas, energy=bulk_energy, momentum=momentum,
                           species_mass=eps_rho_gas*y_initial)

        return cv, eps_rho_solid
            
    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")

        sample_tseed = temperature*1.0

        velocity = np.zeros(dim,)
        velocity[0] = 0.1
        from mirgecom.materials.initializer import PorousWallInitializer
        sample_init = PorousWallInitializer(
            pressure=pressure, temperature=temperature, velocity=velocity,
            species_mass_fractions=y_initial, material_densities=_sample_density,
            porous_region=plug)
        sample_cv, sample_density = sample_init(sample_nodes, gas_model_sample)

#        sample_cv, sample_density = initializer(dim=dim, gas_model=gas_model_sample,
#            material_densities=_sample_density, epsilon_gas=eps_gas,
#            pressure=pressure, temperature=temperature)

#        print(np.min(actx.to_numpy(eps_gas)))
#        print(np.max(actx.to_numpy(eps_gas)))        
#        print(np.min(actx.to_numpy(sample_cv.mass)))
#        print(np.max(actx.to_numpy(sample_cv.mass)))
#        print(np.min(actx.to_numpy(sample_cv.mass))/np.max(actx.to_numpy(sample_cv.mass)))

    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t[0]))

        if rank == 0:
            logger.info("Restarting soln.")

        if restart_order != order:
            restart_dcoll = create_discretization_collection(actx, local_mesh, order=restart_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            sample_connection = make_same_mesh_connection(
                actx, dcoll.discr_from_dd(dd_vol_sample),
                restart_dcoll.discr_from_dd(dd_vol_sample))

            sample_cv = sample_connection(restart_data["sample_cv"])
            sample_tseed = sample_connection(restart_data["sample_temperature_seed"])
            sample_density = sample_connection(restart_data["sample_density"])
        else:
            sample_cv = restart_data["sample_cv"]
            sample_tseed = restart_data["sample_temperature_seed"]
            sample_density = restart_data["sample_density"]

#####################################

    sample_cv = force_evaluation(actx, sample_cv)
    sample_tseed = force_evaluation(actx, sample_tseed)
    sample_density = force_evaluation(actx, sample_density)
    sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)
    sample_cv = sample_state.cv

#####################################

    sponge_sigma_1 = (0.5*(1.0-actx.np.tanh(1.0/0.02*(sample_nodes[0]+0.3))))
#    sponge_sigma_2 = (0.5*(1.0+actx.np.tanh(1.0/0.02*(sample_nodes[0]-0.3))))

##############################################################################

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                          
    from contextlib import nullcontext
    gc_timer = nullcontext()
     
    vis_timer = None
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.5e} s, "),
            ("t_sim.max", "sim time: {value:1.5e} s, "),
            ("t_step.max", "--- step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max",
                                "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer_init = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer_init)
        gc_timer = gc_timer_init.get_sub_timer()

##############################################################################

    initname = original_casename
    eosname = gas_model_sample.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=maximum_sample_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=maximum_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

##############################################################################

    def get_sponge_rhs(cv, wv):

        velocity_ref = make_obj_array([0.1+sample_zeros])
        momentum_ref = velocity_ref*cv.mass
        int_energy = eos.internal_energy(cv)
        kin_energy = 0.5*cv.mass*np.dot(velocity_ref, velocity_ref)
        energy_ref = int_energy + kin_energy

        return cv.replace(energy=energy_ref, momentum=momentum_ref)

    sample_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_sample)

    def my_write_viz(step, t, dt, sample_state):
        if rank == 0:
            print('Writing solution file...')

        cv = sample_state.cv
        wv = sample_state.wv

        sample_mass_rhs, _, _ = decomposition.get_source_terms(
            temperature=sample_state.temperature,
            tau=sample_state.wv.tau,
            rhoY_o2=sample_state.cv.species_mass[idx_O2])

        reference_cv = get_sponge_rhs(cv, wv)
        sponge_rhs_1 = sponge_func(cv=cv, cv_ref=reference_cv, sigma=sponge_sigma_1*1000.0)

        gas_energy = cv.mass * (
            gas_model_sample.eos.get_internal_energy(
                temperature=sample_state.temperature,
                species_mass_fractions=sample_state.species_mass_fractions)
            + 0.5 * np.dot(sample_state.velocity, sample_state.velocity))
        solid_energy = wv.density * (
            gas_model_sample.wall_eos.enthalpy(sample_state.temperature,
                                               sample_state.wv.tau))

        sample_viz_fields = [
            ("x", sample_nodes[0]),
            ("CV", cv),
            ("plug", plug),
            ("pressure", sample_state.pressure),
            ("dp", sample_state.pressure - 2000.0),
            ("temperature", sample_state.temperature),
            ("Vx", sample_state.velocity[0]),
            ("WV_void_fraction", wv.void_fraction),
            ("WV_progress", wv.tau),
            ("WV_permeability", wv.permeability),
            ("WV_density", wv.density),
            ("rhoE_g", gas_energy),
            ("rhoE_s", solid_energy),
            ("rhoE_t", gas_energy + solid_energy),
            #("sponge_sigma_1", sponge_sigma_1),  # FIXME
            #("sponge_rhs", sponge_rhs_1),  # FIXME
            #("reference_cv", reference_cv - cv),  # FIXME
            #("kappa", sample_state.tv.thermal_conductivity),
            #("mu", sample_state.tv.viscosity),
            #("c", sample_state.dv.speed_of_sound),
            #("rhs_sample", rhs[0]),
            #("sample_mass_rhs_0", sample_mass_rhs[0]),
            #("sample_mass_rhs_1", sample_mass_rhs[1]),
            #("sample_mass_rhs_2", sample_mass_rhs[2]),
        ]

        # species mass fractions
        sample_viz_fields.extend(
            ("Y_"+species_names[i], sample_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        mol_weights = eos.get_species_molecular_weights()
        mmw = pyrometheus_mechanism.get_mix_molecular_weight(sample_state.cv.species_mass_fractions)
        sample_viz_fields.extend(
            ("X_"+species_names[i], sample_state.cv.species_mass_fractions[i]/mol_weights[i]*mmw)
                for i in range(nspecies))

        write_visfile(dcoll, sample_viz_fields, sample_visualizer,
            vizname=vizname+"-sample", step=step, t=t, overwrite=True, comm=comm)

#        sys.exit()

    def my_write_restart(step, t, sample_state):
        if rank == 0:
            print('Writing restart file...')

        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "local_mesh": local_mesh,
                "sample_cv": sample_state.cv,
                "sample_density": sample_state.wv.material_densities,
                "sample_temperature_seed": sample_state.dv.temperature,
                "nspecies": nspecies,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

##########################################################################

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_naninf_local(dcoll, "vol", temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

##############################################################################

    from grudge.dt_utils import characteristic_lengthscales
    char_length_solid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_sample)

    #~~~~~~~~~~

    def my_get_wall_timestep(state):
        wdv = sample_state.wv
        wall_mass = wdv.density
        wall_conductivity = material.thermal_conductivity(
            temperature=state.dv.temperature, tau=wdv.tau)
        wall_heat_capacity = material.heat_capacity(
            temperature=state.dv.temperature, tau=wdv.tau)
        wall_diffusivity = wall_conductivity/(wall_mass * wall_heat_capacity)

        return char_length_solid**2/(wall_time_scale * wall_diffusivity)

    def _my_get_timestep_sample(state, t, dt):
        actx = state.cv.mass.array_context
        if local_dt:
            mydt = maximum_cfl*my_get_wall_timestep(state)
        else:
            if constant_cfl:
                ts_field = maximum_cfl*my_get_wall_timestep(state, wdv)
                mydt = actx.to_numpy(
                    nodal_min(dcoll, dd_vol_sample, ts_field, initial=np.inf))[()]

        return mydt

    my_get_timestep_sample = actx.compile(_my_get_timestep_sample)

##############################################################################

#    # FIXME
#    ref_cv, _ = initializer(dim=dim, gas_model=gas_model_sample,
#        material_densities=sample_density, epsilon_gas=eps_gas,
#        pressure=pressure, temperature=temperature)

#    inflow_cv_cond = op.project(dcoll, dd_vol_sample, dd_vol_sample.trace("left"), ref_cv)

#    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
#        material_dens = state_minus.wv.material_densities
#        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
#            temperature_seed=900.0, material_densities=material_dens)

    from mirgecom.boundary import MengaldoBoundaryCondition
    from mirgecom.inviscid import inviscid_flux
    from mirgecom.flux import num_flux_central
    from mirgecom.viscous import viscous_flux
    from mirgecom.flux import num_flux_lfr

    """ """
    class MyPrescribedBoundary(MengaldoBoundaryCondition):
        r"""Prescribed my boundary function."""

        def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            cv_minus = state_minus.cv

            mom_plus = (2.0*0.1 - state_minus.cv.velocity)*state_minus.cv.mass
            #t_plus = 900.0 + state_minus.temperature*0.0
            t_plus = state_minus.temperature
            y_plus = y_atmosphere + state_minus.cv.species_mass*0.0

            energy_plus = cv_minus.mass*(
                gas_model.eos.get_internal_energy(
                    temperature=t_plus,
                    species_mass_fractions=y_plus)) \
                + 0.5*np.dot(mom_plus, mom_plus)/cv_minus.mass

            cv_plus = make_conserved(dim=dim, mass=cv_minus.mass,
                                     energy=energy_plus, momentum=mom_plus,
                                     species_mass=cv_minus.mass*y_plus)

            return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                    temperature_seed=t_plus,
                                    material_densities=state_minus.wv.material_densities)

        def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            cv_minus = state_minus.cv

            mom_bc = 0.1*state_minus.cv.mass + state_minus.cv.momentum*0.0
            # t_bc = 900.0 + state_minus.temperature*0.0
            t_bc = state_minus.temperature
            y_bc = y_atmosphere + state_minus.cv.species_mass*0.0

            energy_bc = cv_minus.mass*(
                gas_model.eos.get_internal_energy(
                    temperature=t_bc,
                    species_mass_fractions=y_bc)) \
                + 0.5*np.dot(mom_bc, mom_bc)/cv_minus.mass

            cv_bc = make_conserved(dim=dim, mass=cv_minus.mass,
                                   energy=energy_bc, momentum=mom_bc,
                                   species_mass=cv_minus.mass*y_bc)

            return make_fluid_state(cv=cv_bc, gas_model=gas_model,
                                    temperature_seed=t_bc,
                                    material_densities=state_minus.wv.material_densities)

        def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
            """Return farfield temperature for use in grad(temperature)."""
            #return 900.0 + state_minus.temperature*0.0
            return state_minus.temperature

        def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                       normal, **kwargs):
            """Return grad(CV) to be used in the boundary calculation of viscous flux."""
            return grad_cv_minus

        def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
            """Return grad(temperature) to be used in viscous flux at wall."""
            return grad_t_minus - np.dot(grad_t_minus, normal)*normal

    inlet_bndry = MyPrescribedBoundary()

    outlet_bndry = PressureOutflowBoundary(boundary_pressure=2000.0)
#    ref_mass = 2000.0/(eos.gas_const(species_mass_fractions=y_products)*900.0)
#    outlet_bndry = LinearizedOutflowBoundary(
#        free_stream_velocity=make_obj_array([0.1]),
#        free_stream_pressure=2000.0,
#        free_stream_density=ref_mass,
#        free_stream_species_mass_fractions=y_products)

    sample_boundaries = {dd_vol_sample.trace("right").domain_tag: outlet_bndry,
                         dd_vol_sample.trace("left").domain_tag: inlet_bndry}

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        sample_cv, sample_tseed, sample_density = state

        sample_cv = force_evaluation(actx, sample_cv)
        sample_tseed = force_evaluation(actx, sample_tseed)
        sample_density = force_evaluation(actx, sample_density)

        # construct species-limited solid state
        sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)
        sample_cv = sample_state.cv

        try:
            state = make_obj_array([
                sample_cv, sample_state.temperature, sample_state.wv.material_densities])

            do_garbage = check_step(step=step, interval=ngarbage)
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_garbage:
                with gc_timer:
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            file_exists = os.path.exists('write_solution')
            if file_exists:
              os.system('rm write_solution')
              do_viz = True
        
            file_exists = os.path.exists('write_restart')
            if file_exists:
              os.system('rm write_restart')
              do_restart = True

            if do_health:
                health_errors = global_reduce(
                    my_health_check(sample_state.cv, sample_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, sample_state=sample_state)

            if do_restart:
                my_write_restart(step, t, sample_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, sample_state=sample_state)
            raise

        return state, dt

#    # ~~~~~~~
#    def darcy_source_terms(cv, tv, wv):
#        """Source term to mimic Darcy's law."""
#        return make_conserved(dim=dim,
#            mass=fluid_zeros,
#            energy=fluid_zeros,
#            momentum=(
#                -1.0 * tv.viscosity * wv.void_fraction/wv.permeability *
#                cv.velocity),
#            species_mass=cv.species_mass*0.0)

    # ~~~~~~~
    from arraycontext import outer
    from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
    from meshmode.discretization.connection import FACE_RESTR_ALL

    #from grudge.geometry.metrics import normal as normal_vector
    from grudge.geometry import normal as normal_vector

    def my_derivative_function(actx, dcoll, field, field_bounds, dd_vol,
                               bnd_cond, comm_tag):    

        dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

        interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll,
                                      quadrature_tag)

        def interior_flux(field_tpair):
            dd_trace_quad = field_tpair.dd.with_discr_tag(quadrature_tag)
            normal_quad = normal_vector(actx, dcoll, dd_trace_quad)
            bnd_tpair_quad = interp_to_surf_quad(field_tpair)
            flux_int = outer(num_flux_central(bnd_tpair_quad.int,
                                              bnd_tpair_quad.ext),
                             normal_quad)

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad, flux_int)

        def boundary_flux(bdtag, bdry):
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = normal_vector(actx, dcoll, dd_bdry_quad)
            int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field)

            if bnd_cond == "symmetry" and bdtag == "-0":
                ext_soln_quad = 0.0*int_soln_quad
            else:
                ext_soln_quad = 1.0*int_soln_quad

            bnd_tpair = TracePair(
                dd_bdry_quad, interior=int_soln_quad, exterior=ext_soln_quad)
            flux_bnd = outer(num_flux_central(bnd_tpair.int, bnd_tpair.ext),
                             normal_quad)
        
            return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad, flux_bnd)

        return -op.inverse_mass(
            dcoll, dd_vol,
            op.weak_local_grad(dcoll, dd_vol, field)
            - op.face_mass(dcoll, dd_allfaces_quad,
                (sum(interior_flux(u_tpair) for u_tpair in
                    interior_trace_pairs(dcoll, field, volume_dd=dd_vol,
                                         comm_tag=comm_tag))
                + sum(boundary_flux(bdtag, bdry) for bdtag, bdry in
                    field_bounds.items())
                )
            )
        )


    my_boundaries = {dd_vol_sample.trace("right").domain_tag: DummyBoundary(),
                     dd_vol_sample.trace("left").domain_tag: DummyBoundary()}

    def radiation_sink_terms(temperature, epsilon):
        """Radiation sink term"""
        radiation_boundaries = normalize_boundaries(my_boundaries)
        grad_epsilon = my_derivative_function(
            actx, dcoll, epsilon, radiation_boundaries, dd_vol_sample,
            "replicate", _RadiationTag)
        epsilon_0 = 1.0
        f_phi = actx.np.sqrt( grad_epsilon[0]**2 )
        
        return - 1.0*5.67e-8*(1.0/epsilon_0*f_phi)*(temperature**4 - 900.0**4)

    def _my_rhs(time, actual_state):

        sample_state, sample_density = actual_state

        tv = sample_state.tv
        wv = sample_state.wv

        sample_operator_states_quad = make_operator_fluid_states(
            dcoll, sample_state, gas_model_sample, sample_boundaries,
            quadrature_tag, dd=dd_vol_sample, comm_tag=_SampleOpStatesTag,
            limiter_func=_limit_sample_cv)

        # sample grad CV
        sample_grad_cv = grad_cv_operator(
            dcoll, gas_model_sample, sample_boundaries, sample_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            comm_tag=_SampleGradCVTag)

        # sample grad T
        sample_grad_temperature = grad_t_operator(
            dcoll, gas_model_sample, sample_boundaries, sample_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            comm_tag=_SampleGradTempTag)

        sample_rhs = ns_operator(
            dcoll, gas_model_sample, sample_state, sample_boundaries,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            grad_cv=sample_grad_cv, grad_t=sample_grad_temperature,
            comm_tag=_SampleOperatorTag, inviscid_terms_on=True)

        # ~~~~
        sample_mass_rhs, m_dot_o2, m_dot_co2 = decomposition.get_source_terms(
            temperature=sample_state.temperature,
            tau=sample_state.wv.tau,
            rhoY_o2=sample_state.cv.species_mass[idx_O2])

        # ~~~~
        species_sources = sample_state.cv.species_mass*0.0
        species_sources[idx_CO2] = m_dot_co2
        species_sources[idx_O2] = m_dot_o2

        radiation = radiation_sink_terms(sample_state.temperature, sample_state.wv.void_fraction)

        darcy_momentum = -1.0*tv.viscosity/wv.permeability*wv.void_fraction*sample_cv.velocity
        darcy_energy = -1.0*tv.viscosity/wv.permeability*(wv.void_fraction)**2*np.dot(sample_cv.velocity, sample_cv.velocity)
        darcy_source = make_conserved(
            dim=dim,
            mass=m_dot_o2+m_dot_co2,
            energy=darcy_energy+radiation,
            momentum=darcy_momentum,
            species_mass=species_sources)

#        sample_mass_rhs = sample_zeros
#        species_sources = sample_state.cv.species_mass*0.0
#        darcy_momentum = -1.0*tv.viscosity/wv.permeability*wv.void_fraction*sample_cv.velocity
#        darcy_energy = -1.0*tv.viscosity/wv.permeability*(wv.void_fraction)**2*np.dot(sample_cv.velocity, sample_cv.velocity)
#        darcy_source = make_conserved(
#            dim=dim,
#            mass=sample_zeros,
#            energy=darcy_energy,
#            momentum=darcy_momentum,
#            species_mass=species_sources)

        #~~~~~~~~~~~~~
        return make_obj_array([sample_rhs + darcy_source,
                               sample_zeros, sample_mass_rhs])

    compiled_rhs = actx.compile(_my_rhs)

    def my_rhs(t, state):

        sample_cv, sample_tseed, sample_density = state

        sample_cv = force_evaluation(actx, sample_cv)
        sample_tseed = force_evaluation(actx, sample_tseed)
        sample_density = force_evaluation(actx, sample_density)

        sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)

        actual_state = make_obj_array([sample_state, sample_density])

        return compiled_rhs(t, actual_state)

    def my_post_step(step, t, dt, state):
        if step == first_step + 1:
            with gc_timer:
                import gc
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections")
                gc.freeze()

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([
        sample_state.cv, sample_state.temperature, sample_state.wv.material_densities])

    dt = maximum_sample_dt
    t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt, t=t, t_final=t_final,
                      force_eval=force_eval, state=stepper_state,
                      compile_rhs=False)

#    # 
#    final_cv, tseed, final_wv, wv_tseed = stepper_state
#    final_state = get_fluid_state(final_cv, tseed)


#    # Dump the final data
#    if rank == 0:
#        logger.info("Checkpointing final state ...")

#    my_write_restart(step=final_step, t=final_t, state=stepper_state)

#    my_write_viz(step=final_step, t=final_t, dt=current_dt,
#                 cv=final_state.cv, dv=current_state.dv,
#                 wv=final_wv, wdv=final_wdv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profiling", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")

    args = parser.parse_args()

    # for writing output
    casename = "burner_mix"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    main(actx_class, use_logmgr=args.log, casename=casename, 
         restart_filename=restart_file)
