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
import os
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
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, DISCR_TAG_BASE, BoundaryDomainTag, VolumeDomainTag
)

from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step, get_sim_timestep, distribute_mesh, write_visfile,
    check_naninf_local, check_range_local, global_reduce, get_box_mesh
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PressureOutflowBoundary,
    AdiabaticSlipBoundary,
    FarfieldBoundary,
    PrescribedFluidBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import SimpleTransport, PowerLawTransport
import cantera
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    make_operator_fluid_states
)
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)
from mirgecom.navierstokes import (
    grad_t_operator,
    grad_cv_operator,
    ns_operator
)
from mirgecom.multiphysics.multiphysics_coupled_fluid_wall import (
    add_interface_boundaries as add_multiphysics_interface_boundaries,
    add_interface_boundaries_no_grad as add_multiphysics_interface_boundaries_no_grad
)

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array

#########################################################################

class _FluidGradCVTag:
    pass

class _FluidGradTempTag:
    pass

class _FluidOperatorTag:
    pass

class _FluidOpStatesTag:
    pass

class _SolidGradCVTag:
    pass

class _SolidGradTempTag:
    pass

class _SolidOperatorTag:
    pass

class _SolidOpStatesTag:
    pass


class Initializer:

    def __init__(self, *, dim, pressure, temperature, species_atm):

        self._dim = dim
        self._pres = pressure
        self._ya = species_atm
        self._temp = temperature     

    def __call__(self, actx, x_vec, eos):

        zeros = actx.np.zeros_like(x_vec[0])

        temperature = self._temp + zeros
        pressure = self._pres + zeros
        y = self._ya + zeros

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)

        velocity = make_obj_array([zeros for i in range(self._dim)])
        momentum = velocity*mass

        #~~~ 
        specmass = mass * y

        #~~~ 
        internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                momentum=momentum, species_mass=specmass)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


#h1 = logging.StreamHandler(sys.stdout)
#f1 = SingleLevelFilter(logging.INFO, False)
#h1.addFilter(f1)
#root_logger = logging.getLogger()
#root_logger.addHandler(h1)
#h2 = logging.StreamHandler(sys.stderr)
#f2 = SingleLevelFilter(logging.INFO, True)
#h2.addFilter(f2)
#root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


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
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 1000
    nrestart = 25000
    nhealth = 1
    nstatus = 100
    ngarbage = 10

    # default timestepping control
    t_final = 40000.0
    dt = 4.0e-4

    constant_cfl = False
    current_cfl = 0.2
    
    # discretization
    order = 1

    my_mechanism = "uiuc_3sp_phenol"

    # wall stuff
    temp_wall = 300.0

    wall_penalty_amount = 1.0
    wall_time_scale = 1.0

    emissivity = 0.85

#    print('397.95645223914176')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    dim = 2

    first_step = 0
    current_step = 0
    current_t = 0.0
    current_dt = dt

    from mirgecom.integrators.ssprk import ssprk43_step
    timestepper = ssprk43_step
    force_eval = True

    if rank == 0:
        print("\n #### Simulation control data: ####")
        print(f"\t nviz = {nviz}")
        print(f"\t nrestart = {nrestart}")
        print(f"\t nhealth = {nhealth}")
        print(f"\t nstatus = {nstatus}")
        print(f"\t current_dt = {current_dt}")
        print(f"\t t_final = {t_final}")
        print(f"\t order = {order}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mesh_filename = "phenolics-v2.msh"
    def get_mesh_data():
        from meshmode.mesh.io import read_gmsh
        mesh, tag_to_elements = read_gmsh(
            mesh_filename, force_ambient_dim=dim,
            return_tag_to_elements_map=True)
        volume_to_tags = {
            "fluid": ["Fluid"],
            "solid": ["Sample"]
            }
        return mesh, tag_to_elements, volume_to_tags

    volume_to_local_mesh_data, global_nelements = distribute_mesh(
        comm, get_mesh_data)

    local_nelements = (volume_to_local_mesh_data["fluid"][0].nelements
                       + volume_to_local_mesh_data["solid"][0].nelements)

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order)

    quadrature_tag = None

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("solid"), DISCR_TAG_BASE)

    fluid_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_solid))

    fluid_zeros = force_evaluation(actx, fluid_nodes[0]*0.0)
    solid_zeros = force_evaluation(actx, solid_nodes[0]*0.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    current_path = os.path.abspath(os.getcwd()) + "/"
    mechanism_file = current_path + my_mechanism

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    x = np.zeros(nspecies)
    x[cantera_soln.species_index("O2")] = 0.21
    x[cantera_soln.species_index("N2")] = 0.79
    cantera_soln.TPX = 300.0, 101325.0, x

    y_atmosphere = np.zeros(nspecies)
    dummy, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    cantera_soln.equilibrate("TP")
    temp_atmosphere, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    pres_atmosphere = cantera_soln.P
    mmw_atmosphere = cantera_soln.mean_molecular_weight

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
                                cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 1000.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    species_names = pyrometheus_mechanism.species_names

    print(f"Pyrometheus mechanism species names {species_names}")

    print(f"Atmosphere:")
    print(f"T = {temp_atmosphere}")
    print(f"D = {rho_atmosphere}")
    print(f"Y = {y_atmosphere}")
    print(f"W = {mmw_atmosphere}")

    # }}}
    
    # {{{ Initialize transport model

    physical_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.01,
                                         species_diffusivity=0.00001*np.ones(nspecies,))

    base_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.1,
                                     species_diffusivity=0.00001*np.ones(nspecies,))
    from mirgecom.wall_model import PorousWallTransport
    solid_transport = PorousWallTransport(base_transport=base_transport)

    # }}}

    # ~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    import mirgecom.materials.tacot as material_solid
    material = material_solid.TacotEOS(char_mass=220., virgin_mass=280.)
    decomposition = material_solid.Pyrolysis()   

    # }}}

    # ~~~~~~~~~~~~~~

    gas_model_fluid = GasModel(eos=eos, transport=physical_transport)

    from mirgecom.wall_model import PorousFlowModel
    gas_model_solid = PorousFlowModel(eos=eos, transport=solid_transport,
                                      wall_eos=material)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model_fluid.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

    # ~~~~~~~~~~

    def _limit_solid_cv(cv, wv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)])

        # normalize to ensure sum_Yi = 1.0
        aux = actx.np.zeros_like(cv.mass)
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        mass_lim = wv.void_fraction*gas_model_solid.eos.get_density(
            pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        velocity = cv.momentum/cv.mass
        mom_lim = mass_lim*velocity

        # recompute energy
        energy_gas = mass_lim*(gas_model_solid.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(velocity, velocity))

        energy_solid = \
            wv.density*gas_model_solid.wall_eos.enthalpy(temperature, wv.tau)

        energy = energy_gas + energy_solid

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy,
            momentum=mom_lim, species_mass=mass_lim*spec_lim)

##############################################################################

    fluid_init = Initializer(dim=dim, pressure=101325.0,
                             temperature=300.0 + fluid_nodes[1]/0.01*1700.0,
                             species_atm=y_atmosphere)

    # ~~~~~~~~~~~

    # soln setup and init
    solid_density = np.empty((3,), dtype=object)

    solid_density[0] = 30.0 + solid_zeros
    solid_density[1] = 90.0 + solid_zeros
    solid_density[2] = 160. + solid_zeros

    from mirgecom.materials.initializer import PorousWallInitializer

#    y_sample = y_atmosphere + solid_zeros

    idx_O2 = cantera_soln.species_index("O2")
    idx_N2 = cantera_soln.species_index("N2")
    idx_X2 = cantera_soln.species_index("X2")

    y_sample = y_atmosphere*0.0 + solid_zeros
    y_sample[idx_O2] = y_atmosphere[idx_O2]*(solid_nodes[1]+0.05)/0.05
    y_sample[idx_N2] = y_atmosphere[idx_N2]*(solid_nodes[1]+0.05)/0.05
    y_sample[idx_X2] = 1.0 - (y_sample[idx_O2] + y_sample[idx_N2])

    solid_init = PorousWallInitializer(
        pressure=101325.0, temperature=300.0,
        species=y_sample, material_densities=solid_density)

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")

        fluid_tseed = temperature_seed*1.0
        solid_tseed = temp_wall*1.0

        fluid_cv = fluid_init(actx, fluid_nodes, eos)

        solid_cv = solid_init(dim, solid_nodes, gas_model_solid)

    else:
        current_step = restart_step
        current_t = restart_data["t"]

        if rank == 0:
            logger.info("Restarting soln.")

        fluid_cv = restart_data["fluid_cv"]
        fluid_tseed = restart_data["fluid_temperature_seed"]
        solid_cv = restart_data["solid_cv"]
        solid_tseed = restart_data["solid_temperature_seed"]
        solid_density = restart_data["solid_density"]


    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_fluid,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

    get_fluid_state = actx.compile(_get_fluid_state)

    def _get_solid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_solid,
            material_densities=solid_density, temperature_seed=temp_seed,
            limiter_func=_limit_solid_cv, limiter_dd=dd_vol_solid)

    get_solid_state = actx.compile(_get_solid_state)

    fluid_cv = force_evaluation(actx, fluid_cv)
    fluid_tseed = force_evaluation(actx, fluid_tseed)
    fluid_state = get_fluid_state(fluid_cv, fluid_tseed)

    solid_cv = force_evaluation(actx, solid_cv)
    solid_tseed = force_evaluation(actx, solid_tseed)
    solid_density = force_evaluation(actx, solid_density)
    solid_state = get_solid_state(solid_cv, solid_tseed)

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

#    y_bnd = y_atmosphere*0.0
#    y_bnd[idx_O2] = y_atmosphere[idx_O2]
#    y_bnd[idx_N2] = y_atmosphere[idx_N2]
#    fluid_bnd_init = Initializer(dim=dim, pressure=101325.0,
#                                temperature=2000.0, species_atm=y_bnd)

#    def _boundary_solution(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
#        actx = state_minus.array_context
#        bnd_discr = dcoll.discr_from_dd(dd_bdry)
#        nodes = actx.thaw(bnd_discr.nodes())
#        return make_fluid_state(fluid_bnd_init(actx, nodes, eos), gas_model,
#                                temperature_seed=2000.0)

    fluid_boundaries = {
        dd_vol_fluid.trace("fluid_side").domain_tag:
            AdiabaticSlipBoundary(),
        dd_vol_fluid.trace("fluid_base").domain_tag:
#            PressureOutflowBoundary(boundary_pressure=101325.0),
            FarfieldBoundary(dim, 101325.0,
                             make_obj_array([0.0 for i in range(dim)]),
                             2000.0, y_atmosphere)
#            PrescribedFluidBoundary(
#                boundary_state_func=_boundary_solution),
    }

    y_sample = y_atmosphere*0.0
    y_sample[idx_X2] = 1.0

    solid_density_bnd = np.empty((3,), dtype=object)

    solid_density_bnd[0] = 30.0
    solid_density_bnd[1] = 90.0
    solid_density_bnd[2] = 160.
    solid_bnd_init = PorousWallInitializer(
        pressure=101325.0, temperature=300.0,
        species=y_sample, material_densities=solid_density_bnd)

    bnd_discr = dcoll.discr_from_dd(dd_vol_solid.trace("sample_base").domain_tag)
    sld_bnd_nodes = actx.thaw(bnd_discr.nodes())
    print(f"{sld_bnd_nodes=}")

    bnd_wv = solid_density_bnd + sld_bnd_nodes[0]*0.0
    print(f"{bnd_wv=}")

    bnd_cv = solid_bnd_init(dim, sld_bnd_nodes, gas_model_solid)
    bnd_cv = force_evaluation(actx, bnd_cv)
    print(f"{bnd_cv=}")

    # 1/0
    def _mksolidbndstate():
        return make_fluid_state(bnd_cv, gas_model_solid, material_densities=bnd_wv,
                                temperature_seed=300.0)

    make_solid_bnd_state = actx.compile(_mksolidbndstate)

    # bnd_state = make_fluid_state(bnd_cv, gas_model_solid, material_densities=bnd_wv,
    #                             temperature_seed=300.0)
    bnd_state = make_solid_bnd_state()

    # bnd_state = force_evaluation(actx, bnd_state)
    print(f"{bnd_state=}")
    # 1/0

    def _solid_bnd_solution(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return bnd_state

    # ~~~~~~~~~~
    solid_boundaries = {
        dd_vol_solid.trace("sample_side").domain_tag: AdiabaticSlipBoundary(),
        dd_vol_solid.trace("sample_base").domain_tag:
            PrescribedFluidBoundary(boundary_state_func=_solid_bnd_solution),
            # AdiabaticSlipBoundary(),
    }

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=current_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=current_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

##############################################################################

    def my_write_viz(step, t, dt, fluid_state, solid_state):
        if rank == 0:
            print('Writing solution file...')

        # ~~~~~~
        fluid_dt = _my_get_timestep_fluid(fluid_state, t, dt)
        fluid_viz_fields = [
            ("rho_g", fluid_state.cv.mass),
            ("rhoU_g", fluid_state.cv.momentum),
            ("rhoE_g", fluid_state.cv.energy),
            ("pressure", fluid_state.pressure),
            ("temperature", fluid_state.temperature),
            ("Vx", fluid_state.velocity[0]),
            ("fluid_dt", fluid_dt),
        ]

        # species mass fractions
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        # ~~~~~~
        solid_cv = solid_state.cv
        solid_dt = _my_get_timestep_sample(solid_state)
        solid_viz_fields = [
            ("rho_g", solid_cv.mass),
            ("rhoU_g", solid_cv.momentum),
            ("rhoE_b", solid_cv.energy),
            ("pressure", solid_state.pressure),
            ("temperature", solid_state.temperature),
            ("Vx", solid_state.velocity[0]),
            ("void_fraction", solid_state.wv.void_fraction),
            ("progress", solid_state.wv.tau),
            ("permeability", solid_state.wv.permeability),
            ("kappa", solid_state.thermal_conductivity),
            ("solid_dt", solid_dt),
        ]

#        solid_viz_fields.extend(
#            ("solid_mass_rhs" + str(i), solid_mass_rhs[i])
#                for i in range(len(solid_state.wv.material_densities)))
        solid_viz_fields.extend(
            ("solid_mass_" + str(i), solid_state.wv.material_densities[i])
                for i in range(len(solid_state.wv.material_densities)))

        # species mass fractions
        solid_viz_fields.extend(
            ("Y_"+species_names[i], solid_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        write_visfile(dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t, overwrite=True, comm=comm)

        write_visfile(dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-solid", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, fluid_state, solid_state):
        if rank == 0:
            print('Writing restart file...')

        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "fluid_cv": fluid_state.cv,
                "fluid_temperature_seed": fluid_state.temperature,
                "solid_cv": solid_state.cv,
                "solid_density": solid_state.wv.material_densities,
                "solid_temperature_seed": solid_state.dv.temperature,
                "nspecies": nspecies,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

##########################################################################

    from grudge.op import nodal_min
    from grudge.dt_utils import characteristic_lengthscales
    from mirgecom.viscous import get_local_max_species_diffusivity

    char_length_fluid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)
    char_length_solid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid)

    def _my_get_timestep_fluid(fluid_state, t, dt):
        return get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
            t_final=10000.0, constant_cfl=True, fluid_dd=dd_vol_fluid)

    def _my_get_timestep_sample(state):
        wv = state.wv
        wall_mass = gas_model_solid.solid_density(state.wv.material_densities)
        wall_conductivity = material.thermal_conductivity(
            temperature=state.dv.temperature, tau=wv.tau)
        wall_heat_capacity = material.heat_capacity(
            temperature=state.dv.temperature, tau=wv.tau)

        wall_heat_diffusivity = wall_conductivity/(wall_mass * wall_heat_capacity)
        wall_spec_diffusivity = get_local_max_species_diffusivity(actx, state.tv.species_diffusivity)
        wall_diffusivity = actx.np.maximum(wall_heat_diffusivity, wall_spec_diffusivity)

        timestep = char_length_solid**2/(wall_time_scale * wall_diffusivity)

        return actx.to_numpy(nodal_min(
                dcoll, dd_vol_solid, current_cfl*timestep, initial=np.inf))[()]

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(dcoll, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

##############################################################################

    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        fluid_cv, fluid_tseed, solid_cv, solid_tseed = state

        fluid_cv = force_evaluation(actx, fluid_cv)
        fluid_tseed = force_evaluation(actx, fluid_tseed)
        solid_cv = force_evaluation(actx, solid_cv)
        solid_tseed = force_evaluation(actx, solid_tseed)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(fluid_cv, fluid_tseed)
        fluid_cv = fluid_state.cv

        # construct species-limited solid state
        solid_state = get_solid_state(solid_cv, solid_tseed)
        solid_cv = solid_state.cv

        try:
            state = make_obj_array([fluid_cv, fluid_state.temperature,
                                    solid_cv, solid_state.temperature])

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
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt,
                    fluid_state=fluid_state, solid_state=solid_state)

            if do_restart:
                my_write_restart(step, t, fluid_state, solid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")

            my_write_viz(step=step, t=t, dt=dt,
                fluid_state=fluid_state, solid_state=solid_state)

            raise

        return state, dt


    from mirgecom.diffusion import(
        DirichletDiffusionBoundary, NeumannDiffusionBoundary,
        diffusion_facial_flux_central, grad_facial_flux_central,
        grad_operator as solid_grad_operator)

    def my_rhs(time, state):

        fluid_cv, fluid_tseed, solid_cv, solid_tseed = state

        # construct species-limited fluid state
        fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model_fluid,
            temperature_seed=fluid_tseed,
            limiter_func=_limit_fluid_cv, limiter_dd=dd_vol_fluid)
        fluid_cv = fluid_state.cv

        # construct species-limited solid state
        solid_state = make_fluid_state(cv=solid_cv, gas_model=gas_model_solid,
            material_densities=solid_density, temperature_seed=solid_tseed,
            limiter_func=_limit_solid_cv, limiter_dd=dd_vol_solid)
        solid_cv = solid_state.cv

        #~~~~~~~~~~~~~
        fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
            add_multiphysics_interface_boundaries_no_grad(
                dcoll, dd_vol_fluid, dd_vol_solid,
                gas_model_fluid, gas_model_solid,
                fluid_state, solid_state,
                fluid_boundaries, solid_boundaries,
                interface_noslip=True, interface_radiation=True)

        # ~~~~~~~~~~~~~~
        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model_fluid, fluid_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_fluid, comm_tag=_FluidOpStatesTag,
            limiter_func=_limit_fluid_cv)

        solid_operator_states_quad = make_operator_fluid_states(
            dcoll, solid_state, gas_model_solid, solid_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_solid, comm_tag=_SolidOpStatesTag,
            limiter_func=_limit_solid_cv)

        # ~~~~~~~~~~~~~~
        # fluid grad CV
        fluid_grad_cv = grad_cv_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradCVTag)

        # fluid grad T
        fluid_grad_temperature = grad_t_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradTempTag)

        # solid grad CV
        solid_grad_cv = grad_cv_operator(
            dcoll, gas_model_solid, solid_all_boundaries_no_grad, solid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_solid,
            operator_states_quad=solid_operator_states_quad,
            comm_tag=_SolidGradCVTag)

        # solid grad T
        solid_grad_temperature = grad_t_operator(
            dcoll, gas_model_solid, solid_all_boundaries_no_grad, solid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_solid,
            operator_states_quad=solid_operator_states_quad,
            comm_tag=_SolidGradTempTag)

        # ~~~~~~~~~~~~~~~~~
        fluid_all_boundaries, solid_all_boundaries = \
            add_multiphysics_interface_boundaries(
                dcoll, dd_vol_fluid, dd_vol_solid,
                gas_model_fluid, gas_model_solid,
                fluid_state, solid_state,
                fluid_grad_cv, solid_grad_cv,
                fluid_grad_temperature, solid_grad_temperature,
                fluid_boundaries, solid_boundaries,
                interface_noslip=True, interface_radiation=True,
                wall_emissivity=emissivity, sigma=5.67e-8, ambient_temperature=300.0,
                wall_penalty_amount=wall_penalty_amount)

        #~~~~~~~~~~~~~
        fluid_rhs = ns_operator(
            dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
            comm_tag=_FluidOperatorTag, inviscid_terms_on=False)

        #~~~~~~~~~~~~~
        solid_rhs = ns_operator(
            dcoll, gas_model_solid, solid_state, solid_all_boundaries,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_solid,
            operator_states_quad=solid_operator_states_quad,
            grad_cv=solid_grad_cv, grad_t=solid_grad_temperature,
            comm_tag=_SolidOperatorTag, inviscid_terms_on=False)

        #~~~~~~~~~~~~~
        return make_obj_array([fluid_rhs, fluid_zeros, solid_rhs, solid_zeros])


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

    stepper_state = make_obj_array([fluid_cv, fluid_state.temperature,
                                    solid_cv, solid_state.temperature])

    dt = current_dt
    t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt, t=t, t_final=t_final,
                      force_eval=force_eval, state=stepper_state)
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
