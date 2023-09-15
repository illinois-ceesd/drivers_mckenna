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
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, BoundaryDomainTag, VolumeDomainTag
)
from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step, get_sim_timestep, distribute_mesh, write_visfile,
    check_naninf_local, check_range_local, global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PressureOutflowBoundary,
    AdiabaticSlipBoundary,
)
from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient, make_conserved
)
from mirgecom.transport import (
    PowerLawTransport,
    MixtureAveragedTransport
)
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
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    add_interface_boundaries as add_thermal_interface_boundaries,
    add_interface_boundaries_no_grad as add_thermal_interface_boundaries_no_grad
)
from mirgecom.diffusion import (
    diffusion_operator,
    grad_operator as wall_grad_t_operator,
    NeumannDiffusionBoundary
)

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array

#########################################################################

class _FluidGradCVTag:
    pass

class _FluidGradTempTag:
    pass

class _SampleGradCVTag:
    pass

class _SampleGradTempTag:
    pass

class _FluidOperatorTag:
    pass

class _SampleOperatorTag:
    pass

class _FluidOpStatesTag:
    pass

class _WallOpStatesTag:
    pass

class FiberSampleInitializer:

    def __init__(self, pressure, temperature, species):

        self._pres = pressure
        self._y = species
        self._temp = temperature

    def __call__(self, actx, x_vec, gas_model, wall_density):

        eos = gas_model.eos
        zeros = x_vec[0]*0.0

        tau = zeros + 1.0

        velocity = make_obj_array([zeros, zeros])

        pressure = self._pres + zeros
        temperature = self._temp + zeros
        y = self._y + zeros

        int_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)

        epsilon = gas_model.wall.void_fraction(tau=tau)
        eps_rho_g = epsilon * mass
        eps_rhoU_g = eps_rho_g * velocity
        eps_rhoY_g = eps_rho_g * y

        eps_rho_s = wall_density + zeros
        enthalpy_s = gas_model.wall.enthalpy(temperature=temperature, tau=tau)

        energy = eps_rho_g * int_energy + eps_rho_s * enthalpy_s

        return make_conserved(dim=2, mass=eps_rho_g,
            momentum=eps_rhoU_g, energy=energy, species_mass=eps_rhoY_g)


class TACOTSampleInitializer:

    def __init__(self, pressure, temperature, species, wall_density):

        self._pres = pressure
        self._y = species
        self._temp = temperature
        self._wall_density = wall_density

    def __call__(self, actx, x_vec, gas_model):

        zeros = actx.np.zeros_like(x_vec[0])

        pressure = self._pres + zeros
        temperature = self._temp + zeros
        species_mass_frac = self._y + zeros

        tau = gas_model.wall.decomposition_progress(self._wall_density)

        eps_gas = gas_model.wall.void_fraction(tau)
        eps_rho_gas = eps_gas*gas_model.eos.get_density(pressure, temperature, species_mass_frac)

        # internal energy (kinetic energy is neglected)
        eps_rho_solid = sum(self._wall_density)
        bulk_energy = (
            eps_rho_solid*gas_model.wall.enthalpy(temperature, tau)
            + eps_rho_gas*gas_model.eos.get_internal_energy(temperature, species_mass_frac)
        )

        momentum = make_obj_array([zeros, zeros])

        species_mass = eps_rho_gas*species_mass_frac

        return make_conserved(dim=2, mass=eps_rho_gas, energy=bulk_energy,
            momentum=momentum, species_mass=species_mass)


class Initializer:

    def __init__(self, *, dim=2, nspecies=7, pressure, temperature, species_atm):

        self._dim = dim
        self._nspecies = nspecies
        self._pres = pressure
        self._ya = species_atm
        self._temp = temperature     

    def __call__(self, actx, x_vec, eos, solve_the_flame=True,
                 state_minus=None, time=None):

        zeros = actx.np.zeros_like(x_vec[0])

        y = zeros + self._ya
        temperature = zeros + 300.0
        pressure = self._pres + zeros

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)

        momentum = make_obj_array([zeros, zeros])
        velocity = momentum/mass

        #~~~ 
        specmass = mass * y

        #~~~ 
        internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                momentum=mass*velocity, species_mass=specmass)


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:

    def __init__(self, *, x_min=None, x_max=None, y_min=None, y_max=None,
                 x_thickness=None, y_thickness=None, amplitude):
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_thickness = x_thickness
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec):
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = 0*xpos

        sponge_x = xpos*0.0
        sponge_y = xpos*0.0

        if (self._y_max is not None):
          y0 = (self._y_max - self._y_thickness)
          dy = +((ypos - y0)/self._y_thickness)
          sponge_y = sponge_y + self._amplitude * actx.np.where(
              actx.np.greater(ypos, y0),
                  actx.np.where(actx.np.greater(ypos, self._y_max),
                                1.0, 3.0*dy**2 - 2.0*dy**3),
                  0.0
          )

        if (self._y_min is not None):
          y0 = (self._y_min + self._y_thickness)
          dy = -((ypos - y0)/self._y_thickness)
          sponge_y = sponge_y + self._amplitude * actx.np.where(
              actx.np.less(ypos, y0),
                  actx.np.where(actx.np.less(ypos, self._y_min),
                                1.0, 3.0*dy**2 - 2.0*dy**3),
              0.0
          )

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

        return actx.np.maximum(sponge_x,sponge_y)


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

    cl_ctx = ctx_factory()

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    # ~~~~~~~~~~~~~~~~~~

    mesh_filename = "phenolics-v2.msh"

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 500
    nrestart = 25000
    nhealth = 1
    nstatus = 100
    ngarbage = 10

    # default timestepping control
#    integrator = "compiled_lsrk45"
    integrator = "ssprk43"
    maximum_fluid_dt = 1.0e-6 #order == 2
    maximum_solid_dt = 1.0e-8
    t_final = 2.0

    niter = 4000001

    local_dt = True
    constant_cfl = True
    maximum_cfl = 0.2
    
    # discretization and model control
    order = 2
    use_overintegration = False

    x0_sponge = 0.30
    sponge_amp = 400.0
    theta_factor = 0.02
    speedup_factor = 1.0

#    my_material = "fiber"
    my_material = "composite"

    my_mechanism = "uiuc_7sp"

#    transport = "Mixture"
    transport = "PowerLaw"

    # wall stuff
    temp_wall = 300

    wall_penalty_amount = 1.0
    wall_time_scale = 10.0

    use_radiation = True
    emissivity = 0.85

    restart_iterations = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    dim = 2

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    if integrator == "compiled_lsrk45":
        from grudge.shortcuts import compiled_lsrk45_step
        timestepper = _compiled_stepper_wrapper
        force_eval = False

    if integrator == "ssprk43":
        from mirgecom.integrators.ssprk import ssprk43_step
        timestepper = ssprk43_step
        force_eval = True

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tmaximum_fluid_dt = {maximum_fluid_dt}")
        print(f"\tmaximum_solid_dt = {maximum_solid_dt}")
        if (constant_cfl == False):
            print(f"\tt_final = {t_final}")
        else:
            print(f"\tconstant_cfl = {constant_cfl}")
            print(f"\tmaximum_cfl = {maximum_cfl}")
            print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    restart_step = None
    if restart_file is None:

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0

        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "fluid": ["Fluid"],
                "sample": ["Sample"]
                }
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        first_step = restart_step+0

        assert comm.Get_size() == restart_data["num_parts"]

    local_nelements = (
          volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["sample"][0].nelements)

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_sample = DOFDesc(VolumeDomainTag("sample"), DISCR_TAG_BASE)

    fluid_nodes = force_evaluation(actx, dcoll.nodes(dd_vol_fluid))
    sample_nodes = force_evaluation(actx, dcoll.nodes(dd_vol_sample))

    fluid_zeros = force_evaluation(actx, fluid_nodes[0]*0.0)
    sample_zeros = force_evaluation(actx, sample_nodes[0]*0.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    import os
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

    if transport == "Mixture":
        physical_transport = MixtureAveragedTransport(
            pyrometheus_mechanism, lewis=np.ones(nspecies,), factor=speedup_factor)
    else:
        if transport == "PowerLaw":
            physical_transport = PowerLawTransport(lewis=np.ones(nspecies,),
               beta=4.093e-7*speedup_factor)
        else:
            print('No transport class defined..')
            print('Use one of "Mixture" or "PowerLaw"')
            sys.exit()

    # }}}

    # ~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    from mirgecom.wall_model import WallEOS, WallDependentVars
    if my_material == "fiber":
        import mirgecom.materials.carbon_fiber as material_sample
        material = material_sample.SolidProperties()
        decomposition = Y3_Oxidation_Model(wall_material=material)

    if my_material == "composite":
        import mirgecom.materials.tacot as material_sample
        material = material_sample.SolidProperties()    
        decomposition = material_sample.Pyrolysis()   

    sample_degradation_model = WallEOS(wall_material=material)

    # }}}

    # ~~~~~~~~~~~~~~

    gas_model_fluid = GasModel(eos=eos, transport=physical_transport)

    gas_model_sample = GasModel(eos=eos, wall=sample_degradation_model, 
                                transport=physical_transport)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reaction_damping(dcoll, nodes, **kwargs):
        ypos = nodes[1]

        y_max = 0.50
        y_thickness = 0.10

        y0 = (y_max - y_thickness)
        dy = +((ypos - y0)/y_thickness)
        return actx.np.where(
            actx.np.greater(ypos, y0),
                actx.np.where(actx.np.greater(ypos, y_max),
                              0.0, 1.0 - (3.0*dy**2 - 2.0*dy**3)),
                1.0
        )

    # ~~~~~~~~~

    def smoothness_region(dcoll, nodes):
        xpos = nodes[0]
        ypos = nodes[1]

        y_max = 0.50
        y_thickness = 0.10

        y0 = (y_max - y_thickness)
        dy = +((ypos - y0)/y_thickness)

        return actx.np.where(
            actx.np.greater(ypos, y0),
                actx.np.where(actx.np.greater(ypos, y_max),
                              1.0, 3.0*dy**2 - 2.0*dy**3),
                0.0
        )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter

    from grudge.discretization import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_MODAL
    from meshmode.transform_metadata import FirstAxisIsElementsTag

    def drop_order(dcoll: DiscretizationCollection, field, theta,
                   positivity_preserving=False, dd=None):

        # Compute cell averages of the state
        def cancel_polynomials(grp):
            return actx.from_numpy(np.asarray([1 if sum(mode_id) == 0
                                               else 0 for mode_id in grp.mode_ids()]))

        # map from nodal to modal
        if dd is None:
            dd = dd_vol_fluid

        dd_nodal = dd
        dd_modal = dd_nodal.with_discr_tag(DISCR_TAG_MODAL)

        modal_map = dcoll.connection_from_dds(dd_nodal, dd_modal)
        nodal_map = dcoll.connection_from_dds(dd_modal, dd_nodal)

        modal_discr = dcoll.discr_from_dd(dd_modal)
        modal_field = modal_map(field)

        # cancel the ``high-order'' polynomials p > 0, and only the average remains
        filtered_modal_field = DOFArray(
            actx,
            tuple(actx.einsum("ej,j->ej",
                              vec_i,
                              cancel_polynomials(grp),
                              arg_names=("vec", "filter"),
                              tagged=(FirstAxisIsElementsTag(),))
                  for grp, vec_i in zip(modal_discr.groups, modal_field))
        )

        # convert back to nodal to have the average at all points
        cell_avgs = nodal_map(filtered_modal_field)

        if positivity_preserving:
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, 1e-13),
                                      cell_avgs, 1e-13)    

        return theta*(field - cell_avgs) + cell_avgs


    def _drop_order_cv(cv, flipped_smoothness, theta_factor, dd=None):

        smoothness = 1.0 - theta_factor*flipped_smoothness

        density_lim = drop_order(dcoll, cv.mass, smoothness)
        momentum_lim = make_obj_array([
            drop_order(dcoll, cv.momentum[0], smoothness),
            drop_order(dcoll, cv.momentum[1], smoothness)])
        energy_lim = drop_order(dcoll, cv.energy, smoothness)

        # limit species
        spec_lim = make_obj_array([
            drop_order(dcoll, cv.species_mass[i], smoothness)
            for i in range(nspecies)
        ])

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=density_lim, energy=energy_lim,
            momentum=momentum_lim, species_mass=spec_lim)

    drop_order_cv = actx.compile(_drop_order_cv)


    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)
        ])

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

        cv = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

        # make a new CV with the limited variables
        return cv

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_fluid,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

    get_fluid_state = actx.compile(_get_fluid_state)

    # ~~~~~~~~~~

    def _limit_sample_cv(cv, wv, pressure, temperature, epsilon, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = epsilon*gas_model_sample.eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        energy_gas = mass_lim*(gas_model_sample.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        eps_rho_solid = sum(wv)
        tau = gas_model_sample.wall.decomposition_progress(wv)
        energy_solid = eps_rho_solid*gas_model_sample.wall.enthalpy(temperature, tau)

        energy = energy_gas + energy_solid

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)


    def _get_sample_state(cv, wv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_sample,
            wall_density=wv, temperature_seed=temp_seed,
            limiter_func=_limit_sample_cv, limiter_dd=dd_vol_sample
        )

    get_sample_state = actx.compile(_get_sample_state)


    def _create_sample_dependent_vars(wall_density):
        gas_model_solid.wall.dependent_vars(wall_density)

    create_sample_dependent_vars = actx.compile(_create_sample_dependent_vars)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_init = Initializer(pressure=101325.0, temperature=300.0, species_atm=y_atmosphere)

    ref_state = Initializer(pressure=101325.0, temperature=300.0, species_atm=y_atmosphere)

##############################################################################

    if my_material == "fiber":
        wall_density = 0.1*1600.0 + sample_zeros

        sample_init = FiberSampleInitializer(pressure=101325.0,
            temperature=300.0, species_atm=y_atmosphere)

    if my_material == "composite":
        # soln setup and init
        wall_density = np.empty((3,), dtype=object)

        wall_density[0] = 30.0 + sample_zeros
        wall_density[1] = 90.0 + sample_zeros
        wall_density[2] = 160. + sample_zeros

        sample_init = TACOTSampleInitializer(pressure=101325.0,
            temperature=300.0, species=y_atmosphere,
            wall_density=wall_density)

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")

        fluid_tseed = temperature_seed*1.0
        sample_tseed = temp_wall*1.0


        fluid_cv = fluid_init(actx, fluid_nodes, eos)

        sample_density = wall_density
        sample_cv = sample_init(actx, sample_nodes, gas_model_sample)

    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t[0]))

        if restart_iterations:
            current_t = 0.0
            current_step = 0

        if rank == 0:
            logger.info("Restarting soln.")

        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=restart_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                restart_dcoll.discr_from_dd(dd_vol_fluid)
            )
            sample_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_sample),
                restart_dcoll.discr_from_dd(dd_vol_sample)
            )
            fluid_cv = fluid_connection(restart_data["fluid_cv"])
            fluid_tseed = fluid_connection(restart_data["fluid_temperature_seed"])
            sample_cv = sample_connection(restart_data["sample_cv"])
            #sample_tseed = sample_connection(restart_data["wall_temperature_seed"])
            sample_tseed = 300.0 + sample_zeros
            sample_density = sample_connection(restart_data["sample_density"])
        else:
            fluid_cv = restart_data["fluid_cv"]
            fluid_tseed = restart_data["fluid_temperature_seed"]
            sample_cv = restart_data["sample_cv"]
            sample_tseed = restart_data["sample_temperature_seed"]
            sample_density = restart_data["sample_density"]

    fluid_cv = force_evaluation(actx, fluid_cv)
    fluid_tseed = force_evaluation(actx, fluid_tseed)
    fluid_state = get_fluid_state(fluid_cv, fluid_tseed)

    sample_cv = force_evaluation(actx, sample_cv)
    sample_tseed = force_evaluation(actx, sample_tseed)
    sample_density = force_evaluation(actx, sample_density)
    sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)

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

    smooth_region = force_evaluation(
        actx, smoothness_region(dcoll, fluid_nodes))

#    reaction_rates_damping = force_evaluation(
#        actx, reaction_damping(dcoll, fluid_nodes))

##############################################################################

    # initialize the sponge field
    sponge_x_thickness = 0.20

    xMaxLoc = x0_sponge + sponge_x_thickness

    sponge_init = InitSponge(amplitude=sponge_amp,
        x_max=xMaxLoc, x_min=-xMaxLoc, x_thickness=sponge_x_thickness)

    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=fluid_nodes))
    ref_cv = force_evaluation(actx, ref_state(actx, fluid_nodes, eos))

##############################################################################

    fluid_boundaries = {
        dd_vol_fluid.trace("fluid_base").domain_tag:
            AdiabaticSlipBoundary(),
        dd_vol_fluid.trace("outflow").domain_tag:
            PressureOutflowBoundary(boundary_pressure=101325.0),
    }

    # ~~~~~~~~~~
    sample_boundaries = {
        dd_vol_sample.trace("sample_base").domain_tag: AdiabaticSlipBoundary()
    }

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    sample_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_sample)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=maximum_fluid_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=maximum_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

##############################################################################

    def my_write_viz(
        step, t, dt, fluid_state, sample_state, smoothness):

        wdv = gas_model_sample.wall.dependent_vars(sample_state.dv.wall_density)

        fluid_viz_fields = [
            ("rho_g", fluid_state.cv.mass),
            ("rhoU_g", fluid_state.cv.momentum),
            ("rhoE_g", fluid_state.cv.energy),
            ("pressure", fluid_state.pressure),
            ("temperature", fluid_state.temperature),
            ("Vx", fluid_state.velocity[0]),
            ("Vy", fluid_state.velocity[1]),
#            ("grad_t", fluid_grad_temperature),
            ("dt", dt[0] if local_dt else None),
            ("sponge", sponge_sigma),
            ("smoothness", 1.0 - theta_factor*smoothness),
#            ("RR", chem_rate*reaction_rates_damping)
        ]

        # species mass fractions
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        energy_source = (energy_source_term(sample_nodes, sample_state.cv)).energy

        sample_viz_fields = [
            ("rho_g", sample_state.cv.mass),
            ("rhoU_g", sample_state.cv.momentum),
            ("rhoE_b", sample_state.cv.energy),
            ("pressure", sample_state.pressure),
            ("temperature", sample_state.temperature),
            ("solid_mass", sample_state.dv.wall_density),
            ("Vx", sample_state.velocity[0]),
            ("Vy", sample_state.velocity[1]),
            ("void_fraction", wdv.void_fraction),
            ("progress", wdv.tau),
            ("permeability", wdv.permeability),
            ("kappa", sample_state.thermal_conductivity),
#            ("grad_t", sample_grad_temperature),
            ("dt", dt[2] if local_dt else None),
            ("source", energy_source)
        ]

        # species mass fractions
        sample_viz_fields.extend(
            ("Y_"+species_names[i], sample_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        write_visfile(dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t, overwrite=True, comm=comm)
        write_visfile(dcoll, sample_viz_fields, sample_visualizer,
            vizname=vizname+"-sample", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, fluid_state, sample_state):
        if rank == 0:
            print('Writing restart file...')

        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "fluid_cv": fluid_state.cv,
                "fluid_temperature_seed": fluid_state.temperature,
                "sample_cv": sample_state.cv,
                "sample_density": sample_state.dv.wall_density,
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

        if global_reduce(check_naninf_local(dcoll, "vol", pressure),
                         op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature),
                         op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

##############################################################################

    def gravity_source_terms(cv):
        """Gravity."""
        delta_rho = cv.mass - rho_atmosphere
        return make_conserved(dim=2, mass=cv.mass*0.0,
            energy=delta_rho*cv.velocity[1]*-9.80665,
            momentum=make_obj_array([cv.mass*0.0, delta_rho*-9.80665]),
            species_mass=cv.species_mass*0.0)

    # ~~~~~~~
    def chemical_source_term(cv, temperature):
        return speedup_factor*eos.get_species_source_terms(cv, temperature)
#        return speedup_factor*reaction_rates_damping*(
#            eos.get_species_source_terms(cv, temperature))

    def energy_source_term(x_vec, cv):
        actx = x_vec[0].array_context
        zeros = actx.np.zeros_like(x_vec[0])
        center = np.zeros(2,)
        center[1] = 0.025
        energy_source = 10000.0*actx.np.exp(-0.5*np.dot(x_vec-center, x_vec-center)/0.01**2)
        return make_conserved(dim=2, mass=zeros, energy=energy_source,
            momentum=make_obj_array([zeros, zeros]),
            species_mass=cv.species_mass*0.0
        )

##############################################################################

#    from grudge.dof_desc import DD_VOLUME_ALL
    def _my_get_timestep_sample():
        return maximum_wall_dt

    def _my_get_timestep_fluid(fluid_state, t, dt):

        if not constant_cfl:
            return dt

        return get_sim_timestep(dcoll, fluid_state, t, dt,
            maximum_cfl, gas_model_fluid, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd_vol_fluid)

    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)
    my_get_timestep_sample = actx.compile(_my_get_timestep_sample)

##############################################################################

    import os
    from grudge.reductions import integral
    #FIXME initial_mass = integral(dcoll, dd_vol_sample, np.pi*sample_nodes[0]*0.1*1600.0 + sample_zeros)
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        fluid_cv, fluid_tseed, sample_cv, sample_tseed, sample_density = state

        fluid_cv = force_evaluation(actx, fluid_cv)
        fluid_tseed = force_evaluation(actx, fluid_tseed)
        sample_cv = force_evaluation(actx, sample_cv)
        sample_tseed = force_evaluation(actx, sample_tseed)

        # include both outflow and sponge in the damping region
        # apply outflow damping
        smoothness = force_evaluation(actx,
            smooth_region + sponge_sigma/sponge_amp)
        fluid_cv = drop_order_cv(fluid_cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(fluid_cv, fluid_tseed)
        fluid_cv = fluid_state.cv

        # construct species-limited solid state
        sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)
        sample_cv = sample_state.cv

        wall_density = gas_model_sample.wall.solid_density(sample_state.dv.wall_density)
#        current_mass = integral(dcoll, dd_vol_sample, np.pi*sample_nodes[0]*wall_density)
#        print(current_mass*1000 - initial_mass*1000,'g')
#        print(current_mass*1000,'g')
#        print(initial_mass*1000,'g')

        if local_dt:
            t = force_evaluation(actx, t)

            dt_fluid = force_evaluation(actx, actx.np.minimum(
                maximum_fluid_dt, my_get_timestep_fluid(fluid_state, t[0], dt[0])))

            dt_sample = force_evaluation(actx, maximum_solid_dt + sample_zeros)

            dt = make_obj_array([dt_fluid, fluid_zeros,
                                 dt_sample, sample_zeros, dt_sample])
        else:
            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, maximum_cfl,
                                      t_final, constant_cfl, local_dt, dd_vol_fluid)

        try:
            state = make_obj_array([
                fluid_cv, fluid_state.temperature,
                sample_cv, sample_state.temperature, sample_state.dv.wall_density])

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
                ## FIXME warning in lazy compilation
                from warnings import warn
                warn(f"Lazy does not like the health_check", stacklevel=2)
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    sample_state=sample_state, smoothness=smoothness)

            if do_restart:
                my_write_restart(step, t, fluid_state, sample_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                sample_state=sample_state, smoothness=smoothness)
            raise

        return state, dt


    def my_rhs(time, state):

        fluid_cv, fluid_tseed, sample_cv, sample_tseed, sample_density = state

        # include both outflow and sponge in the damping region
        # apply outflow damping
        smoothness = smooth_region + sponge_sigma/sponge_amp
        fluid_cv = _drop_order_cv(fluid_cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model_fluid,
            temperature_seed=fluid_tseed,
            limiter_func=_limit_fluid_cv, limiter_dd=dd_vol_fluid)
        fluid_cv = fluid_state.cv

        # construct species-limited solid state
        sample_state = make_fluid_state(cv=sample_cv, gas_model=gas_model_sample,
            wall_density=sample_density, temperature_seed=sample_tseed,
            limiter_func=_limit_sample_cv, limiter_dd=dd_vol_sample)
        sample_cv = sample_state.cv

        #~~~~~~~~~~~~~
        fluid_all_boundaries_no_grad, sample_all_boundaries_no_grad = \
            add_multiphysics_interface_boundaries_no_grad(
                dcoll, dd_vol_fluid, dd_vol_sample,
                fluid_state, sample_state,
                fluid_boundaries, sample_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                use_kappa_weighted_grad_flux_in_fluid=False,
                wall_penalty_amount=wall_penalty_amount)

        # ~~~~~~~~~~~~~~
        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model_fluid, fluid_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_fluid, comm_tag=_FluidOpStatesTag,
            limiter_func=_limit_fluid_cv)

        sample_operator_states_quad = make_operator_fluid_states(
            dcoll, sample_state, gas_model_sample, sample_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_sample, comm_tag=_WallOpStatesTag,
            limiter_func=_limit_sample_cv)

        # ~~~~~~~~~~~~~~
        # fluid grad CV
        fluid_grad_cv = grad_cv_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            #comm_tag=_FluidGradCVTag
        )

        # fluid grad T
        fluid_grad_temperature = grad_t_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            #comm_tag=_FluidGradTempTag
        )

        # sample grad CV
        sample_grad_cv = grad_cv_operator(
            dcoll, gas_model_sample, sample_all_boundaries_no_grad, sample_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            #comm_tag=_SampleGradCVTag
        )

        # sample grad T
        sample_grad_temperature = grad_t_operator(
            dcoll, gas_model_sample, sample_all_boundaries_no_grad, sample_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            #comm_tag=_SampleGradTempTag
        )

        # ~~~~~~~~~~~~~~~~~
        fluid_all_boundaries, sample_all_boundaries = \
            add_multiphysics_interface_boundaries(
                dcoll, dd_vol_fluid, dd_vol_sample,
                fluid_state, sample_state,
                fluid_grad_cv, sample_grad_cv,
                fluid_grad_temperature, sample_grad_temperature,
                fluid_boundaries, sample_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                wall_emissivity=emissivity, sigma=5.67e-8, ambient_temperature=300.0,
                use_kappa_weighted_grad_flux_in_fluid=False,
                wall_penalty_amount=wall_penalty_amount)

        #~~~~~~~~~~~~~
        fluid_rhs = ns_operator(
            dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
            comm_tag=_FluidOperatorTag, inviscid_terms_on=True)

        fluid_sources = (
            chemical_source_term(fluid_cv, fluid_state.temperature)
            + sponge_func(cv=fluid_cv, cv_ref=ref_cv, sigma=sponge_sigma)
            + gravity_source_terms(fluid_cv)
        )

        #~~~~~~~~~~~~~

        sample_rhs = ns_operator(
            dcoll, gas_model_sample, sample_state, sample_all_boundaries,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            grad_cv=sample_grad_cv, grad_t=sample_grad_temperature,
            comm_tag=_SampleOperatorTag, inviscid_terms_on=False)

#        sample_mass_rhs = decomposition.get_source_terms()
        sample_mass_rhs = sample_zeros

        sample_sources = (
            energy_source_term(sample_nodes, sample_cv)
#            eos.get_species_source_terms(sample_cv, sample_state.temperature)
        )

        #~~~~~~~~~~~~~
        return make_obj_array([
            fluid_rhs + fluid_sources, fluid_zeros,
            sample_rhs + sample_sources, sample_zeros, sample_mass_rhs])


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

        min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([
        fluid_cv, fluid_state.temperature,
        sample_cv, sample_state.temperature, sample_state.dv.wall_density])

    if local_dt == True:
        dt_fluid = force_evaluation(actx, actx.np.minimum(
            maximum_fluid_dt,
            my_get_timestep_fluid(fluid_state,
                                  force_evaluation(actx, current_t + fluid_zeros),
                                  force_evaluation(actx, maximum_fluid_dt + fluid_zeros))
            )
        )

        dt_sample = force_evaluation(actx, maximum_solid_dt + sample_zeros)

        dt = make_obj_array([
            dt_fluid, fluid_zeros, dt_sample, sample_zeros, dt_sample])

        t_fluid = force_evaluation(actx, current_t + fluid_zeros)
        t_sample = force_evaluation(actx, current_t + sample_zeros)

        t = make_obj_array([
            t_fluid, t_fluid, t_sample, t_sample, t_sample])
    else:
        if constant_cfl:
            dt = get_sim_timestep(dcoll, fluid_state, t, maximum_fluid_dt,
                maximum_cfl, t_final, constant_cfl, local_dt, dd_vol_fluid)
        else:
            dt = 1.0*maximum_fluid_dt
            t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt, t=t, t_final=t_final,
                      max_steps=niter, local_dt=local_dt,
                      force_eval=force_eval, state=stepper_state)

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
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

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

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile, casename=casename,
         lazy=args.lazy, restart_filename=restart_file)
