""" Fri 10 Mar 2023 02:22:21 PM CST """

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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
import yaml
import logging
import sys
import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cla  # noqa
import pyopencl.tools as cl_tools
from functools import partial
from dataclasses import dataclass, fields

from arraycontext import (
    dataclass_array_container, with_container_arithmetic,
    get_container_context_recursively
)

from meshmode.dof_array import DOFArray
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, BoundaryDomainTag, VolumeDomainTag
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
    IsothermalWallBoundary,
    PressureOutflowBoundary,
    AdiabaticSlipBoundary,
    PrescribedFluidBoundary,
    AdiabaticNoslipWallBoundary,
    LinearizedOutflowBoundary
)

from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient, make_conserved
)
from mirgecom.transport import (
    PowerLawTransport,
    MixtureAveragedTransport
)
import cantera
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    make_operator_fluid_states
)
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)

from pytools.obj_array import make_obj_array

from mirgecom.navierstokes import (
    grad_t_operator,
    grad_cv_operator,
    ns_operator
)
from mirgecom.multiphysics.phenolics_coupled_fluid_wall import (
    add_interface_boundaries as add_thermal_interface_boundaries,
    add_interface_boundaries_no_grad as add_thermal_interface_boundaries_no_grad
)
from mirgecom.diffusion import (
    diffusion_operator,
    grad_operator as wall_grad_t_operator,
    NeumannDiffusionBoundary
)

from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)
from mirgecom.wall_model import (
    SolidWallModel, SolidWallState, SolidWallConservedVars
)

#########################################################################

class _FluidOpStatesTag:
    pass


class _FluidGradCVTag:
    pass


class _FluidGradTempTag:
    pass


class _FluidOperatorTag:
    pass


class _SolidGradTempTag:
    pass


class _SolidOperatorTag:
    pass


class _WallOxDiffCommTag:
    pass


class _OxCommTag:
    pass


class _MyGradTag_Bdry:
    pass


class _MyGradTag_1:
    pass


class _MyGradTag_2:
    pass


class _MyGradTag_3:
    pass


class _MyGradTag_4:
    pass


class _MyGradTag_5:
    pass


class _MyGradTag_6:
    pass


class Burner2D_Reactive:

    def __init__(
            self, *, dim=2, sigma, sigma_flame,
            flame_loc, pressure, temperature, speedup_factor,
            mass_rate_burner, mass_rate_shroud,
            species_unburn, species_burned, species_shroud, species_atm):

        self._dim = dim
        self._sigma = sigma
        self._sigma_flame = sigma_flame
        self._pres = pressure
        self._speedup_factor = speedup_factor
        self._yu = species_unburn
        self._yb = species_burned
        self._ys = species_shroud
        self._ya = species_atm
        self._temp = temperature
        self._flaLoc = flame_loc

        self._mass_rate_burner = mass_rate_burner
        self._mass_rate_shroud = mass_rate_shroud

    def __call__(self, actx, x_vec, eos, state_minus=None):

        if x_vec.shape != (self._dim,):
            raise ValueError(f'Position vector has unexpected dimensionality,'
                             f' expected {self._dim}.')

        _ya = self._ya
        _yb = self._yb
        _ys = self._ys
        _yu = self._yu
        cool_temp = 300.0

        upper_bnd = 0.105

        sigma_factor = 12.0 - 11.0*(upper_bnd - x_vec[1])**2/(upper_bnd - 0.10)**2
        _sigma = self._sigma*(
            actx.np.where(actx.np.less(x_vec[1], upper_bnd),
                              actx.np.where(actx.np.greater(x_vec[1], .10),
                                            sigma_factor, 1.0),
                              12.0)
        )
#        _sigma = self._sigma

        int_diam = 2.38*25.4/2000.0  # radius, actually
        ext_diam = 2.89*25.4/2000.0  # radius, actually

        # ~~~ shroud
        S1 = 0.5*(1.0 + actx.np.tanh(1.0/(_sigma)*(x_vec[0] - int_diam)))
        S2 = actx.np.where(actx.np.greater(x_vec[0], ext_diam),
                 0.0, - actx.np.tanh(1.0/(_sigma)*(x_vec[0] - ext_diam))
             )
        shroud = S1*S2

        # ~~~ flame ignition
        core = 0.5*(1.0 - actx.np.tanh(1.0/_sigma*(x_vec[0] - int_diam)))

        # ~~~ atmosphere
        atmosphere = 1.0 - (shroud + core)
             
        # ~~~ after combustion products
        upper_atm = 0.5*(1.0 + actx.np.tanh(1.0/(2.0*self._sigma)*(x_vec[1] - upper_bnd)))

        # ~~~ flame ignition
        flame = 0.5*(1.0 + actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - self._flaLoc)))

        # ~~~ species
        yf = (flame*_yb + (1.0-flame)*_yu)*(1.0-upper_atm) + _ya*upper_atm
        ys = _ya*upper_atm + (1.0 - upper_atm)*_ys
        y = atmosphere*_ya + shroud*ys + core*yf

        # ~~~ temperature and EOS
        temp = (flame*self._temp + (1.0-flame)*cool_temp)*(1.0-upper_atm) + 300.0*upper_atm
        temperature = temp*core + 300.0*(1.0 - core)
        
        if state_minus is None:
            pressure = self._pres + 0.0*x_vec[0]
            mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        else:
            mass = state_minus.cv.mass

        # ~~~ velocity and/or momentum
        mom_burner = self._mass_rate_burner*self._speedup_factor
        mom_shroud = self._mass_rate_shroud*self._speedup_factor
        smoothY = mom_burner*(1.0-upper_atm) + 0.0*upper_atm
        mom_x = 0.0*x_vec[0]
        mom_y = core*smoothY + shroud*mom_shroud*(1.0-upper_atm)
        momentum = make_obj_array([mom_x, mom_y])
        velocity = momentum/mass

        # ~~~
        specmass = mass * y

        # ~~~
        internal_energy = eos.get_internal_energy(
            temperature, species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(
            dim=self._dim, mass=mass, energy=energy,
            momentum=mass*velocity, species_mass=specmass)


def reaction_damping(dcoll, nodes, **kwargs):
    ypos = nodes[1]
    actx = ypos.array_context

    y_max = 0.25
    y_thickness = 0.10

    y0 = (y_max - y_thickness)
    dy = +((ypos - y0)/y_thickness)
    return actx.np.where(
        actx.np.greater(ypos, y0),
            actx.np.where(actx.np.greater(ypos, y_max),
                          0.0, 1.0 - (3.0*dy**2 - 2.0*dy**3)),
            1.0
    )


def smoothness_region(dcoll, nodes):
    xpos = nodes[0]
    ypos = nodes[1]
    actx = ypos.array_context

    y_max = 0.55
    y_thickness = 0.20

    y0 = (y_max - y_thickness)
    dy = +((ypos - y0)/y_thickness)

    return actx.np.where(
        actx.np.greater(ypos, y0),
            actx.np.where(actx.np.greater(ypos, y_max),
                          1.0, 3.0*dy**2 - 2.0*dy**3),
            0.0
    )


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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr, filename=(f'{casename}.sqlite'),
                               mode='wo', mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, 'queue', None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    mesh_filename = 'mesh_01_round_10mm_020um_2domains-v2.msh'

    rst_path = 'restart_data/'
    viz_path = 'viz_data/'
    vizname = viz_path+casename
    rst_pattern = rst_path+'{cname}-{step:06d}-{rank:04d}.pkl'

    # default i/o frequencies
    nviz = 25000
    nrestart = 25000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = 'ssprk43'
    current_dt = 2.5e-6  # order == 2
    t_final = 2.0
    niter = 4000001
    local_dt = True
    constant_cfl = True
    current_cfl = 0.2

    # discretization and model control
    order = 2
    use_overintegration = False

    x0_sponge = 0.150
    sponge_amp = 400.0
    theta_factor = 0.02
    speedup_factor = 7.5

    mechanism_file = 'uiuc_8sp_phenol'
    equiv_ratio = 0.7
    chem_rate = 1.0
    flow_rate = 25.0
    shroud_rate = 11.85
    temp_products = 2000.0

#    transport = 'Mixture'
    transport = 'PowerLaw'

    # wall stuff
    ignore_wall = False

    temp_wall = 300.0

    wall_penalty_amount = 1.0
    wall_time_scale = 10.0*speedup_factor  # wall speed-up

    use_radiation = True
    emissivity = 0.85*speedup_factor

    restart_iterations = False

##########################################################################
    
    dim = 2

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    if integrator == 'ssprk43':
        from mirgecom.integrators.ssprk import ssprk43_step
        timestepper = ssprk43_step
        force_eval = True

    if rank == 0:
        print('\n#### Simulation control data: ####')
        print(f'\tnviz = {nviz}')
        print(f'\tnrestart = {nrestart}')
        print(f'\tnhealth = {nhealth}')
        print(f'\tnstatus = {nstatus}')
        print(f'\tcurrent_dt = {current_dt}')
        if constant_cfl is False:
            print(f'\tt_final = {t_final}')
        else:
            print(f'\tconstant_cfl = {constant_cfl}')
            print(f'\tcurrent_cfl = {current_cfl}')
            print(f'\tniter = {niter}')
        print(f'\torder = {order}')
        print(f'\tTime integration = {integrator}')

##############################################################################

    restart_step = None
    if restart_file is None:

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0

        if rank == 0:
            print(f'Reading mesh from {mesh_filename}')

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                'fluid': ['fluid'],
                'solid': ['wall_sample', 'wall_alumina', 'wall_graphite']
                }
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data['step']
        volume_to_local_mesh_data = restart_data['volume_to_local_mesh_data']
        global_nelements = restart_data['global_nelements']
        restart_order = int(restart_data['order'])
        first_step = restart_step+0

        assert comm.Get_size() == restart_data['num_parts']

    local_nelements = (
          volume_to_local_mesh_data['fluid'][0].nelements +
          volume_to_local_mesh_data['solid'][0].nelements)

    from grudge.dof_desc import DISCR_TAG_QUAD
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
        logger.info('Done making discretization')

    dd_vol_fluid = DOFDesc(VolumeDomainTag('fluid'), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag('solid'), DISCR_TAG_BASE)

    from mirgecom.utils import mask_from_elements
    wall_vol_discr = dcoll.discr_from_dd(dd_vol_solid)
    wall_tag_to_elements = volume_to_local_mesh_data['solid'][1]
    wall_sample_mask = mask_from_elements(
        dcoll, dd_vol_solid, actx, wall_tag_to_elements['wall_sample'])
    wall_alumina_mask = mask_from_elements(
        dcoll, dd_vol_solid, actx, wall_tag_to_elements['wall_alumina'])
    wall_graphite_mask = mask_from_elements(
        dcoll, dd_vol_solid, actx, wall_tag_to_elements['wall_graphite'])

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))

    fluid_zeros = force_evaluation(actx, fluid_nodes[0]*0.0)
    solid_zeros = force_evaluation(actx, solid_nodes[0]*0.0)

    # ~~~~~~~~~~
    from grudge.dt_utils import characteristic_lengthscales
    char_length_fluid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)
    char_length_solid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid)

##########################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name='gas', yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = temp_products

    air = 'O2:0.21,N2:0.79'
    fuel = 'C2H4:1'
    cantera_soln.set_equivalence_ratio(phi=equiv_ratio,
                                       fuel=fuel, oxidizer=air)
    x_unburned = cantera_soln.X
    pres_unburned = 101325.0

    rho_int = cantera_soln.density

    r_int = 2.38*25.4/2000.0
    r_ext = 2.89*25.4/2000.0
    
    mass_react = flow_rate*1.0
    mass_shroud = shroud_rate*1.0
    A_int = np.pi*r_int**2
    A_ext = np.pi*(r_ext**2 - r_int**2)
    lmin_to_m3s = 1.66667e-5
    u_int = mass_react*lmin_to_m3s/A_int
    u_ext = mass_shroud*lmin_to_m3s/A_ext
    rhoU_int = rho_int*u_int

    mmw_N2 = cantera_soln.molecular_weights[cantera_soln.species_index('N2')]
    rho_ext = 101325.0/((8314.46/mmw_N2)*300.0)
    mdot_ext = rho_ext*u_ext*A_ext
    rhoU_ext = rho_ext*u_ext

    print('V_dot=', mass_react, '(L/min)')
    print(f'{A_int= }', '(m^2)')
    print(f'{A_ext= }', '(m^2)')
    print(f'{u_int= }', '(m/s)')
    print(f'{u_ext= }', '(m/s)')
    print(f'{rho_int= }', '(kg/m^3)')
    print(f'{rho_ext= }', '(kg/m^3)')
    print(f'{rhoU_int= }')
    print(f'{rhoU_ext= }')
    print('ratio=', u_ext/u_int)

    # Let the user know about how Cantera is being initilized
    print(f'Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x_unburned}')
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x_unburned

    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY
    mmw_unburned = cantera_soln.mean_molecular_weight

    cantera_soln.TPX = temp_ignition, pres_unburned, x_unburned
    cantera_soln.equilibrate('TP')
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
#    pres_burned = cantera_soln.P
    mmw_burned = cantera_soln.mean_molecular_weight

    # Pull temperature, density, mass fractions, and pressure from Cantera
    x = np.zeros(nspecies)
    x[cantera_soln.species_index('O2')] = 0.21
    x[cantera_soln.species_index('N2')] = 0.79
    cantera_soln.TPX = temp_unburned, pres_unburned, x

    y_atmosphere = np.zeros(nspecies)
    dummy, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    cantera_soln.equilibrate('TP')
    temp_atmosphere, rho_atmosphere, y_atmosphere = cantera_soln.TDY
#    pres_atmosphere = cantera_soln.P
    mmw_atmosphere = cantera_soln.mean_molecular_weight

    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_shroud = y_atmosphere*0.0
    y_shroud[cantera_soln.species_index('N2')] = 1.0

    cantera_soln.TPY = 300.0, 101325.0, y_shroud
    temp_shroud, rho_shroud = cantera_soln.TD
    mmw_shroud = cantera_soln.mean_molecular_weight

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
                                cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 1000.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    species_names = pyrometheus_mechanism.species_names

    print(f'Pyrometheus mechanism species names {species_names}')
    print('Unburned:')
    print(f'T = {temp_unburned}')
    print(f'D = {rho_unburned}')
    print(f'Y = {y_unburned}')
    print(f'W = {mmw_unburned}')
    print(' ')
    print('Burned:')
    print(f'T = {temp_burned}')
    print(f'D = {rho_burned}')
    print(f'Y = {y_burned}')
    print(f'W = {mmw_burned}')
    print(' ')
    print('Atmosphere:')
    print(f'T = {temp_atmosphere}')
    print(f'D = {rho_atmosphere}')
    print(f'Y = {y_atmosphere}')
    print(f'W = {mmw_atmosphere}')
    print(' ')
    print('Shroud:')
    print(f'T = {temp_shroud}')
    print(f'D = {rho_shroud}')
    print(f'Y = {y_shroud}')
    print(f'W = {mmw_shroud}')

    # }}}

    # {{{ Initialize transport model
    physical_transport = None
    if transport == 'Mixture':
        physical_transport = \
            MixtureAveragedTransport(pyrometheus_mechanism,
                                     lewis=np.ones(nspecies,),
                                     factor=speedup_factor)
    else:
        if transport == 'PowerLaw':
            physical_transport = \
                PowerLawTransport(lewis=np.ones((nspecies,)),
                                  beta=4.093e-7*speedup_factor)
        else:
            print('No transport class defined..')
            print('Use one of "Mixture" or "PowerLaw"')
            sys.exit()

    # {{{ Initialize wall model

    my_material = 'composite'
    if my_material == 'fiber':
        wall_sample_density = 0.1*1600.0 + solid_zeros

        import mirgecom.materials.carbon_fiber as material_sample
        material = material_sample.FiberEOS(char_mass=0.0,
                                            virgin_mass=160.0)
        decomposition = material_sample.Y3_Oxidation_Model(wall_material=material)

    if my_material == 'composite':
        wall_sample_density = np.empty((3,), dtype=object)

        wall_sample_density[0] = 30.0 + solid_zeros
        wall_sample_density[1] = 90.0 + solid_zeros
        wall_sample_density[2] = 160. + solid_zeros

        import mirgecom.materials.tacot as material_sample
        material = material_sample.TacotEOS(char_mass=220.0,
                                            virgin_mass=280.0)
        decomposition = material_sample.Pyrolysis()

    # Averaging from https://www.azom.com/properties.aspx?ArticleID=52
    wall_alumina_rho = 3500.0
    wall_alumina_cp = 700.0
    wall_alumina_kappa = 25.00

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    # TODO There is a table with the temperature-dependent data for graphite
    wall_graphite_rho = 1625.0
    wall_graphite_cp = 770.0
    wall_graphite_kappa = 50.0

#    def _get_solid_density(wv):
#        wall_sample_rho = sum(wv.mass)
#        return (wall_sample_rho * wall_sample_mask
#                + wall_alumina_rho * wall_alumina_mask
#                + wall_graphite_rho * wall_graphite_mask)

    def _get_solid_enthalpy(temperature, tau):
        wall_sample_h = material.enthalpy(temperature, tau)
        wall_alumina_h = wall_alumina_cp * temperature
        wall_graphite_h = wall_graphite_cp * temperature
        return (wall_sample_h * wall_sample_mask
                + wall_alumina_h * wall_alumina_mask
                + wall_graphite_h * wall_graphite_mask)

    def _get_solid_heat_capacity(temperature, tau):
        wall_sample_cp = material.heat_capacity(temperature, tau)
        return (wall_sample_cp * wall_sample_mask
                + wall_alumina_cp * wall_alumina_mask
                + wall_graphite_cp * wall_graphite_mask)

    def _get_solid_thermal_conductivity(temperature, tau):
        wall_sample_kappa = material.thermal_conductivity(temperature, tau)
        return (wall_sample_kappa * wall_sample_mask
                + wall_alumina_kappa * wall_alumina_mask
                + wall_graphite_kappa * wall_graphite_mask)

    def _get_solid_decomposition_progress(mass):
        wall_sample_tau = material.decomposition_progress(mass)
        return (wall_sample_tau * wall_sample_mask
                + 1.0 * wall_alumina_mask
                + 1.0 * wall_graphite_mask)

    solid_wall_model = SolidWallModel(
        #density_func=_get_solid_density,
        enthalpy_func=_get_solid_enthalpy,
        heat_capacity_func=_get_solid_heat_capacity,
        thermal_conductivity_func=_get_solid_thermal_conductivity,
        decomposition_func=_get_solid_decomposition_progress)

    # }}}

    gas_model = GasModel(eos=eos, transport=physical_transport)

#############################################################################

    def reaction_damping(dcoll, nodes, **kwargs):

        ypos = nodes[1]

        y_max = 0.25
        y_thickness = 0.10

        y0 = (y_max - y_thickness)
        dy = +((ypos - y0)/y_thickness)
        damping = actx.np.where(
            actx.np.greater(ypos, y0),
            actx.np.where(actx.np.greater(ypos, y_max),
                          0.0, 1.0 - (3.0*dy**2 - 2.0*dy**3)),
            1.0
        )

        return damping

#############################################################################

    def smoothness_region(dcoll, nodes):
        ypos = nodes[1]

        y_max = 0.55
        y_thickness = 0.20

        y0 = (y_max - y_thickness)
        dy = +((ypos - y0)/y_thickness)
        region = actx.np.where(
            actx.np.greater(ypos, y0),
            actx.np.where(actx.np.greater(ypos, y_max),
                          1.0, 3.0*dy**2 - 2.0*dy**3),
            0.0
        )

        return region

##############################################################################

    from mirgecom.limiter import bound_preserving_limiter

    from grudge.discretization import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_MODAL
    from meshmode.transform_metadata import FirstAxisIsElementsTag

    def drop_order(dcoll: DiscretizationCollection, field, theta,
                   positivity_preserving=False, dd=None):
        # Compute cell averages of the state
        def cancel_polynomials(grp):
            return actx.from_numpy(
                np.asarray([1 if sum(mode_id) == 0
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

        # cancel the ``high-order'' polynomials p > 0 and keep the average
        filtered_modal_field = DOFArray(
            actx,
            tuple(actx.einsum('ej,j->ej',
                              vec_i,
                              cancel_polynomials(grp),
                              arg_names=('vec', 'filter'),
                              tagged=(FirstAxisIsElementsTag(),))
                  for grp, vec_i in zip(modal_discr.groups, modal_field))
        )

        # convert back to nodal to have the average at all points
        cell_avgs = nodal_map(filtered_modal_field)

        if positivity_preserving:
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, 1e-13),
                                                      cell_avgs, 1e-13)    

        return theta*(field - cell_avgs) + cell_avgs

    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                                     mmin=0.0, mmax=1.0, modify_average=True,
                                     dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = eos.get_density(pressure=pressure,
                                   temperature=temperature,
                                   species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        cv = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                            momentum=mass_lim*cv.velocity,
                            species_mass=mass_lim*spec_lim)

        # make a new CV with the limited variables
        return cv

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

##################################

    fluid_init = Burner2D_Reactive(
        dim=dim, sigma=0.00020,
        sigma_flame=0.00005, temperature=temp_ignition, pressure=101325.0,
        flame_loc=0.10025, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

    ref_state = Burner2D_Reactive(
        dim=dim, sigma=0.00020,
        sigma_flame=0.00001, temperature=temp_ignition, pressure=101325.0,
        flame_loc=0.1050, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

###############################################################################

    smooth_region = force_evaluation(
            actx, smoothness_region(dcoll, fluid_nodes))

    reaction_rates_damping = force_evaluation(
            actx, reaction_damping(dcoll, fluid_nodes))

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(
            cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid,
        )

    get_fluid_state = actx.compile(_get_fluid_state)

    def _get_solid_state(wv, wv_tseed):
#        print(wv)
        wdv = solid_wall_model.dependent_vars(wv=wv, tseed=wv_tseed)
        return SolidWallState(cv=wv, dv=wdv)

    get_solid_state = actx.compile(_get_solid_state)

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info('Initializing soln.')
        current_cv = fluid_init(actx, fluid_nodes, eos)

        tseed = force_evaluation(actx, 1000.0 + fluid_zeros)
        wv_tseed = force_evaluation(actx, temp_wall + solid_zeros)

        if isinstance(wall_sample_density, DOFArray):
            wall_densities = (
                wall_sample_density * wall_sample_mask
                + wall_alumina_rho * wall_alumina_mask
                + wall_graphite_rho * wall_graphite_mask)
        else:
            wall_alumina_density = actx.np.zeros_like(wall_sample_density)
            wall_graphite_density = actx.np.zeros_like(wall_sample_density)

            wall_alumina_density[-1] = wall_alumina_rho
            wall_graphite_density[-1] = wall_graphite_rho

            wall_densities = (
                wall_sample_density * wall_sample_mask
                + wall_alumina_density * wall_alumina_mask
                + wall_graphite_density * wall_graphite_mask)

        # summ all the phases of the material
        tau = solid_wall_model.decomposition_progress(wall_densities)
        wall_mass = solid_wall_model.solid_density(wall_densities)

        wall_sample_h = material.enthalpy(wv_tseed, tau)
        wall_alumina_h = wall_alumina_cp * wv_tseed
        wall_graphite_h = wall_graphite_cp * wv_tseed
        wall_enthalpy = (
            wall_sample_h * wall_sample_mask
            + wall_alumina_h * wall_alumina_mask
            + wall_graphite_h * wall_graphite_mask)

        wall_energy = wall_mass * wall_enthalpy

        current_wv = SolidWallConservedVars(mass=wall_densities,
                                            energy=wall_energy)

    else:
        current_step = restart_step
        current_t = restart_data['t']
        if np.isscalar(current_t) is False:
            current_t = np.min(actx.to_numpy(current_t[0]))

        if restart_iterations:
            current_t = 0.0
            current_step = 0

        if rank == 0:
            logger.info('Restarting soln.')

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
            wall_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_solid),
                restart_dcoll.discr_from_dd(dd_vol_solid)
            )
            current_cv = fluid_connection(restart_data['cv'])
            tseed = fluid_connection(restart_data['temperature_seed'])
            current_wv = wall_connection(restart_data['wv'])
            wv_tseed = fluid_connection(restart_data['wall_temperature_seed'])
        else:
            current_cv = restart_data['cv']
            tseed = restart_data['temperature_seed']
            current_wv = restart_data['wv']
            wv_tseed = restart_data['wall_temperature_seed']

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

#########################################################################

    current_cv = force_evaluation(actx, current_cv)
    tseed = force_evaluation(actx, tseed)
    current_fluid_state = get_fluid_state(current_cv, tseed)

    current_wv = force_evaluation(actx, current_wv)
    wv_tseed = force_evaluation(actx, wv_tseed)
    current_solid_state = get_solid_state(current_wv, wv_tseed)

#########################################################################

    original_casename = casename
    casename = f'{casename}-d{dim}p{order}e{global_nelements}n{nparts}'
    logmgr = initialize_logmgr(use_logmgr, filename=(f'{casename}.sqlite'),
                               mode='wo', mpi_comm=comm)

    vis_timer = None
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ('step.max', 'step = {value}, '),
            ('dt.max', 'dt: {value:1.5e} s, '),
            ('t_sim.max', 'sim time: {value:1.5e} s, '),
            ('t_step.max', '--- step walltime: {value:5g} s\n')
            ])

        try:
            logmgr.add_watches(['memory_usage_python.max',
                                'memory_usage_gpu.max'])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(['pyopencl_array_time.max'])

        vis_timer = IntervalTimer('t_vis', 'Time spent visualizing')
        logmgr.add_quantity(vis_timer)

        gc_timer = IntervalTimer('t_gc', 'Time spent garbage collecting')
        logmgr.add_quantity(gc_timer)

##############################################################################

    # initialize the sponge field
    sponge_x_thickness = 0.055
    sponge_y_thickness = 0.055

    xMaxLoc = x0_sponge + sponge_x_thickness
    yMinLoc = 0.04

    sponge_init = InitSponge(amplitude=sponge_amp,
        x_max=xMaxLoc, y_min=yMinLoc,
        x_thickness=sponge_x_thickness,
        y_thickness=sponge_y_thickness)

    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=fluid_nodes))
    ref_cv = force_evaluation(actx, ref_state(actx, fluid_nodes, eos))

##############################################################################

    inflow_nodes = force_evaluation(actx,
                                    dcoll.nodes(dd_vol_fluid.trace('inlet')))
    inflow_temperature = inflow_nodes[0]*0.0 + 300.0
    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return inflow_temperature

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_cv_cond = ref_state(actx=actx, x_vec=inflow_nodes, eos=eos,
                                   state_minus=state_minus)
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
                                temperature_seed=300.0)

    from mirgecom.inviscid import inviscid_flux
    from mirgecom.flux import num_flux_central
    from mirgecom.viscous import viscous_flux

    """
    """
    class MyPrescribedBoundary(PrescribedFluidBoundary):
        r"""My prescribed boundary function. """

        def __init__(self, bnd_state_func, temperature_func):
            """Initialize the boundary condition object."""
            self.bnd_state_func = bnd_state_func
            PrescribedFluidBoundary.__init__(
               self,
               boundary_state_func=bnd_state_func,
               inviscid_flux_func=self.inviscid_wall_flux,
               viscous_flux_func=self.viscous_wall_flux,
               boundary_temperature_func=temperature_func,
               boundary_gradient_cv_func=self.grad_cv_bc,
            )

        def prescribed_state_for_advection(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
            state_plus = self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                             state_minus, **kwargs)

            mom_x = -state_minus.cv.momentum[0]
            mom_y = 2.0*state_plus.cv.momentum[1] - state_minus.cv.momentum[1]
            mom_plus = make_obj_array([mom_x, mom_y])

            kin_energy_ref = 0.5*np.dot(state_plus.cv.momentum, state_plus.cv.momentum)/state_plus.cv.mass
            kin_energy_mod = 0.5*np.dot(mom_plus, mom_plus)/state_plus.cv.mass
            energy_plus = state_plus.cv.energy - kin_energy_ref + kin_energy_mod

            cv = make_conserved(
                dim=2, mass=state_plus.cv.mass, energy=energy_plus,
                momentum=mom_plus, species_mass=state_plus.cv.species_mass)

            return make_fluid_state(cv=cv, gas_model=gas_model, temperature_seed=300.0)

        def prescribed_state_for_diffusion(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
            return self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                       state_minus, **kwargs)

        def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                numerical_flux_func, **kwargs):

            state_plus = self.prescribed_state_for_advection(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model, 
                state_minus=state_minus, **kwargs)

            state_pair = TracePair(dd_bdry, interior=state_minus,
                                   exterior=state_plus)

            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            actx = state_pair.int.array_context
            lam = actx.np.maximum(state_pair.int.wavespeed,
                                  state_pair.ext.wavespeed)
            from mirgecom.flux import num_flux_lfr
            return num_flux_lfr(
                f_minus_normal=inviscid_flux(state_pair.int)@normal,
                f_plus_normal=inviscid_flux(state_pair.ext)@normal,
                q_minus=state_pair.int.cv,
                q_plus=state_pair.ext.cv, lam=lam)

        def grad_cv_bc(
                self, state_plus, state_minus, grad_cv_minus, normal, **kwargs):
            """Return grad(CV) for boundary calculation of viscous flux."""
            return grad_cv_minus

        def viscous_wall_flux(
                self, dcoll, dd_bdry, gas_model, state_minus,
                grad_cv_minus, grad_t_minus, numerical_flux_func, **kwargs):
            """Return the boundary flux for viscous flux."""
            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            state_plus = self.prescribed_state_for_diffusion(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, **kwargs)

            grad_cv_plus = self.grad_cv_bc(
                state_plus=state_plus, state_minus=state_minus,
                grad_cv_minus=grad_cv_minus, normal=normal, **kwargs)

            grad_t_plus = self._bnd_grad_temperature_func(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                grad_t_minus=grad_t_minus)

            # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
            # *not* the numerical viscous flux as advised by [Bassi_1997]_.
            f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                                 grad_t=grad_t_plus)
            return f_ext@normal

    linear_bnd = LinearizedOutflowBoundary(
        free_stream_density=rho_atmosphere, free_stream_pressure=101325.0,
        free_stream_velocity=np.zeros(shape=(dim,)),
        free_stream_species_mass_fractions=y_atmosphere)

    fluid_boundaries = {
        dd_vol_fluid.trace('inlet').domain_tag:
            MyPrescribedBoundary(bnd_state_func=inlet_bnd_state_func,
                                 temperature_func=bnd_temperature_func),
        dd_vol_fluid.trace('symmetry').domain_tag:
            AdiabaticSlipBoundary(),
        dd_vol_fluid.trace('burner').domain_tag:
            AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace('linear').domain_tag:
            linear_bnd,
        dd_vol_fluid.trace('outlet').domain_tag:
            PressureOutflowBoundary(boundary_pressure=101325.0),
    }

    wall_symmetry = NeumannDiffusionBoundary(0.0)
    solid_boundaries = {
        dd_vol_solid.trace('wall_sym').domain_tag: wall_symmetry
    }

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(
        dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=current_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=current_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

#########################################################################

    def my_write_viz(
            step, t, dt, fluid_state, solid_state, smoothness=None,
            grad_cv_fluid=None, grad_t_fluid=None, grad_t_solid=None):

#        heat_rls = pyrometheus_mechanism.heat_release(fluid_state)

        fluid_viz_fields = [
            ('CV_rho', fluid_state.cv.mass),
            ('CV_rhoU', fluid_state.cv.momentum),
            ('CV_rhoE', fluid_state.cv.energy),
            ('DV_P', fluid_state.pressure),
            ('DV_T', fluid_state.temperature),
            ('DV_U', fluid_state.velocity[0]),
            ('DV_V', fluid_state.velocity[1]),
#            ('dt', dt[0] if local_dt else None),
            ('sponge', sponge_sigma),
            ('smoothness', 1.0 - theta_factor*smoothness),
            ('RR', chem_rate*reaction_rates_damping),
        ]

        if grad_cv_fluid is not None:
            fluid_viz_fields.extend((
                ('fluid_grad_cv_rho', grad_cv_fluid.mass),
                ('fluid_grad_cv_rhoU', grad_cv_fluid.momentum[0]),
                ('fluid_grad_cv_rhoV', grad_cv_fluid.momentum[1]),
                ('fluid_grad_cv_rhoE', grad_cv_fluid.energy),
            ))

        if grad_t_fluid is not None:
            fluid_viz_fields.append(('fluid_grad_t', grad_t_fluid))

        # species mass fractions
        fluid_viz_fields.extend((
            'Y_'+species_names[i], fluid_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        wv = solid_state.cv
        wdv = solid_state.dv
        solid_mass_rhs = decomposition.get_source_terms(
            temperature=wdv.temperature, chi=wv.mass)
        solid_viz_fields = [
            ('wv_energy', wv.energy),
            ('cfl', solid_zeros),  # FIXME
#            ('wall_h', wdv.enthalpy),
#            ('wall_cp', wdv.heat_capacity),
            ('wall_kappa', wdv.thermal_conductivity),
            ('wall_alpha', solid_wall_model.thermal_diffusivity(solid_state)),
            ('wall_temperature', wdv.temperature),
            ('wall_grad_t', grad_t_solid),
#            ('dt', dt[2] if local_dt else None),
        ]

        if wv.mass.shape[0] > 1:
            solid_viz_fields.extend(('wv_mass_' + str(i), wv.mass[i])
                                     for i in range(wv.mass.shape[0]))
            solid_viz_fields.extend(('mass_rhs' + str(i), solid_mass_rhs[i])
                                     for i in range(wv.mass.shape[0]))
        else:
            solid_viz_fields.append(('wv_mass', wv.mass))

        if grad_t_solid is not None:
            solid_viz_fields.append(('solid_grad_t', grad_t_solid))

        print('Writing solution file...')
        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+'-fluid', step=step, t=t,
            overwrite=True, comm=comm)
        write_visfile(
            dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+'-wall', step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        if rank == 0:
            print('Writing restart file...')

        cv, tseed, wv, wv_tseed, _ = state
        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                'volume_to_local_mesh_data': volume_to_local_mesh_data,
                'cv': cv,
                'temperature_seed': tseed,
                'nspecies': nspecies,
                'wv': wv,
                'wall_temperature_seed': wv_tseed,
                't': t,
                'step': step,
                'order': order,
                'global_nelements': global_nelements,
                'num_parts': nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

#########################################################################

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(dcoll, 'vol', pressure),
                         op='lor'):
            health_error = True
            logger.info(f'{rank=}: NANs/Infs in pressure data.')

        if global_reduce(check_naninf_local(dcoll, 'vol', temperature),
                         op='lor'):
            health_error = True
            logger.info(f'{rank=}: NANs/Infs in temperature data.')

        return health_error

##############################################################################

    from mirgecom.boundary import DummyBoundary
    from mirgecom.diffusion import (
        DiffusionBoundary, grad_facial_flux_weighted,
        diffusion_facial_flux_harmonic)

    class DummyDiffusionBoundary(DiffusionBoundary):
        """."""
        def get_grad_flux(self, dcoll, dd_bdry, kappa_minus, u_minus, *,
                numerical_flux_func=grad_facial_flux_weighted):
            return None
        def get_diffusion_flux(
                self, dcoll, dd_bdry, kappa_minus, u_minus,
                grad_u_minus, lengthscales_minus, *, penalty_amount=None,
                numerical_flux_func=diffusion_facial_flux_harmonic):
            return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from arraycontext import outer
    from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
    from grudge import op
    from meshmode.discretization.connection import FACE_RESTR_ALL

    fluid_field = fluid_nodes[0]*0.0
    wall_field = solid_nodes[0]*0.0
    pairwise_field = {
        (dd_vol_fluid, dd_vol_solid): (fluid_field, wall_field)}
    pairwise_field_tpairs = inter_volume_trace_pairs(
        dcoll, pairwise_field, comm_tag=_MyGradTag_Bdry)
    field_tpairs_F = pairwise_field_tpairs[dd_vol_solid, dd_vol_fluid]
    field_tpairs_W = pairwise_field_tpairs[dd_vol_fluid, dd_vol_solid]

    axisym_fluid_boundaries = {}
    axisym_fluid_boundaries.update(fluid_boundaries)
    axisym_fluid_boundaries.update({
            tpair.dd.domain_tag: DummyBoundary()
            for tpair in field_tpairs_F})

    axisym_wall_boundaries = {}
    axisym_wall_boundaries.update(solid_boundaries)
    axisym_wall_boundaries.update({
            tpair.dd.domain_tag: DummyDiffusionBoundary()
            for tpair in field_tpairs_W})

    def my_derivative_function(actx, dcoll, field, field_bounds, dd_vol,
                               bnd_cond, comm_tag):    

        dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

        interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll,
                                      quadrature_tag)

        def interior_flux(field_tpair):
            dd_trace_quad = field_tpair.dd.with_discr_tag(quadrature_tag)
            normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
            bnd_tpair_quad = interp_to_surf_quad(field_tpair)
            flux_int = outer(num_flux_central(bnd_tpair_quad.int,
                                              bnd_tpair_quad.ext),
                             normal_quad)

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad, flux_int)

        def boundary_flux(bdtag, bdry):
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = actx.thaw(dcoll.normal(dd_bdry_quad)) 
            int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field)

            if bnd_cond == 'symmetry' and bdtag == '-0':
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    off_axis_x = 1e-7
    fluid_nodes_are_off_axis = actx.np.greater(fluid_nodes[0], off_axis_x)
    solid_nodes_are_off_axis = actx.np.greater(solid_nodes[0], off_axis_x)

    def axisym_source_fluid(actx, dcoll, state, grad_cv, grad_t):
        cv = state.cv
        dv = state.dv

        mu = state.tv.viscosity
        beta = physical_transport.volume_viscosity(cv, dv, gas_model.eos)
        kappa = state.tv.thermal_conductivity
        d_ij = state.tv.species_diffusivity

        grad_v = velocity_gradient(cv, grad_cv)
        grad_y = species_mass_fraction_gradient(cv, grad_cv)

        u = state.velocity[0]
        v = state.velocity[1]

        dudr = grad_v[0][0]
        dudy = grad_v[0][1]
        dvdr = grad_v[1][0]
        dvdy = grad_v[1][1]

        drhoudr = (grad_cv.momentum[0])[0]

        d2udr2   = my_derivative_function(actx, dcoll, dudr, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_1)[0] #XXX
        d2vdr2   = my_derivative_function(actx, dcoll, dvdr, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_2)[0] #XXX
        d2udrdy  = my_derivative_function(actx, dcoll, dudy, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_3)[0] #XXX
                
        dmudr    = my_derivative_function(actx, dcoll,   mu, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_4)[0]
        dbetadr  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_5)[0]
        dbetady  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_6)[1]
        
        qr = - kappa*grad_t[0]
        dqrdr = 0.0 #- (dkappadr*grad_t[0] + kappa*d2Tdr2) #XXX
        
        dyidr = grad_y[:,0]
        #dyi2dr2 = my_derivative_function(actx, dcoll, dyidr, 'replicate')[:,0]   #XXX
        
        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              fluid_nodes_are_off_axis, u/fluid_nodes[0], dudr )

        dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

        """
        """
        source_mass_dom = - cv.momentum[0]

        source_rhoU_dom = - cv.momentum[0]*u \
                          + tau_rr - tau_tt \
                          + u*dbetadr + beta*dudr \
                          + beta*actx.np.where(
                              fluid_nodes_are_off_axis, -u/fluid_nodes[0], -dudr )
                              
        source_rhoV_dom = - cv.momentum[0]*v \
                          + tau_ry \
                          + u*dbetady + beta*dudy
        
        source_rhoE_dom = -( (cv.energy+dv.pressure)*u + qr ) \
                          + u*tau_rr + v*tau_ry \
                          + u**2*dbetadr + beta*2.0*u*dudr \
                          + u*v*dbetady + u*beta*dvdy + v*beta*dudy

        source_spec_dom = - cv.species_mass*u + d_ij*dyidr
        """
        """

        source_mass_sng = - drhoudr
        source_rhoU_sng = 0.0  # mu*d2udr2 + 0.5*beta*d2udr2  #XXX
        source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady
        source_rhoE_sng = -( (cv.energy+dv.pressure)*dudr + dqrdr ) \
                                + tau_rr*dudr + v*dtaurydr \
                                + 2.0*beta*dudr**2 \
                                + beta*dudr*dvdy \
                                + v*dudr*dbetady \
                                + v*beta*d2udrdy
        source_spec_sng = - cv.species_mass*dudr #+ d_ij*dyi2dr2
        
        """
        """
        source_mass = actx.np.where( fluid_nodes_are_off_axis,
                          source_mass_dom/fluid_nodes[0], source_mass_sng )
        source_rhoU = actx.np.where( fluid_nodes_are_off_axis,
                          source_rhoU_dom/fluid_nodes[0], source_rhoU_sng )
        source_rhoV = actx.np.where( fluid_nodes_are_off_axis,
                          source_rhoV_dom/fluid_nodes[0], source_rhoV_sng )
        source_rhoE = actx.np.where( fluid_nodes_are_off_axis,
                          source_rhoE_dom/fluid_nodes[0], source_rhoE_sng )
        source_spec = make_obj_array([
                      actx.np.where( fluid_nodes_are_off_axis,
                          source_spec_dom[i]/fluid_nodes[0], source_spec_sng[i] )
                      for i in range(nspecies)])
        
        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                       momentum=make_obj_array([source_rhoU, source_rhoV]),
                       species_mass=source_spec)

#    compiled_axisym_source_fluid = actx.compile(axisym_source_fluid)

    # ~~~~~~~
    def axisym_source_solid(actx, dcoll, solid_state, grad_t):
        dkappadr = 0.0*solid_nodes[0]

        temperature = solid_state.dv.temperature
        kappa = solid_state.dv.thermal_conductivity
        
        qr = - kappa*grad_t[0]
#        d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0], axisym_wall_boundaries, dd_vol_solid,  'symmetry')[0]
#        dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)
                
        source_mass = solid_state.cv.mass*0.0

        source_rhoE_dom = - qr
        source_rhoE_sng = 0.0 #- dqrdr
        source_rhoE = actx.np.where( solid_nodes_are_off_axis,
                          source_rhoE_dom/solid_nodes[0], source_rhoE_sng)

        return SolidWallConservedVars(mass=source_mass, energy=source_rhoE)

    compiled_axisym_source_solid = actx.compile(axisym_source_solid)

    # ~~~~~~~
    #FIXME do I have to increase the gravity by speedup_factor?
    def gravity_source_terms(cv):
        """Gravity."""
        gravity = - 9.80665 * speedup_factor 
        delta_rho = cv.mass - rho_atmosphere
        return make_conserved(
            dim=2,
            mass=fluid_zeros,
            energy=delta_rho*cv.velocity[1]*gravity,
            momentum=make_obj_array([fluid_zeros, delta_rho*gravity]),
            species_mass=cv.species_mass*0.0)

    # ~~~~~~~
    def chemical_source_term(cv, temperature):
        return chem_rate*speedup_factor*reaction_rates_damping*(
            eos.get_species_source_terms(cv, temperature))

    # ~~~~~~

    from grudge.discretization import filter_part_boundaries
    from grudge.reductions import integral
    solid_dd_list = filter_part_boundaries(dcoll, volume_dd=dd_vol_solid,
                                           neighbor_volume_dd=dd_vol_fluid)
    fluid_dd_list = filter_part_boundaries(dcoll, volume_dd=dd_vol_fluid,
                                           neighbor_volume_dd=dd_vol_solid)

    interface_nodes = op.project(
        dcoll, dd_vol_solid, solid_dd_list[0], solid_nodes*wall_sample_mask)

    interface_zeros = actx.np.zeros_like(interface_nodes[0])

    interface_sample = op.project(
        dcoll, dd_vol_solid, solid_dd_list[0], wall_sample_mask)

    # surface integral of the density
    # dS = 2 pi r dx
    dS = 2.0*np.pi*interface_nodes[0]
    dV = 2.0*np.pi*solid_nodes[0]

    integral_volume = integral(dcoll, dd_vol_solid, wall_sample_mask*dV)
    integral_surface = integral(dcoll, solid_dd_list[0], dS)

    radius = 0.015875
    height = 0.01905

    volume = np.pi*radius**2*height
    area = np.pi*radius**2 + 2.0*np.pi*radius*height

#    print(integral_volume - volume)
#    print(integral_surface - area)

#    assert integral_volume - volume < 1e-9
#    assert integral_surface - area < 1e-9

    def blowing_velocity(fluid_mass, source):

        # volume integral of the source terms
        integral_volume_source = \
            integral(dcoll, dd_vol_solid, source*wall_sample_mask*dV)

        # restrict to coupled surface 
        surface_density = op.project(
            dcoll, dd_vol_fluid, fluid_dd_list[0], fluid_mass) 

        # surface integral of the density
        integral_surface_density = \
            integral(dcoll, solid_dd_list[0], surface_density*dS)

        velocity = \
            integral_volume_source/integral_surface_density*interface_sample

        #since I am prescribing the flux, I have to multiply by the normal
        return force_evaluation(actx, velocity)


##############################################################################

    from grudge.dof_desc import DD_VOLUME_ALL
    def my_get_wall_timestep(solid_state):
        return 1e-8 + solid_zeros
#        wall_diffusivity = solid_wall_model.thermal_diffusivity(solid_state)
#        return char_length_solid**2/(wall_diffusivity)

    def _my_get_timestep_wall(solid_state, t, dt):
        return current_cfl*my_get_wall_timestep(solid_state)

    def _my_get_timestep_fluid(fluid_state, t, dt):
        return get_sim_timestep(
            dcoll, fluid_state, t, dt, current_cfl, gas_model,
            constant_cfl=constant_cfl, local_dt=local_dt, fluid_dd=dd_vol_fluid)

    my_get_timestep_wall = actx.compile(_my_get_timestep_wall)
    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)


    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        cv, tseed, wv, wv_tseed, _ = state

        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)
        wv = force_evaluation(actx, wv)
        wv_tseed = force_evaluation(actx, wv_tseed)

        # include both outflow and sponge in the damping region
        smoothness = force_evaluation(actx,
            smooth_region + sponge_sigma/sponge_amp)

        # damp the outflow
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed)
        cv = fluid_state.cv

        # wall variables

        solid_state = get_solid_state(wv, wv_tseed)
        wv = solid_state.cv
        wdv = solid_state.dv

        if my_material == "fiber":
            boundary_velocity = interface_zeros

        if my_material == "composite":
            solid_mass_rhs = decomposition.get_source_terms(
                temperature=solid_state.dv.temperature, chi=solid_state.cv.mass)

            boundary_velocity = \
                speedup_factor * blowing_velocity(fluid_state.cv.mass,
                                                  -1.0*sum(solid_mass_rhs))

        t = force_evaluation(actx, t)
        dt_fluid = force_evaluation(actx, actx.np.minimum(
            current_dt, my_get_timestep_fluid(fluid_state, t[0], dt[0])))
#        dt_solid = force_evaluation(actx, actx.np.minimum(
#            1.0e-8, my_get_timestep_wall(solid_state, t[2], dt[2])))
        dt_solid = force_evaluation(actx, 1e-8 + solid_zeros)
        dt = make_obj_array([dt_fluid, fluid_zeros, dt_solid, solid_zeros,
                             interface_zeros])

        try:
            state = make_obj_array([
                fluid_state.cv, fluid_state.dv.temperature,
                solid_state.cv, solid_state.dv.temperature,
                boundary_velocity])

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            ngarbage = 50
            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn('Running gc.collect() to work around memory growth issue ')
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
                warn(f'Lazy does not like the health_check', stacklevel=2)
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op='lor')
                if health_errors:
                    if rank == 0:
                        logger.info('Fluid solution failed health check.')
                    raise MyRuntimeError('Failed simulation health check.')

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    solid_state=solid_state, smoothness=smoothness)

            if do_restart:
                my_write_restart(step, t, state)

        except MyRuntimeError:
            if rank == 0:
                logger.info('Errors detected; attempting graceful exit.')
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                solid_state=solid_state, smoothness=smoothness)
            raise

        return state, dt


    def _get_rhs(t, state):

        fluid_state, solid_state, boundary_velocity = state

        cv = fluid_state.cv
        wv = solid_state.cv
        wdv = solid_state.dv

        #~~~~~~~~~~~~~
        fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
            add_thermal_interface_boundaries_no_grad(
                dcoll, gas_model,
                dd_vol_fluid, dd_vol_solid,
                fluid_state, wdv.thermal_conductivity, wdv.temperature,
                boundary_velocity,
                fluid_boundaries, solid_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                )

        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_fluid, comm_tag=_FluidOpStatesTag,
            limiter_func=_limit_fluid_cv)

        # fluid grad CV
        fluid_grad_cv = grad_cv_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=t, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradCVTag
        )

        # fluid grad T
        fluid_grad_temperature = grad_t_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=t, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradTempTag
        )

        # solid grad T
        solid_grad_temperature = wall_grad_t_operator(
            dcoll, wdv.thermal_conductivity,
            solid_all_boundaries_no_grad, wdv.temperature,
            quadrature_tag=quadrature_tag, dd=dd_vol_solid,
            comm_tag=_SolidGradTempTag
        )

        fluid_all_boundaries, solid_all_boundaries = \
            add_thermal_interface_boundaries(
                dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
                fluid_state, wdv.thermal_conductivity,
                wdv.temperature,
                boundary_velocity,
                fluid_grad_temperature, solid_grad_temperature,
                fluid_boundaries, solid_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                wall_emissivity=emissivity, sigma=5.67e-8,
                ambient_temperature=300.0,
                wall_penalty_amount=wall_penalty_amount)

        fluid_rhs = ns_operator(
            dcoll, gas_model, fluid_state, fluid_all_boundaries,
            time=t, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
            comm_tag=_FluidOperatorTag, inviscid_terms_on=True)

        fluid_sources = (
            chemical_source_term(fluid_state.cv, fluid_state.temperature)
            + sponge_func(cv=fluid_state.cv, cv_ref=ref_cv,
                          sigma=sponge_sigma)
            + gravity_source_terms(fluid_state.cv)
            + axisym_source_fluid(actx, dcoll, fluid_state, fluid_grad_cv,
                                  fluid_grad_temperature)
        )

        #~~~~~~~~~~~~~
        if ignore_wall:
            solid_rhs = SolidWallConservedVars(mass=wv.mass*0.0,
                                               energy=solid_zeros)
            solid_sources = 0.0

        else:
            solid_mass_rhs = decomposition.get_source_terms(
                temperature=solid_state.dv.temperature,
                chi=solid_state.cv.mass)

            solid_energy_rhs = diffusion_operator(
                dcoll, wdv.thermal_conductivity, solid_all_boundaries,
                wdv.temperature,
                penalty_amount=wall_penalty_amount,
                quadrature_tag=quadrature_tag,
                dd=dd_vol_solid,
                grad_u=solid_grad_temperature,
                comm_tag=_SolidOperatorTag
            )

            solid_sources = axisym_source_solid(
                actx, dcoll, solid_state, solid_grad_temperature)

            solid_rhs = wall_time_scale * SolidWallConservedVars(
                mass=solid_mass_rhs*wall_sample_mask,
                energy=solid_energy_rhs)

        #~~~~~~~~~~~~~

#        from pytato.analysis import get_max_node_depth
#        print(f"{get_max_node_depth(fluid_rhs.mass[0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.energy[0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.momentum[0][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.momentum[1][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[0][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.mass[0])=}")
#        print(f"{get_max_node_depth(fluid_sources.energy[0])=}")
#        print(f"{get_max_node_depth(fluid_sources.momentum[0][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.momentum[1][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[0][0])=}")
#        print('')
#        print(f"{get_max_node_depth(solid_rhs.mass[0][0])=}")
#        print(f"{get_max_node_depth(solid_rhs.mass[1][0])=}")
#        print(f"{get_max_node_depth(solid_rhs.mass[2][0])=}")
#        print(f"{get_max_node_depth(solid_rhs.energy[0])=}")
#        print(f"{get_max_node_depth(solid_sources.mass[0][0])=}")
#        print(f"{get_max_node_depth(solid_sources.mass[1][0])=}")
#        print(f"{get_max_node_depth(solid_sources.mass[2][0])=}")
#        print(f"{get_max_node_depth(solid_sources.energy[0])=}")

#        def get_node_count(ary):
#            if not isinstance(ary, DOFArray):
#                from arraycontext import map_reduce_array_container
#                return map_reduce_array_container(sum, get_node_count, ary)

#            from pytato.analysis import get_num_nodes
#            return get_num_nodes(ary[0])

#        print('')
#        print(f"{get_node_count(fluid_rhs)=}")
#        print(f"{get_node_count(fluid_sources)=}")
#        print(f"{get_node_count(solid_rhs)=}")
#        print(f"{get_node_count(solid_sources)=}")

#        sys.exit()

        #~~~~~~~~~~~~~

        return make_obj_array([fluid_rhs + fluid_sources, fluid_zeros,
                               solid_rhs + solid_sources, solid_zeros,
                               interface_zeros])

    get_rhs_compiled = actx.compile(_get_rhs)

    def my_rhs(t, state):
        cv, tseed, wv, wv_tseed, boundary_velocity = state

        t = force_evaluation(actx, t)
        smoothness = force_evaluation(actx, smooth_region + sponge_sigma/sponge_amp)

        # apply outflow damping
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed)
        fluid_state = force_evaluation(actx, fluid_state)
        cv = fluid_state.cv

        # construct wall state
        solid_state = get_solid_state(wv, wv_tseed)
        solid_state = force_evaluation(actx, solid_state)
        wv = solid_state.cv
        wdv = solid_state.dv

        boundary_velocity = force_evaluation(actx, boundary_velocity)

        actual_state = make_obj_array([fluid_state, solid_state,
                                       boundary_velocity])

        rhs_state = get_rhs_compiled(t, actual_state)

#        if use_rhs_filtered:
#            filtered_sample_rhs = filter_sample_rhs_compiled(rhs_state[2])
#            return make_obj_array([
#                rhs_state[0], fluid_zeros,
#                filtered_sample_rhs, sample_zeros, rhs_state[4],
#                rhs_state[5], interface_zeros])

        return rhs_state

    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer.start_sub_timer():
                import gc
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info('Freezing GC objects to reduce overhead of '
                            'future GC collections')
                gc.freeze()

        min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([current_fluid_state.cv,
                                    current_fluid_state.dv.temperature,
                                    current_solid_state.cv,
                                    current_solid_state.dv.temperature,
                                    interface_zeros])

    dt_fluid = force_evaluation(actx, actx.np.minimum(
        current_dt, my_get_timestep_fluid(current_fluid_state,
                        force_evaluation(actx, current_t + fluid_zeros),
                        force_evaluation(actx, current_dt + fluid_zeros))))
    dt_solid = force_evaluation(actx, 1.0e-8 + solid_zeros)
    dt = make_obj_array([dt_fluid, fluid_zeros, dt_solid, solid_zeros, interface_zeros])

    t_fluid = force_evaluation(actx, current_t + fluid_zeros)
    t_solid = force_evaluation(actx, current_t + solid_zeros)
    t = make_obj_array([t_fluid, t_fluid, t_solid, t_solid, interface_zeros])

    if rank == 0:
        logging.info('Stepping.')

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt, t=t, t_final=t_final,
                      max_steps=niter, local_dt=local_dt,
                      force_eval=force_eval, state=stepper_state,
                      compile_rhs=False)

    # 
    final_cv, tseed, final_wv, wv_tseed = stepper_state
    final_state = get_fluid_state(final_cv, tseed)

    final_wdv = wall_model.dependent_vars(final_wv, wv_tseed)

    # Dump the final data
    if rank == 0:
        logger.info('Checkpointing final state ...')

    my_write_restart(step=final_step, t=final_t, state=stepper_state)

    my_write_viz(step=final_step, t=final_t, dt=current_dt,
                 cv=final_state.cv, dv=current_state.dv,
                 wv=final_wv, wdv=final_wdv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == '__main__':
    import sys
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description='MIRGE-Com 1D Flame Driver')
    parser.add_argument('-r', '--restart_file',  type=ascii,
                        dest='restart_file', nargs='?', action='store',
                        help='simulation restart file')
    parser.add_argument('-i', '--input_file',  type=ascii,
                        dest='input_file', nargs='?', action='store',
                        help='simulation config file')
    parser.add_argument('-c', '--casename',  type=ascii,
                        dest='casename', nargs='?', action='store',
                        help='simulation case name')
    parser.add_argument('--profiling', action='store_true', default=False,
        help='enable kernel profiling [OFF]')
    parser.add_argument('--log', action='store_true', default=True,
        help='enable logging profiling [ON]')
    parser.add_argument("--esdg", action='store_true',
        help='use flux-differencing/entropy stable DG for inviscid computations.')
    parser.add_argument('--lazy', action='store_true', default=False,
        help='enable lazy evaluation [OFF]')
    parser.add_argument('--numpy', action='store_true',
        help='use numpy-based eager actx.')

    args = parser.parse_args()

    # for writing output
    casename = 'burner_mix'
    if(args.casename):
        print(f'Custom casename {args.casename}')
        casename = (args.casename).replace("'", '')
    else:
        print(f'Default casename {casename}')

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", '')
        print(f'Restarting from file: {restart_file}')

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", '')
        print(f'Reading user input from {args.input_file}')
    else:
        print('No user input file, using default values')

    print(f'Running {sys.argv[0]}\n')

    from warnings import warn
    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError('ESDG requires lazy or numpy context.')
        if not args.overintegration:
            warn('ESDG requires overintegration, enabling --overintegration.')

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    main(actx_class, use_logmgr=args.log, casename=casename, 
         restart_filename=restart_file)
