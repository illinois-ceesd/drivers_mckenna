"""Mon 12 Sep 2022 09:02:22 AM CDT"""

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
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce,
    force_evaluation
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.integrators import euler_step
from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    IsothermalWallBoundary,
    #PressureOutflowBoundary
    OutflowBoundary,
    SymmetryBoundary,
    PrescribedFluidBoundary,
    MyPrescribedBoundary_v3,
    AdiabaticNoslipWallBoundary,
    LinearizedBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import (
    #SimpleTransport,
    PowerLawTransport,
    ArtificialViscosityTransport,
    #MixtureAveragedTransport
)
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state, ViscousFluidState
import cantera

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)

from pytools.obj_array import make_obj_array

from mirgecom.fluid import velocity_gradient, species_mass_fraction_gradient

#########################################################################

class Burner2D_Reactive:

    def __init__(
            self, *, dim=2,
                     nspecies=7,
                     sigma=0.001,
                     sigma_flame = 0.00020,
                     flame_loc=None,
                     pressure,
                     temperature,
                     speedup_factor,
                     mass_atm, mass_srd,
                     mass_burned, mass_unburned,
                     species_unburn,
                     species_burned,
                     species_shroud,
                     species_atm):

        self._dim = dim
        self._nspecies = nspecies 
        self._sigma = sigma
        self._sigma_flame = sigma_flame
        self._pres = pressure
        self._speedup_factor = speedup_factor
        self._mass_burned = mass_burned
        self._mass_unburned = mass_unburned
        self._mass_srd = mass_srd
        self._mass_atm = mass_atm
        self._yu = species_unburn
        self._yb = species_burned
        self._ys = species_shroud
        self._ya = species_atm
        self._temp = temperature
        self._flaLoc = flame_loc

        if (flame_loc is None):
            raise ValueError(f"Specify flame_loc")

    def __call__(self, x_vec, eos, time=None, state_minus=None, init=False):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        actx = x_vec[0].array_context

        time_func = 1.0

        _speedup_factor = self._speedup_factor

        _ya = self._ya
        _ys = self._ys
        _yb = self._yb
        _yu = self._yu
        cool_temp = 300.0

        mass_prd = self._mass_burned
        mass_mix = self._mass_unburned
        mass_srd = self._mass_srd
        mass_atm = self._mass_atm

        sigma_factor = 10.0 - 9.0*(0.12 - x_vec[1])**2/(0.12 - 0.10)**2
        _sigma = self._sigma*(
            actx.np.where(actx.np.less(x_vec[1], .12),
                              actx.np.where(actx.np.greater(x_vec[1], .1),
                                            sigma_factor, 1.),
                              10.0)
        )

        int_diam = 0.030
        ext_diam = 0.035     

        #~~~ shroud
        S1 = 0.5*(1.0 + actx.np.tanh(1.0/(_sigma)*(x_vec[0] - int_diam)))
        S2 = actx.np.where(actx.np.greater(x_vec[0], ext_diam),
                 0.0, - actx.np.tanh(1.0/(_sigma)*(x_vec[0] - ext_diam))
             )
        shroud = S1*S2

        #~~~ flame ignition
        core = 0.5*(1.0 - actx.np.tanh(1.0/_sigma*(x_vec[0] - int_diam)))
        flame = 0.5*(1.0 + actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - self._flaLoc)))
             
        #~~~ after combustion products
        upper_atm = 0.5*(1.0 + actx.np.tanh( 1.0/(2.0*_sigma)*(x_vec[1] - 0.12)))
        yf = (flame*_yb + (1.0-flame)*_yu)*(1.0-upper_atm) + _ya*upper_atm

        #~~~ shroud
        ys = _ya*upper_atm + (1.0 - upper_atm)*_ys

        #~~~ atmosphere
        atmosphere = 1.0 - (shroud + core)
        y = atmosphere*_ya + shroud*ys + core*yf

        #~~~ temperature
        temp = (flame*self._temp + (1.0-flame)*cool_temp)*(1.0-upper_atm) + 300.0*upper_atm
        temperature = temp*core + 300.0*(1.0 - core)

        #~~~ velocity
        v_inlet = 0.117892*_speedup_factor
        v_shroud = v_inlet*2.769230769230767
        smoothY = (flame*(self._temp/300.0)*v_inlet + (1.0-flame)*v_inlet)*(1.0-upper_atm) + 0.0*upper_atm
        u_x = 0.0*x_vec[0]
        u_y = core*smoothY + shroud*v_shroud*(1.0-upper_atm)
        velocity = make_obj_array([u_x, u_y])

        #~~~ 

        if init is True:
            pressure = self._pres + 0.0*x_vec[0]
            mass = eos.get_density(pressure, temperature, species_mass_fractions=y)

            specmass = mass * y

            internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
            kinetic_energy = 0.5 * np.dot(velocity, velocity)
            energy = mass * (internal_energy + kinetic_energy)

            return (
                make_conserved(dim=self._dim, mass=mass, energy=energy,
                               momentum=mass*velocity, species_mass=specmass),
                temperature
            )
        else:

            smoothY = (flame*mass_prd + (1.0-flame)*mass_mix)*(1.0-upper_atm) + mass_atm*upper_atm
            mass = core*smoothY + shroud*(1.0-upper_atm)*mass_srd + atmosphere*mass_atm + shroud*upper_atm*mass_atm

            specmass = mass * y

            internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
            kinetic_energy = 0.5 * np.dot(velocity, velocity)
            energy = mass * (internal_energy + kinetic_energy)

            return make_conserved(dim=self._dim, mass=mass, energy=energy,
                momentum=mass*velocity, species_mass=specmass)



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


def get_mesh(dim, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh_filename = "mesh_08m_ReactionRate-v2.msh"
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)

    return mesh


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""
    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x_min=None, x_max=None, y_min=None, y_max=None,
                 x_thickness=None, y_thickness=None, amplitude):
        r"""Initialize the sponge parameters.
        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_thickness = x_thickness
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec):
        """Create the sponge intensity at locations *x_vec*.
        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
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


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):

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

    rst_path = "restart_data/"
    viz_path = "viz_data_dummy/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 2500
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 1.0e-9
    t_final = 2.0

    niter = 3000001
    
    # discretization and model control
    order = 1

    local_dt = True
    constant_cfl = True
    current_cfl = 0.4

    speedup_factor = 7.5

    use_AV = True
    use_overintegration = False
    plot_gradients = False
    
    current_step = 0
    current_t = 0.0
    dim = 2

##########################################################################

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval = False

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        if (constant_cfl == False):
            print(f"\tt_final = {t_final}")
        else:
            print(f"\tconstant_cfl = {constant_cfl}")
            print(f"\tcurrent_cfl = {current_cfl}")
            print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")


##########################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    mechanism_file = "uiuc_based_on_usc_v5a"
    #mechanism_file = "uiuc_sharp"
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    air = "O2:1.0,N2:3.76"
    fuel = "C2H4:1,H2:1"
    cantera_soln.set_equivalence_ratio(phi=1.0, fuel=fuel, oxidizer=air)
    x_unburned = cantera_soln.X

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = 2400.0

    # one_atm = 101325.0
    pres_unburned = cantera.one_atm

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x_unburned}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x_unburned
    # Pull temperature, density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the gas state.
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY

    # now find the conditions for the burned gas
    cantera_soln.TPX = temp_ignition, pres_unburned, x_unburned
    cantera_soln.equilibrate("TP")
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P

    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    i_CO2 = cantera_soln.species_index("CO2")
    i_CO = cantera_soln.species_index("CO")

    x = np.zeros(nspecies)
    x[:] = 0.0
    x[i_ox] = 0.21
    x[i_di] = 0.79
    cantera_soln.TPX = temp_unburned, pres_unburned, x
    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_atmosphere = np.zeros(nspecies)
    dummy, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    cantera_soln.equilibrate("TP")
    temp_atmosphere, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    pres_atmosphere = cantera_soln.P
    
    y_shroud = y_atmosphere*0.0
    y_shroud[i_di] = 0.98
    y_shroud[i_CO2] = 0.01
    y_shroud[i_CO] = 0.01

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = \
        get_pyrometheus_wrapper_class_from_cantera(
             cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 1000.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    species_names = pyrometheus_mechanism.species_names

    # }}}    
    
    # {{{ Initialize transport model
#    physical_transport = MixtureAveragedTransport(pyrometheus_mechanism,
#                                                  lewis=np.ones((nspecies)),
#                                                  factor=speedup_factor)
    physical_transport = PowerLawTransport(lewis=np.ones((nspecies)),
                                           factor=speedup_factor)
    if use_AV:
        s0 = np.log10(1.0e-4 / np.power(order, 4))
        alpha = 1.0e-4
        kappa_av = 0.5
        av_species = 1.0e-4
    else:
        s0 = np.log10(1.0e-4 / np.power(order, 4))
        alpha = 0.0   
        kappa_av = 0.0
        av_species = 0.0

    def smoothness_indicator(dcoll, field, **kwargs):
        
        actx = field.array_context
        nodes = force_evaluation(actx, dcoll.nodes())
        xpos = nodes[0]
        ypos = nodes[1]

        return xpos*0.0

#        if use_AV is False: return xpos*0.0

#        y_max = 0.75
#        y_thickness = 0.15

#        smoothness = xpos*0.0

#        y0 = (y_max - y_thickness)
#        dy = +((ypos - y0)/y_thickness)
#        smoothness = smoothness + actx.np.where(
#            actx.np.greater(ypos, y0),
#                actx.np.where(actx.np.greater(ypos, y_max),
#                              1.0, 3.0*dy**2 - 2.0*dy**3),
#                0.0
#        )

#        return smoothness

    transport_model = \
        ArtificialViscosityTransport(physical_transport=physical_transport,
                                     nspecies=nspecies,
                                     av_mu=alpha, av_prandtl=0.71,
                                     av_species_diffusivity=av_species)
    # }}}

    gas_model = GasModel(eos=eos, transport=transport_model)

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned:")
    print(f"T = {temp_unburned}")
    print(f"Y = {y_unburned}")
    print(f" ")
    print(f"Burned:")
    print(f"T = {temp_burned}")
    print(f"Y = {y_burned}")
    print(f" ")
    print(f"Atmosphere:")
    print(f"T = {temp_atmosphere}")
    print(f"Y = {y_atmosphere}")

    def _get_fluid_state(cv, temp_seed, smoothness):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temp_seed,
                                smoothness=smoothness)

    get_fluid_state = actx.compile(_get_fluid_state)

##################################

    flow_init = Burner2D_Reactive(dim=dim,
                                  nspecies=nspecies,
                                  sigma=0.00020,
                                  sigma_flame=0.00075,
                                  temperature=temp_ignition,
                                  pressure=pres_atmosphere,
                                  flame_loc=0.103,
                                  speedup_factor=speedup_factor,
                                  mass_atm=rho_atmosphere,
                                  mass_srd=1.14231135,
                                  mass_burned=rho_burned,
                                  mass_unburned=1.1118397678902912,
                                  species_shroud=y_shroud,
                                  species_atm=y_atmosphere,
                                  species_unburn=y_unburned,
                                  species_burned=y_burned)

    ref_state = Burner2D_Reactive(dim=dim,
                                  nspecies=nspecies,
                                  sigma=0.00020,
                                  sigma_flame=0.00001,
                                  temperature=temp_ignition,
                                  pressure=pres_atmosphere,
                                  flame_loc=0.103,
                                  speedup_factor=speedup_factor,
                                  mass_atm=rho_atmosphere,
                                  mass_srd=1.14231135,
                                  mass_burned=rho_burned,
                                  mass_unburned=1.1118397678902912,
                                  species_shroud=y_shroud,
                                  species_atm=y_atmosphere,
                                  species_unburn=y_unburned,
                                  species_burned=y_burned)

##############################################################################

    restart_step = None
    if restart_file is None:        
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm, get_mesh(dim=dim))
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order, 
                                             mpi_communicator=comm)
    nodes = actx.thaw(dcoll.nodes())

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]


    from grudge.dt_utils import characteristic_lengthscales
    length_scales = characteristic_lengthscales(actx, dcoll)
    h_min = vol_min(length_scales)
    h_max = vol_max(length_scales)

    if rank == 0:
        print("----- Discretization info ----")
        print(f"Discr: {nodes.shape=}, {order=}, {h_min=}, {h_max=}")
    for i in range(nparts):
        if rank == i:
            print(f"{rank=},{local_nelements=},{global_nelements=}")
        comm.Barrier()

#########################################################################

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                               
    vis_timer = None
    #log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

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

        #logmgr_add_device_memory_usage(logmgr, queue)
        try:
            logmgr.add_watches(["memory_usage_python.max",
                                "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)
        
#################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv, tseed = flow_init(nodes, eos, init=True)
    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t))

##XXX        current_t = 0.0
##XXX        current_step = 0

        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol"))

            current_cv = connection(restart_data["state"])
            tseed = connection(restart_data["temperature_seed"])
        else:
            current_cv = restart_data["state"]
            tseed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)


    tseed = force_evaluation(actx, tseed)
    current_cv = force_evaluation(actx, current_cv)

    if True: #use_AV:
        smoothness = force_evaluation(actx, smoothness_indicator(
            dcoll, current_cv.mass, kappa=kappa_av, s0=s0))

    current_state = get_fluid_state(current_cv, tseed, smoothness=smoothness)
    current_state = force_evaluation(actx, current_state)

#####################################################################################

    inflow_btag = DTAG_BOUNDARY("inlet")
    inflow_bnd_discr = dcoll.discr_from_dd(inflow_btag)
    inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
    inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos)
    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
                                temperature_seed=300.0,
                                smoothness=inflow_nodes[0]*0.0)

    vel_ref = np.zeros(shape=(dim,))
    wall_bnd = AdiabaticNoslipWallBoundary()
    inflow_bnd = MyPrescribedBoundary_v3(bnd_state_func=inlet_bnd_state_func,
                                         wall_temperature=300.0)
    outflow_bnd = OutflowBoundary(boundary_pressure=101325.0)
    linear_bnd = LinearizedBoundary(dim=2, 
                                    ref_density=rho_atmosphere,
                                    ref_pressure=101325.0,
                                    ref_velocity=vel_ref,
                                    ref_species_mass_fractions=y_atmosphere)
    symmmetry_bnd = SymmetryBoundary()
    
    boundaries = {DTAG_BOUNDARY("inlet"): inflow_bnd,
                  DTAG_BOUNDARY("symmetry"): symmmetry_bnd,
                  DTAG_BOUNDARY("outlet"): outflow_bnd,
                  DTAG_BOUNDARY("sponge"): linear_bnd,
                  DTAG_BOUNDARY("wall"): wall_bnd}

#####################################################################################

    # initialize the sponge field
    #sponge_x_thickness = 0.125
    sponge_x_thickness = 0.05
    sponge_y_thickness = 0.05
    sponge_amp = 125.0

    #xMaxLoc = 0.275 + sponge_x_thickness
    xMaxLoc = 0.05 + sponge_x_thickness
    yMinLoc = 0.04

    sponge_init = InitSponge(amplitude=sponge_amp,
                             x_max=xMaxLoc,
                             y_min=yMinLoc,
                             x_thickness=sponge_x_thickness,
                             y_thickness=sponge_y_thickness
                             )

    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=nodes))

    ref_cv = ref_state(nodes, eos, time=current_t)
    ref_cv = force_evaluation(actx, ref_cv)

####################################################################################

    visualizer = make_visualizer(dcoll)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     t_initial=current_t,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

#########################################################################

#    def get_production_rates(cv, temperature):
#        return make_obj_array([eos.get_production_rates(cv, temperature)])
#    compute_production_rates = actx.compile(get_production_rates)

    from grudge.dt_utils import characteristic_lengthscales, dt_geometric_factors
    def my_write_viz(step, t, state, dt=None,
                     ns_rhs=None, chem_sources=None,
                     grad_cv=None, grad_t=None,
                     ref_cv=None, sources=None, plot_gradients=False):

        #reaction_rates, = compute_production_rates(state.cv, state.temperature)
        viz_fields = [("CV_rho", state.cv.mass),
                      ("CV_rhoU", state.cv.momentum),
                      ("CV_rhoE", state.cv.energy),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      ("DV_U", state.velocity[0]),
                      ("DV_V", state.velocity[1]),
                      ("DV_AV", state.dv.smoothness),
                      #("reaction_rates", reaction_rates),
                      ("dt", dt),
                      ("sponge", sponge_sigma),
                      ]

        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        if plot_gradients:

#            mu = state.tv.viscosity
#            beta = state.tv.volume_viscosity
#            kappa = state.tv.thermal_conductivity

            grad_v = velocity_gradient(state.cv, grad_cv)
#            dudr = grad_v[0][0]
#            dudy = grad_v[0][1]
#            dvdr = grad_v[1][0]
#            dvdy = grad_v[1][1]

#            zVort = dvdr - dudy   
#            tau_ry = 1.0*mu*(dudy + dvdr)

#            qr = kappa*grad_t[0]
#            dqrdr   = my_derivative_function(actx, dcoll,  qr)[0]

#            d2udr2  = my_derivative_function(actx, dcoll, dudr)[0]
#            d2vdy2  = my_derivative_function(actx, dcoll, dvdy)[1]

#            d2udrdy  = my_derivative_function(actx, dcoll, dudy)[0]
#            dtaurydr = my_derivative_function(actx, dcoll, tau_ry)[0]

            grad_P = my_derivative_function(actx, dcoll, state.dv.pressure)

            grad_y = species_mass_fraction_gradient(state.cv, grad_cv)
            dyi2dr2  = my_derivative_function(actx, dcoll, grad_y[:,0])[:,0]

            viz_ext = [
#                       ("NS_rhs", ns_rhs),
#                       ("Z_vort", zVort),
                       ("Grad_U", grad_v[0]),
                       ("Grad_V", grad_v[1]),
                       ("Grad_P", grad_P),
                       ("Grad_T", grad_t),
                       ("Grad_rhoU", grad_cv.momentum[0]),
                       ("Grad_rhoV", grad_cv.momentum[1]),
                       ("Grad_rho", grad_cv.mass),
                       ("Grad_rhoE", grad_cv.energy),
#                       ("axi_d2udrdy",d2udrdy),
#                       ("axi_qr",qr),
#                       ("axi_dqrdr",dqrdr),
#                       ("axi_tau_ry",tau_ry),
#                       ("axi_dtaurydr",dtaurydr)
                          ]
            viz_fields.extend(viz_ext)

            if sources is not None:
                viz_ext = [("source_mass", sources.mass),
                           ("source_momentum", sources.momentum),
                           ("source_energy", sources.energy)]
                viz_fields.extend(viz_ext)

            viz_fields.extend(
                ("Grad_Y_"+species_names[i], grad_y[i])
                    for i in range(nspecies))
            viz_fields.extend(
                ("axi_d2Ydx2_"+species_names[i], dyi2dr2[i])
                    for i in range(nspecies))
                      
        print('Writing solution file...')
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                "temperature_seed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            print('Writing restart file...')
            write_restart_file(actx, rst_data, rst_fname, comm)

#########################################################################

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

    from grudge.dof_desc import DOFDesc, as_dofdesc, DISCR_TAG_BASE            
    from mirgecom.flux import num_flux_central
    from arraycontext import outer
    from grudge.trace_pair import TracePair, interior_trace_pair
   #from grudge.trace_pair import TracePair, interior_trace_pairs
    from mirgecom.boundary import DummyBoundary
    from mirgecom.operators import grad_operator
    from arraycontext import outer
    from grudge import op

    def _elbnd_flux(dcoll, compute_interior_flux, compute_boundary_flux,
                int_tpair, boundaries):
        return (compute_interior_flux(int_tpair)
            + sum(compute_boundary_flux(btag) for btag in boundaries))
            
            
    def central_flux_interior(actx, dcoll, int_tpair):
        """Compute a central flux for interior faces."""       
        normal = actx.thaw(dcoll.normal(int_tpair.dd))
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        dd_all_faces = int_tpair.dd.with_dtag("all_faces")
        
        return op.project(dcoll, int_tpair.dd, dd_all_faces, flux_weak)


    def central_flux_boundary(actx, dcoll, field, btag):
        """Compute a central flux for boundary faces."""
        dd_base_vol = DOFDesc("vol")
#        bnd_solution_quad = op.project(dcoll, dd_base_vol,      
#              as_dofdesc(btag).with_discr_tag(quadrature_tag), field)
        bnd_solution_quad = op.project(dcoll, dd_base_vol, as_dofdesc(btag), field)     
        bnd_nhat = actx.thaw(dcoll.normal(btag))
        bnd_tpair = TracePair(btag, interior=bnd_solution_quad,
                                    exterior=bnd_solution_quad)
        flux_weak = outer(num_flux_central(bnd_tpair.int, bnd_tpair.ext), bnd_nhat)
        dd_all_faces = bnd_tpair.dd.with_dtag("all_faces")
        
        return op.project(dcoll, bnd_tpair.dd, dd_all_faces, flux_weak)


    field_bounds = {DTAG_BOUNDARY("symmetry"): DummyBoundary(),
                    DTAG_BOUNDARY("inlet"): DummyBoundary(),
                    DTAG_BOUNDARY("outlet"): DummyBoundary(),
                    DTAG_BOUNDARY("sponge"): DummyBoundary(),
                    DTAG_BOUNDARY("wall"): DummyBoundary()}
    def my_derivative_function(actx, dcoll, field):
        
        int_flux = partial(central_flux_interior, actx, dcoll)
        bnd_flux = partial(central_flux_boundary, actx, dcoll, field)        

        int_tpair = interior_trace_pair(dcoll, field) #XXX
        flux_bnd = _elbnd_flux(dcoll, int_flux, bnd_flux, int_tpair, field_bounds)

        dd_vol = as_dofdesc("vol")
        dd_faces = as_dofdesc("all_faces")
        
        return grad_operator(dcoll, dd_vol, dd_faces, field, flux_bnd)

    #########################################################################

    off_axis_x = 1e-7
    nodes_are_off_axis = actx.np.greater(nodes[0], off_axis_x)  # noqa
   
    def axisymmetry_source_terms(actx, dcoll, state, grad_cv, grad_t):

        nodes = actx.thaw(dcoll.nodes())
        
        cv = state.cv
        dv = state.dv
        
        mu = state.tv.viscosity
        beta = transport_model.volume_viscosity(cv, dv, eos)
        kappa = state.tv.thermal_conductivity
        d_ij = state.tv.species_diffusivity
        
        grad_v = velocity_gradient(cv,grad_cv)
        grad_y = species_mass_fraction_gradient(cv, grad_cv)

        u = state.velocity[0]
        v = state.velocity[1]

        dudr = grad_v[0][0]
        dudy = grad_v[0][1]
        dvdr = grad_v[1][0]
        dvdy = grad_v[1][1]
        
        drhoudr = (grad_cv.momentum[0])[0]

        d2udrdy = my_derivative_function(actx, dcoll, dudy)[0]
                
        dbetadr = my_derivative_function(actx, dcoll, beta)[0]
        dbetady = my_derivative_function(actx, dcoll, beta)[1]
        
        dyidr = grad_y[:,0]
        dyi2dr2 = my_derivative_function(actx, dcoll, dyidr)[:,0]   
        
        qr = kappa*grad_t[0]
        dqrdr  = my_derivative_function(actx, dcoll,  qr)[0]
        
        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              nodes_are_off_axis, u/nodes[0], dudr )

        dtaurydr = my_derivative_function(actx, dcoll, tau_ry)[0]

        """
        """
        source_mass_dom = - cv.momentum[0]

        source_rhoU_dom = - cv.momentum[0]*u \
                          + tau_rr - tau_tt \
                          + u*dbetadr + beta*dudr \
                          + beta*actx.np.where(
                              nodes_are_off_axis, -u/nodes[0], -dudr )
                              
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
        source_rhoU_sng = 0.0
        source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady
        source_rhoE_sng = -( (cv.energy+dv.pressure)*dudr + dqrdr ) \
                                + tau_rr*dudr + v*dtaurydr \
                                + 2.0*beta*dudr**2 \
                                + beta*dudr*dvdy \
                                + v*dudr*dbetady \
                                + v*beta*d2udrdy

        source_spec_sng = -( grad_cv.species_mass[:,0]*u \
                             + cv.species_mass*dudr ) #\
        #                        #+ d_ij*dyi2dr2 
        
        """
        """
        source_mass = actx.np.where( nodes_are_off_axis,
                          source_mass_dom/nodes[0], source_mass_sng )
        source_rhoU = actx.np.where( nodes_are_off_axis,
                          source_rhoU_dom/nodes[0], source_rhoU_sng )
        source_rhoV = actx.np.where( nodes_are_off_axis,
                          source_rhoV_dom/nodes[0], source_rhoV_sng )
        source_rhoE = actx.np.where( nodes_are_off_axis,
                          source_rhoE_dom/nodes[0], source_rhoE_sng )
        source_spec = make_obj_array([
                      actx.np.where( nodes_are_off_axis,
                          source_spec_dom[i]/nodes[0], source_spec_sng[i] )
                      for i in range(nspecies)])
        
        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                       momentum=make_obj_array([source_rhoU, source_rhoV]),
                       species_mass=source_spec)

        
#    def gravity_source_terms(cv):
#        """Gravity."""
#        return make_conserved(dim=2,
#                              mass=cv.mass*0.0,
#                              energy=cv.momentum[1]*-9.80665,
#                              momentum=make_obj_array([cv.mass* 0.0,
#                                                       cv.mass*-9.80665]),
#                              species_mass=cv.species_mass*0.0)

##############################################################################

    from mirgecom.limiter import bound_preserving_limiter
    def limiter(cv, pressure, temperature):

        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                                     mmin=0.0, mmax=1.0, modify_average=True)
            for i in range(nspecies)
        ])

        aux = cv.mass*0.0
        for i in range(0,nspecies):
          aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        mass_lim = eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        cv_limited = make_conserved(dim=dim, 
                                    mass=mass_lim,
                                    energy=energy_lim,
                                    momentum=mass_lim*cv.velocity,
                                    species_mass=mass_lim*spec_lim)

        return cv_limited

    apply_limiter = actx.compile(limiter)

    def _update_dv(cv, temperature, smoothness):
        return eos.update_dependent_vars(cv, temperature, smoothness)

    update_dv = actx.compile(_update_dv)

    def _update_tv(cv, dv):
        return gas_model.transport.transport_vars(cv, dv, eos)

    update_tv = actx.compile(_update_tv)

    def _update_fluid_state(cv, dv, tv):
        return ViscousFluidState(cv, dv, tv)

    update_fluid_state = actx.compile(_update_fluid_state)

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        cv, tseed = state
        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)

        if True: #use_AV:
            smoothness = force_evaluation(actx, smoothness_indicator(
                dcoll, cv.mass, kappa=kappa_av, s0=s0))

        # update temperature value
        fluid_state = get_fluid_state(cv, tseed, smoothness=smoothness)

        # apply limiter and reevaluate CV
        limited_cv = apply_limiter(
            fluid_state.cv, fluid_state.pressure, fluid_state.temperature)

        # get new fluid_state with limited species and respective energy
        new_dv = update_dv(limited_cv, fluid_state.temperature, smoothness)
        new_tv = update_tv(limited_cv, new_dv)
        fluid_state = update_fluid_state(limited_cv, new_dv, new_tv)

        if local_dt:
            t = force_evaluation(actx, t)
            dt = (#force_evaluation(actx, 
                get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                 gas_model,
                                 constant_cfl=constant_cfl,
                                 local_dt=local_dt)
            )
            dt = force_evaluation(actx, actx.np.maximum(current_dt, dt))
        else:
            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                      t_final, constant_cfl, local_dt)

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

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

            if do_restart:
                my_write_restart(step=step, t=t, cv=fluid_state.cv,
                                 tseed=fluid_state.temperature)

            if do_viz:
                grad_cv = None
                grad_t = None
                sources = None
                if plot_gradients:
                    cv_rhs, grad_cv, grad_t = \
                        ns_operator(dcoll, state=fluid_state, time=t,
                                    boundaries=boundaries,
                                    gas_model=gas_model,
                                    return_gradients=True,
                                    quadrature_tag=quadrature_tag)

                    sources = (
                        axisymmetry_source_terms(actx, dcoll, fluid_state, 
                                                 grad_cv, grad_t)
                    )
                
                my_write_viz(step=step, t=t, state=fluid_state, dt=dt,
                             ns_rhs=None, grad_cv=grad_cv, grad_t=grad_t,
                             sources=sources, plot_gradients=plot_gradients)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state)
            raise

        #return make_obj_array([cv, tseed]), dt
        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt

    def my_rhs(t, state):
        cv, tseed = state

        if True: #use_AV:
            smoothness = smoothness_indicator(dcoll, cv.mass,
                                              kappa=kappa_av, s0=s0)

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed, 
                                       smoothness=smoothness)

        ns_rhs, grad_cv, grad_t = (
            ns_operator(dcoll, state=fluid_state, time=t,
                        boundaries=boundaries, gas_model=gas_model,
                        return_gradients=True, quadrature_tag=quadrature_tag)
        )

        chem_rhs = (#speedup_factor * 
            0.25*eos.get_species_source_terms(cv, fluid_state.temperature)
        )

        sources = (
#            speedup_factor*gravity_source_terms(cv) +
            axisymmetry_source_terms(actx, dcoll, fluid_state, grad_cv, grad_t)
        )

        sponge = sponge_func(cv=cv, cv_ref=ref_cv, sigma=sponge_sigma)
 
        cv_rhs = ns_rhs + chem_rhs + sponge + sources
        #cv_rhs = ns_rhs + sponge + sources
        
        return make_obj_array([cv_rhs, fluid_state.temperature])

    def my_post_step(step, t, dt, state):
        min_dt = np.min(actx.to_numpy(dt))
        if logmgr:
            set_dt(logmgr, min_dt)         
            logmgr.tick_after()

        return state, dt

##############################################################################

    if local_dt == True:
        dt = (#force_evaluation(actx, 
             get_sim_timestep(dcoll, current_state, current_t,
                     current_dt, current_cfl, gas_model,
                     constant_cfl=constant_cfl, local_dt=local_dt)
        )
        dt = force_evaluation(actx, actx.np.maximum(current_dt, dt))

        t = force_evaluation(actx, current_t + dt*0.0)
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=make_obj_array([current_state.cv, tseed]),
                      dt=dt, t_final=t_final, t=t,
                      max_steps=niter, local_dt=local_dt,
                      istep=current_step)
    current_cv, tseed = stepper_state

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    if True: #use_AV:
        smoothness = smoothness_indicator(dcoll, current_cv.mass,
                                          kappa=kappa_av, s0=s0)

    current_state = make_fluid_state(current_cv, gas_model, tseed, 
                                     smoothness=smoothness)

    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     tseed=tseed)

    my_write_viz(step=current_step, t=current_t, #dt=current_dt,
                 cv=current_state.cv, dv=current_state.dv)

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
         lazy=args.lazy, rst_filename=restart_file)
