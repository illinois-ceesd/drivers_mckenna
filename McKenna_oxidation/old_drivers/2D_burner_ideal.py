"""Fri Sep 30 10:39:39 CDT 2022"""

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

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    global_reduce
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    IsothermalWallBoundary,
    OutflowBoundary,
    SymmetryBoundary,
    PrescribedFluidBoundary,
    AdiabaticNoslipWallBoundary,
    LinearizedBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import PowerLawTransport
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)

from pytools.obj_array import make_obj_array

from mirgecom.fluid import velocity_gradient, species_mass_fraction_gradient

from grudge.dof_desc import BoundaryDomainTag, VolumeDomainTag
from grudge.dof_desc import DOFDesc, as_dofdesc, DISCR_TAG_BASE
from grudge.dof_desc import DD_VOLUME_ALL

#########################################################################

class Burner2D_Reactive:

    def __init__( self, *, dim=2, sigma=0.001, pressure, temperature, speedup_factor):

        self._dim = dim
        self._sigma = sigma
        self._pres = pressure
        self._speedup_factor = speedup_factor
        self._temp = temperature

    def __call__(self, x_vec, eos, flow_rate, state_minus=None):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        actx = x_vec[0].array_context

        cool_temp = 300.0

        sigma_factor = 10.0 - 9.0*(0.12 - x_vec[1])**2/(0.12 - 0.10)**2
        _sigma = self._sigma*(
            actx.np.where(actx.np.less(x_vec[1], 0.12),
                              actx.np.where(actx.np.greater(x_vec[1], .10),
                                            sigma_factor, 1.0),
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
        flame = 1.0
        upper_atm = 0.5*(1.0 + actx.np.tanh( 1.0/(15.0*self._sigma)*(x_vec[1] - 0.120)))
            
        #~~~ atmosphere
        atmosphere = 1.0 - (shroud + core)

        #~~~ velocity
        v_inlet = flow_rate/20.0*(0.117892*self._speedup_factor)
        v_shroud = 20.0/20.0*(0.117892*self._speedup_factor)*2.769230769230767
        u_x = 0.0*x_vec[0]
        u_y = ((self._temp/300.0)*v_inlet*core + shroud*v_shroud)*(1.0-upper_atm)
        velocity = make_obj_array([u_x, u_y])

        #~~~ 

        temp = (self._temp)*(1.0-upper_atm) + 300.0*upper_atm
        temperature = temp*core + 300.0*(1.0 - core)

        if state_minus is None:
            pressure = self._pres + 0.0*x_vec[0]
        else:
            pressure = state_minus.dv.pressure
        mass = eos.get_density(pressure, temperature)

#        # initial condition and reference state for sponge
#        if state_minus is None:
#            temp = (self._temp)*(1.0-upper_atm) + 300.0*upper_atm
#            temperature = temp*core + 300.0*(1.0 - core)

#            pressure = self._pres + 0.0*x_vec[0]
#            gas_const = eos.gas_const()
#            mass = pressure/(gas_const*temperature)

#        # inlet boundary condition
#        else:
#            mass = 1.177877967786594*(1.0 - core) + 0.1688*core
#            pressure = state_minus.dv.pressure
#            gas_const = eos.gas_const()
#            temperature = pressure/(gas_const*mass)

        gamma = 1.4
        internal_energy = pressure/(gamma-1.0)
        kinetic_energy = 0.5 * mass * np.dot(velocity, velocity)
        energy = internal_energy + kinetic_energy

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                momentum=mass*velocity)


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


class OutflowDamping:
    def __init__(self, amplitude, y_max, y_thickness):
        self._y_max = y_max
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec):
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = 0*xpos

        if (self._y_max is not None):
          y0 = (self._y_max - self._y_thickness)
          dy = +((ypos - y0)/self._y_thickness)
          sponge_y = self._amplitude * actx.np.where(
              actx.np.greater(ypos, y0),
                  actx.np.where(actx.np.greater(ypos, self._y_max),
                                1.0, 3.0*dy**2 - 2.0*dy**3),
                  0.0
          )

        return sponge_y


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

    mesh_filename = "mesh_29m_noFlame_40mm_025um-v2.msh"

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 1000
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 1.0e-1
    t_final = 2.0

    niter = 3000001
    
    # discretization and model control
    order = 2
    speedup_factor = 10.0
    x0_sponge = 0.12
    sponge_amp = 200.0
    flow_rate = 40.0
    theta_factor = 0.01

    local_dt = True
    constant_cfl = True
    current_cfl = 0.4

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

    eos = IdealSingleGas()
    transport_model = PowerLawTransport(beta=4.093e-7*speedup_factor)
    gas_model = GasModel(eos=eos, transport=transport_model)

#############################################################################

    def smoothness_region(dcoll, field):        
        actx = field.array_context
        nodes = force_evaluation(actx, dcoll.nodes())
        xpos = nodes[0]
        ypos = nodes[1]

        y_max = 0.45
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

#    from mirgecom.limiter import bound_preserving_limiter
#    from mirgecom.drop_order import chop_highest_polynomial
#    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

#        # modify pressure
#        #pressure = chop_highest_polynomial(dcoll, pressure, dd=dd)

#        # recompute density
#        mass_lim = eos.get_density(pressure=pressure,
#            temperature=temperature)

#        # recompute energy
#        energy_lim = pressure/(eos.gamma() - 1.0) + 0.5*mass_lim*np.dot(cv.velocity, cv.velocity)

#        cv = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
#            momentum=mass_lim*cv.velocity)

#        return cv

    #limit_fluid_cv = actx.compile(_limit_fluid_cv)

    from mirgecom.drop_order import drop_order
    def _drop_order_cv(cv, flipped_smoothness, theta_factor, dd=None):

        smoothness = 1.0 - theta_factor*flipped_smoothness

        density_lim = drop_order(dcoll, cv.mass, smoothness)
        momentum_lim = make_obj_array([
            drop_order(dcoll, cv.momentum[0], smoothness),
            drop_order(dcoll, cv.momentum[1], smoothness)])
        energy_lim = drop_order(dcoll, cv.energy, smoothness)

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=density_lim, energy=energy_lim,
            momentum=momentum_lim)

    drop_order_cv = actx.compile(_drop_order_cv)

##############################################################################

    restart_step = None
    if restart_file is None:        
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            tag_to_elements = None
            volume_to_tags = None
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

        local_mesh = volume_to_local_mesh_data
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
    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    nodes = actx.thaw(dcoll.nodes())

    dd_vol = DD_VOLUME_ALL

#####################################################################################

    def _get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model,
#            limiter_func=_limit_fluid_cv
        )

    get_fluid_state = actx.compile(_get_fluid_state)

#####################################################################################

    from grudge.op import nodal_min_loc, nodal_max_loc, nodal_min, nodal_max
    def vol_min(x):
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
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

        flow_init = Burner2D_Reactive(dim=dim, sigma=0.00020, temperature=2100,
            pressure=101325.0, speedup_factor=speedup_factor)
        current_cv = flow_init(nodes, eos, flow_rate=flow_rate)

    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t))

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
            #tseed = connection(restart_data["temperature_seed"])
        else:
            current_cv = restart_data["state"]
            #tseed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)


    current_cv = force_evaluation(actx, current_cv)
    current_state = get_fluid_state(current_cv)

#####################################################################################

    from mirgecom.boundary import PrescribedFluidBoundary
    from mirgecom.inviscid import inviscid_flux, inviscid_facial_flux_rusanov
    from mirgecom.flux import num_flux_central
    from mirgecom.viscous import viscous_facial_flux_central
    from grudge.trace_pair import TracePair

    """
    """
    class MyPrescribedBoundary(PrescribedFluidBoundary):
        r"""My prescribed boundary function. """

        def __init__(self, bnd_state_func, temperature_func):
            """Initialize the boundary condition object."""
            self.bnd_state_func = bnd_state_func
            PrescribedFluidBoundary.__init__(self,
            boundary_state_func=self.prescribed_state_for_advection,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_temperature_func=temperature_func,
            boundary_gradient_cv_func=self.grad_cv_bc,
            )

        def prescribed_state_for_advection(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            state_plus = self.bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs)

#            pressure = state_minus.dv.pressure
#            temperature = state_plus.dv.temperature
#            mass = state_plus.cv.mass
#            momentum = state_plus.cv.momentum

            return state_plus

        def prescribed_state_for_diffusion(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs): #XXX pass grad_cv_minus
            return self.bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs) #XXX pass grad_cv_minus

        def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                numerical_flux_func, **kwargs):

            state_plus = self.prescribed_state_for_advection(dcoll=dcoll, dd_bdry=dd_bdry,
                gas_model=gas_model, state_minus=state_minus, **kwargs)

            state_pair = TracePair(dd_bdry, interior=state_minus, exterior=state_plus)

            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            actx = state_pair.int.array_context
            lam = actx.np.maximum(state_pair.int.wavespeed, state_pair.ext.wavespeed)
            from mirgecom.flux import num_flux_lfr
            return num_flux_lfr(f_minus_normal=inviscid_flux(state_pair.int)@normal,
                f_plus_normal=inviscid_flux(state_pair.ext)@normal,
                q_minus=state_pair.int.cv,
                q_plus=state_pair.ext.cv, lam=lam)

#        def grad_cv_bc(self, state_plus, state_minus, grad_cv_minus, normal, **kwargs):
#            return grad_cv_minus
        def grad_cv_bc(self, state_plus, state_minus, grad_cv_minus, normal, **kwargs):
            """Return grad(CV) to be used in the boundary calculation of viscous flux."""

            # extrapolate density and its gradient
            mass_plus = state_minus.mass_density
            grad_mass_plus = grad_cv_minus.mass

            v_minus = state_minus.velocity               
            grad_v_minus = velocity_gradient(state_minus.cv, grad_cv_minus)

            # du/dx and u are zero at the surface, so is d(rho u)/dx
            # this implies that d(rho v)/dy = 0
            v_plus = state_plus.velocity
            grad_mom_plus = 1.0*grad_cv_minus.momentum - grad_cv_minus.momentum*np.eye(dim)
            grad_v_plus = 1.0/mass_plus*( grad_mom_plus - v_plus*grad_mass_plus )

            # the energy has to be modified accordingly:
            # first, get gradient of internal energy, i.e., no kinetic energy
            grad_kin_energy_minus = 0.5*(
                np.dot(v_minus, grad_cv_minus.momentum) + np.dot(state_minus.cv.momentum, grad_v_minus)
            )
            grad_int_energy_minus = grad_cv_minus.energy - grad_kin_energy_minus

            # extrapolate gradient of internal energy
            grad_int_energy_plus = 1.0*grad_int_energy_minus

            # then modify gradient of kinetic energy to match the changes in velocity
            grad_kin_energy_plus = 0.5*(
                np.dot(v_plus, grad_mom_plus) + mass_plus*np.dot(v_plus, grad_v_plus)
            )

            grad_energy_plus = grad_int_energy_plus + grad_kin_energy_plus

            return make_conserved(grad_cv_minus.dim, mass=grad_cv_minus.mass,
                energy=grad_energy_plus, momentum=grad_mom_plus,
                species_mass=grad_cv_minus.species_mass #FIXME
            )

        def viscous_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
            grad_cv_minus, grad_t_minus, numerical_flux_func, **kwargs):
            """Return the boundary flux for the divergence of the viscous flux."""
            from mirgecom.viscous import viscous_flux
            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            state_plus = self.prescribed_state_for_diffusion(dcoll=dcoll, dd_bdry=dd_bdry,
                gas_model=gas_model, state_minus=state_minus, **kwargs)

            grad_cv_plus = self.grad_cv_bc(state_plus=state_plus,
                state_minus=state_minus, grad_cv_minus=grad_cv_minus, normal=normal, **kwargs)

            grad_t_plus = self._bnd_grad_temperature_func(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                grad_t_minus=grad_t_minus)

            # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
            # *not* the numerical viscous flux as advised by [Bassi_1997]_.
            f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                                 grad_t=grad_t_plus)
            return f_ext@normal  


#    inflow_nodes = force_evaluation(actx, dcoll.nodes(dd_vol.trace('inlet')))
#    inflow_cv_cond = force_evaluation(actx, ref_state(x_vec=inflow_nodes, eos=eos, flow_rate=flow_rate))
#    inflow_temperature = get_fluid_state(cv=inflow_cv_cond).temperature
#    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
#        return inflow_temperature

    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
        inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
        inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos,
            flow_rate=flow_rate, state_minus=state_minus,
        )
        inflow_state = make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
#            limiter_func=_limit_fluid_cv, limiter_dd=dd_bdry
        )
        return inflow_state.temperature

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
        inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
        inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos,
            flow_rate=flow_rate, state_minus=state_minus,
        )
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
#            limiter_func=_limit_fluid_cv, limiter_dd=dd_bdry
        )

    rho_ref = 101325.0/(eos.gas_const()*300.0)
    inflow_bnd = MyPrescribedBoundary(bnd_state_func=inlet_bnd_state_func,
        temperature_func=bnd_temperature_func)
    linear_bnd = LinearizedBoundary(dim=2, free_stream_density=rho_ref,
        free_stream_pressure=101325.0, free_stream_velocity=np.zeros(shape=(dim,)))
    
    boundaries = {BoundaryDomainTag("inlet"): inflow_bnd,
                  BoundaryDomainTag("linear"): linear_bnd,
                  BoundaryDomainTag("symmetry"): SymmetryBoundary(),
                  BoundaryDomainTag("outlet"): OutflowBoundary(boundary_pressure=101325.0),
                  BoundaryDomainTag("burner"): AdiabaticNoslipWallBoundary(),
                  BoundaryDomainTag("solid"): IsothermalWallBoundary(wall_temperature=300.0)}

#####################################################################################

    # initialize the sponge field
    sponge_x_thickness = 0.05
    sponge_y_thickness = 0.09
    xMaxLoc = x0_sponge + sponge_x_thickness
    yMinLoc = 0.0

    sponge_init = InitSponge(amplitude=sponge_amp,
        x_max=xMaxLoc, y_min=yMinLoc,
        x_thickness=sponge_x_thickness, y_thickness=sponge_y_thickness)
    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=nodes))

    ref_state = Burner2D_Reactive(dim=dim, sigma=0.00020, temperature=2100,
         pressure=101325.0, speedup_factor=speedup_factor)
    ref_cv = force_evaluation(actx, ref_state(nodes, eos, flow_rate))

####################################################################################

    visualizer = make_visualizer(dcoll)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
        global_nelements=global_nelements, dt=current_dt, t_final=t_final,
        nstatus=nstatus, nviz=nviz, t_initial=current_t, cfl=current_cfl,
        constant_cfl=constant_cfl, initname=initname, eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

#########################################################################

    from grudge.dt_utils import characteristic_lengthscales, dt_geometric_factors
    def my_write_viz(step, t, state, dt=None, smoothness=None,
         ns_rhs=None, chem_sources=None, grad_cv=None, grad_t=None,
         ref_cv=None, sources=None, plot_gradients=False):

        viz_fields = [("CV_rho", state.cv.mass),
                      ("CV_rhoU", state.cv.momentum),
                      ("CV_rhoE", state.cv.energy),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      ("DV_U", state.velocity[0]),
                      ("DV_V", state.velocity[1]),
                      ("dt", dt),
                      #("sponge", sponge_sigma),
                      ("smoothness", 1.0 - theta_factor*smoothness),
                      ]

        if plot_gradients:
            dkappadr = my_derivative_function(actx, dcoll,     kappa, 'replicate')[0]
            off_axis_x = 1e-7
            nodes_are_off_axis = actx.np.greater(nodes[0], off_axis_x)  # noqa
            d2Tdr2   = my_derivative_function(actx, dcoll, grad_t[0],  'symmetry')[0]
            qr = - kappa*grad_t[0]
            dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)
            source_qr = actx.np.where( nodes_are_off_axis, qr/nodes[0], dqrdr )

            viz_fields.extend([
                ("grad_T", grad_t),
                ("grad_rho", grad_cv.mass),
                ("kappa", kappa),
                ("axi_qr", qr),
                ("axi_dqrdr", dqrdr),
                ("axi_d2Tdr2", d2Tdr2),
                ("axi_dkappadr", dkappadr),
                ("axi_source_qr", source_qr)
                ])
                      
        print('Writing solution file...')
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                #"temperature_seed": tseed,
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
          
    from arraycontext import outer
    from grudge.trace_pair import TracePair, interior_trace_pairs, tracepair_with_discr_tag
    from grudge import op
    from meshmode.discretization.connection import FACE_RESTR_ALL
    from mirgecom.flux import num_flux_central

    class _MyGradTag:
        pass

    def my_derivative_function(actx, dcoll, field, field_bounds, bnd_cond, dd_vol=DD_VOLUME_ALL):    

        dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

        interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

        def interior_flux(field_tpair):
            dd_trace_quad = field_tpair.dd.with_discr_tag(quadrature_tag)
            normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
            bnd_tpair_quad = interp_to_surf_quad(field_tpair)
            flux_int = outer(num_flux_central(bnd_tpair_quad.int, bnd_tpair_quad.ext), normal_quad)

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad, flux_int)

        def boundary_flux(bdtag, bdry):
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = actx.thaw(dcoll.normal(dd_bdry_quad)) 
            int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field)

            if bnd_cond == 'symmetry' and bdtag  == 'symmetry':
                ext_soln_quad = 0.0*int_soln_quad
            else:
                ext_soln_quad = 1.0*int_soln_quad

            bnd_tpair = TracePair(bdtag, interior=int_soln_quad, exterior=ext_soln_quad)
            flux_bnd = outer(num_flux_central(bnd_tpair.int, bnd_tpair.ext), normal_quad)
        
            return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad, flux_bnd)

        return -op.inverse_mass(
            dcoll, dd_vol,
            op.weak_local_grad(dcoll, dd_vol, field)
            - op.face_mass(dcoll, dd_allfaces_quad,
                (sum(interior_flux(u_tpair) for u_tpair in interior_trace_pairs(dcoll, field, volume_dd=dd_vol, comm_tag=_MyGradTag))
                + sum(boundary_flux(bdtag, bdry) for bdtag, bdry in field_bounds.items())
                )
            )
        )

    #########################################################################

    off_axis_x = 1e-7
    nodes_are_off_axis = actx.np.greater(nodes[0], off_axis_x)  # noqa
   
    def axisymmetry_source_terms(actx, dcoll, state, grad_cv, grad_t):

        nodes = actx.thaw(dcoll.nodes())
        
        cv = state.cv
        dv = state.dv
        
        mu = state.tv.viscosity
        beta = gas_model.transport.volume_viscosity(cv, dv, eos)
        kappa = state.tv.thermal_conductivity
       #d_ij = state.tv.species_diffusivity
        
        grad_v = velocity_gradient(cv,grad_cv)
       #grad_y = species_mass_fraction_gradient(cv, grad_cv)

        u = state.velocity[0]
        v = state.velocity[1]

        dudr = grad_v[0][0]
        dudy = grad_v[0][1]
        dvdr = grad_v[1][0]
        dvdy = grad_v[1][1]
        
        drhoudr = (grad_cv.momentum[0])[0]

       #d2udr2  = my_derivative_function(actx, dcoll,   dudr, boundaries, 'replicate')[0] #XXX
        d2vdr2  = my_derivative_function(actx, dcoll,   dvdr, boundaries, 'replicate')[0] #XXX
        d2udrdy = my_derivative_function(actx, dcoll,   dudy, boundaries, 'replicate')[0] #XXX
                
        dmudr    = my_derivative_function(actx, dcoll,    mu, boundaries, 'replicate')[0]
        dbetadr  = my_derivative_function(actx, dcoll,  beta, boundaries, 'replicate')[0]
        dbetady  = my_derivative_function(actx, dcoll,  beta, boundaries, 'replicate')[1]
       #dkappadr = my_derivative_function(actx, dcoll, kappa, boundaries, 'replicate')[0]
        
        qr = - kappa*grad_t[0]
       #d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0],  'symmetry')[0]
        dqrdr = 0.0 #- (dkappadr*grad_t[0] + kappa*d2Tdr2)
        
        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              nodes_are_off_axis, u/nodes[0], dudr )

        #dtaurydr = my_derivative_function(actx, dcoll, tau_ry)[0]
        dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

        """
        """
        source_mass_dom = - cv.momentum[0]

        source_rhoU_dom = - cv.momentum[0]*u \
            + tau_rr - tau_tt \
            + u*dbetadr + beta*dudr \
            + beta*actx.np.where(nodes_are_off_axis, -u/nodes[0], -dudr )
                              
        source_rhoV_dom = - cv.momentum[0]*v \
            + tau_ry \
            + u*dbetady + beta*dudy
        
        source_rhoE_dom = -( (cv.energy+dv.pressure)*u + qr ) \
            + u*tau_rr + v*tau_ry \
            + u**2*dbetadr + beta*2.0*u*dudr \
            + u*v*dbetady + u*beta*dvdy + v*beta*dudy

        """
        """

        source_mass_sng = - drhoudr
        source_rhoU_sng = 0.0 #mu*d2udr2 + 0.5*beta*d2udr2  #XXX
        source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady
        source_rhoE_sng = -( (cv.energy+dv.pressure)*dudr + dqrdr ) \
            + tau_rr*dudr + v*dtaurydr + 2.0*beta*dudr**2 \
            + beta*dudr*dvdy + v*dudr*dbetady + v*beta*d2udrdy

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
        
        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
            momentum=make_obj_array([source_rhoU, source_rhoV]))

        
    def gravity_source_terms(cv):
        """Gravity."""
        delta_rho = cv.mass - rho_ref
        return make_conserved(dim=2,
            mass=cv.mass*0.0, energy=delta_rho*cv.velocity[1]*-9.80665,
            momentum=make_obj_array([cv.mass*0.0, delta_rho*-9.80665])
        )

##############################################################################

    zeros = nodes[0]*0.0
    ones = nodes[0]*0.0 + 1.0

    smooth_region = force_evaluation(actx, smoothness_region(dcoll, nodes[0]))

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        state = force_evaluation(actx, state)

        cv = drop_order_cv(state, smooth_region, theta_factor)

        # update temperature value
        fluid_state = get_fluid_state(cv)

        if local_dt:
            t = force_evaluation(actx, t)
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                  gas_model, constant_cfl=constant_cfl, local_dt=local_dt)
            dt = force_evaluation(actx, actx.np.minimum(dt, current_dt))
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
                ns_rhs = None
                if plot_gradients:
                    cv_rhs, grad_cv, grad_t = ns_operator(
                        dcoll, state=fluid_state,time=t,
                        boundaries=boundaries, gas_model=gas_model,
                        return_gradients=True, quadrature_tag=quadrature_tag)
                
                my_write_viz(step=step, t=t, state=fluid_state, dt=dt,
                    smoothness=smooth_region, ns_rhs=ns_rhs, grad_cv=grad_cv,
                    grad_t=grad_t, sources=sources, plot_gradients=plot_gradients)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state, dt=dt,
                         smoothness=smooth_region)
            raise

        return fluid_state.cv, dt

    def my_rhs(t, state):

        cv = _drop_order_cv(state, smooth_region, theta_factor)

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
#            limiter_func=_limit_fluid_cv
        )

        ns_rhs, grad_cv, grad_t = ns_operator(dcoll, state=fluid_state, time=t,
            boundaries=boundaries, gas_model=gas_model,
            return_gradients=True, quadrature_tag=quadrature_tag)

        sources = (
            speedup_factor*gravity_source_terms(cv) +
            axisymmetry_source_terms(actx, dcoll, fluid_state, grad_cv, grad_t) +
            sponge_func(cv=cv, cv_ref=ref_cv, sigma=sponge_sigma)
        )
        
        return ns_rhs + sources

    def my_post_step(step, t, dt, state):
        min_dt = np.min(actx.to_numpy(dt)) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)         
            logmgr.tick_after()

        return state, dt

##############################################################################

    if local_dt == True:
        dt = force_evaluation(actx, actx.np.minimum(
            current_dt,
            get_sim_timestep(dcoll, current_state, current_t, current_dt, current_cfl,
                gas_model, constant_cfl=constant_cfl, local_dt=local_dt))
        )
        t = force_evaluation(actx, current_t + zeros)
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper, pre_step_callback=my_pre_step,
            post_step_callback=my_post_step, state=current_state.cv, dt=dt,
            t_final=t_final, t=t, max_steps=niter, local_dt=local_dt, istep=current_step)
    current_cv, tseed = stepper_state

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_state = make_fluid_state(current_cv, gas_model, tseed)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv, tseed=tseed)
    my_write_viz(step=curent_step, t=current_t, state=current_state)

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
