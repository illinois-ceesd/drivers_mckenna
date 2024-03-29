"""mirgecom driver for the porous-cylinder flow demonstration."""

__copyright__ = """
Copyright (C) 2024 University of Illinois Board of Trustees
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
import gc
import numpy as np
import pyopencl as cl
from functools import partial

from arraycontext import thaw

from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.shortcuts import compiled_lsrk45_step

from meshmode.dof_array import DOFArray

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local, check_range_local,
    global_reduce
)
from mirgecom.utils import force_evaluation
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    LinearizedInflowBoundary,
    LinearizedOutflowBoundary,
    PressureOutflowBoundary,
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import SimpleTransport
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import make_fluid_state
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_add_cl_device_info, logmgr_set_time,
    logmgr_add_device_memory_usage
)
from mirgecom.wall_model import (
    PorousWallTransport, PorousFlowModel, PorousWallVars
)

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass

def get_mesh(dim, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh_filename = "grid-v2.msh"
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)

    return mesh


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:

    def __init__(self, *, x_min=None, x_max=None, y_min=None, y_max=None,
                 x_thickness=None, y_thickness=None, amplitude):
        """ """
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_thickness = x_thickness
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec):
        """ """
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


from mirgecom.materials.carbon_fiber import FiberEOS as OriginalFiberEOS
class FiberEOS(OriginalFiberEOS):
    """Inherits and modified the original carbon fiber."""

    def thermal_conductivity(self, temperature, tau):
        return 0.0 + temperature*0.0

    def permeability(self, tau):
        virgin = 1.0e-6
        char = 1e+9
        return virgin*tau + char*(1.0 - tau)


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=False,
                           use_einsum_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    restart_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    snapshot_pattern = restart_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    Reynolds_number = 40.0
    Mach_number = 0.3

     # default i/o frequencies
    nviz = 1000
    nrestart = 10000
    nhealth = 1
    nstatus = 100
    ngarbage = 10

    # default timestepping control
    integrator = "compiled_lsrk45"
    t_final = 100.0

    constant_cfl = True
    current_cfl = 0.10
    current_dt = 0.0 #dummy if constant_cfl = True
    local_dt = True
    
    # discretization and model control
    order = 3
    use_overintegration = False

######################################################

    dim = 2
    current_t = 0
    current_step = 0

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
        if (constant_cfl == False):
          print(f"\tcurrent_dt = {current_dt}")
          print(f"\tt_final = {t_final}")
        else:
          print(f"\tconstant_cfl = {constant_cfl}")
          print(f"\tcurrent_cfl = {current_cfl}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")
        print(f"\tuse_overintegration = {use_overintegration}")

####################################################################

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

####################################################################

    if rank == 0:
        print("Making discretization")
        logging.info("Making discretization")

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
    dcoll = create_discretization_collection(actx, local_mesh, order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = nodes[0]*0.0

    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

    from grudge.dof_desc import DD_VOLUME_ALL as dd

####################################################################

    eos = IdealSingleGas()

    _temperature = 300.0
    _pressure = 100000.0
    _mass = _pressure/(eos.gas_const()*_temperature)
    _c = np.sqrt(eos.gamma()*_pressure/_mass)

    mu = _mass*(_c*Mach_number)*1.0/Reynolds_number
    kappa = 1000.0*mu/0.71
    base_transport = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    sample_transport = PorousWallTransport(base_transport=base_transport)

    # ~~~
    material_densities = 168.0/10.0 + zeros

    import mirgecom.materials.carbon_fiber as material_sample
    material = FiberEOS(dim=2, char_mass=0.0, virgin_mass=168.0/10.0,
                        anisotropic_direction=0)
    # decomposition = No_Oxidation_Model()

    # ~~~
    gas_model = PorousFlowModel(eos=eos, transport=sample_transport,
                                wall_eos=material)

#####################################################################

#    def uniform_flow(x_vec, eos):

#        gamma = eos.gamma()
#        R = eos.gas_const()
#        
#        x = x_vec[0]

#        pressure = 100000.0 + 0.0*x
#        mass = 1.0 + 0.0*x
#        c = actx.np.sqrt(gamma*pressure/mass)
#        
#        u_x = 0.0*x + c*Mach_number
#        u_y = 0.0*x

#        velocity = make_obj_array([u_x, u_y])   
#        ke = .5*np.dot(velocity, velocity)*mass

#        rho_e = pressure/(eos.gamma()-1) + ke

##        # ~~~
##        # FIXME prescribe temperature or use internal value?
##        internal_energy = gas_model.eos.get_internal_energy(temperature, species_mass_fractions=y)
##        kinetic_energy = 0.5 * np.dot(velocity, velocity)
##        solid_energy = mass*0.0
##        if boundary is False:
##            material_dens = self._plug * self._solid_mass + x_vec[0]*0.0
##            eps_rho_solid = gas_model.solid_density(material_dens)
##            tau = gas_model.decomposition_progress(material_dens)
##            solid_energy = eps_rho_solid * gas_model.wall_eos.enthalpy(temperature, tau)
##        energy = mass * (internal_energy + kinetic_energy) + solid_energy

#        return make_conserved(dim, mass=mass, energy=rho_e,
#                              momentum=mass*velocity)

#    flow_init = uniform_flow

    def plug_region(x_vec, thickness):

        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = 0*xpos

        cylinder_radius = 0.5

        radius = actx.np.sqrt(xpos**2 + ypos**2)
        radius = actx.np.where(actx.np.greater(radius, 0.5), 0.5, radius)
        sponge = actx.np.where(actx.np.less(radius, cylinder_radius-thickness), 0.0, 0.5*(radius-(cylinder_radius-thickness))/(thickness))
        circle = 1.0 - 2.0*(-20.0*sponge**7 + 70*sponge**6 - 84*sponge**5 + 35*sponge**4)

        return circle

    plug = force_evaluation(actx, plug_region(x_vec=nodes, thickness=0.03))
    
    u_x = _c*Mach_number
    u_y = 0.0
    from mirgecom.materials.initializer import PorousWallInitializer
    flow_init = PorousWallInitializer(
        temperature=300.0, material_densities=material_densities,
        velocity=make_obj_array([u_x, u_y]), pressure=100000.0,
        porous_region=plug)

######################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
    
        temperature_seed = 300.0 + nodes[0]*0.0
        current_cv, sample_densities = flow_init(x_vec=nodes, gas_model=gas_model)
        first_step = 0
    else:
        current_t = restart_data["t"]
        if np.isscalar(current_t) is False:
            current_t = np.min(actx.to_numpy(current_t[0]))
        current_step = restart_step
        first_step = current_step + 0

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
            temperature_seed = connection(restart_data["temperature_seed"])
            sample_densities = connection(restart_data["sample_densities"])
        else:
            current_cv = restart_data["state"]
            temperature_seed = restart_data["temperature_seed"]
            sample_densities = restart_data["sample_densities"]

    ##################################################

    def _get_fluid_state(cv, sample_densities, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temp_seed,
                                material_densities=sample_densities)

    get_fluid_state = actx.compile(_get_fluid_state)

    current_cv = force_evaluation(actx, current_cv)
    temperature_seed = force_evaluation(actx, temperature_seed)
    sample_densities = force_evaluation(actx, sample_densities)
    current_state = get_fluid_state(cv=current_cv, temp_seed=temperature_seed,
                                    sample_densities=sample_densities)

    ##################################################

#    sponge_init = PorousWallInitializer(
#        temperature=300.0, material_densities=material_densities,
#        velocity=make_obj_array([u_x, u_y]), pressure=100000.0,
#        porous_region=0.0)

#    initial_cv, _ = sponge_init(x_vec=nodes, gas_model=gas_model)
#    ref_state = make_fluid_state(cv=initial_cv, gas_model=gas_model,
#                                 temperature_seed=temperature_seed,
#                                 material_densities=sample_densities)

#    # initialize the sponge field
#    sponge_x_thickness = 10.0
#    sponge_y_thickness = 10.0

#    xMaxLoc = +50.0
#    xMinLoc = -25.0
#    yMaxLoc = +25.0
#    yMinLoc = -25.0

#    sponge_amp = 100.0 #may need to be modified. Let's see...

#    sponge_init = InitSponge(amplitude=sponge_amp,
#        x_min=xMinLoc, x_max=xMaxLoc,
#        y_min=yMinLoc, y_max=yMaxLoc,
#        x_thickness=sponge_x_thickness,
#        y_thickness=sponge_y_thickness
#    )

#    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=nodes))
    
    ############################################################################

#    inflow_init = PorousWallInitializer(
#        temperature=300.0, material_densities=0.0,
#        velocity=make_obj_array([u_x, u_y]), pressure=100000.0,
#        porous_region=0.0)

#    inflow_nodes = actx.thaw(dcoll.nodes(dd.trace('inflow')))
#    _inflow_cv, _ = inflow_init(x_vec=inflow_nodes, gas_model=gas_model)
#    inflow_cv = force_evaluation(actx, _inflow_cv)
#    inflow_samp_dens = force_evaluation(actx, inflow_nodes[0]*0.0)
#    inflow_state = get_fluid_state(cv=inflow_cv, temp_seed=300.0,
#                                   sample_densities=inflow_samp_dens)

#    def _inflow_boundary_state_func(**kwargs):
#        return inflow_state
   
    #inflow_boundary  = PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state_func)

    side_boundary = LinearizedOutflowBoundary(
        free_stream_density=_mass, free_stream_velocity=make_obj_array([u_x, u_y]),
        free_stream_pressure=_pressure)
    inflow_boundary = LinearizedInflowBoundary(
        free_stream_density=_mass, free_stream_velocity=make_obj_array([u_x, u_y]),
        free_stream_pressure=_pressure)
    outflow_boundary = PressureOutflowBoundary(boundary_pressure=_pressure)

    boundaries = {dd.trace("side").domain_tag: side_boundary,
                  dd.trace("inflow").domain_tag: inflow_boundary,
                  dd.trace("outflow").domain_tag: outflow_boundary}

    ##################################################

    vis_timer = None

    logmgr_add_device_memory_usage(logmgr, queue)
    try:
        logmgr.add_watches(["memory_usage_python.max"])
    except KeyError:
        pass

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step: {value}, "),
            ("dt.max", "dt: {value:1.6e} s, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s")])

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)

#########################################################################

    visualizer = make_visualizer(dcoll)

    initname = "cylinder"
    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

#########################################################################

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]
        
#########################################################################
            
    from mirgecom.fluid import velocity_gradient
    def my_write_viz(step, t, dt, fluid_state,
                     grad_cv=None, ref_cv=None, sponge_sigma=None):       

        viz_fields = [("CV", fluid_state.cv),
                      ("DV_U", fluid_state.cv.velocity[0]),
                      ("DV_V", fluid_state.cv.velocity[1]),
                      ("DV_P", fluid_state.dv.pressure),
                      ("DV_T", fluid_state.dv.temperature),
                      ("WV", fluid_state.wv),
                      ("dt", dt[0] if local_dt else None),
                      ("plug", plug)
                      ]

        """
        zVort = None

        if (grad_cv is not None):
            grad_v = velocity_gradient(state.cv,grad_cv)
            dudx = grad_v[0][0]
            dudy = grad_v[0][1]
            dvdx = grad_v[1][0]
            dvdy = grad_v[1][1]
            
            zVort = dvdx - dudy

            viz_fields.extend((
                ("Z_vort", zVort),
                ("ref_cv", ref_cv),
                ("sponge_sigma", sponge_sigma),
            ))
        """
                   
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        cv, tseed, sample_dens = state
        rst_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "temperature_seed": tseed,
                "sample_densities": sample_dens,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(cv, dv):
        health_error = False
#        pressure = force_evaluation(actx, dv.pressure)
#        temperature = force_evaluation(actx, dv.temperature)
        
        if check_naninf_local(dcoll, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_naninf_local(dcoll, "vol", dv.temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")
            
        return health_error 

#########################################################################

    def _my_get_timestep_fluid(fluid_state, t, dt):

        if not constant_cfl:
            return dt

        return get_sim_timestep(dcoll, fluid_state, t, dt,
            current_cfl, t_final=t_final, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd)

    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)

#########################################################################

    def darcy_source_terms(cv, tv, wv):
        """Source term to mimic Darcy's law."""
        return make_conserved(dim=2,
            mass=zeros,
            energy=zeros,
            momentum=(
                -1.0 * tv.viscosity * wv.void_fraction/wv.permeability *
                cv.velocity),
            species_mass=cv.species_mass*0.0)

    # ~~~~~~~

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed, sample_densities = state

        fluid_state = get_fluid_state(cv=cv, sample_densities=sample_densities,
                                      temp_seed=tseed)
        fluid_state = force_evaluation(actx, fluid_state)

        try:
         
            state = make_obj_array([fluid_state.cv, fluid_state.temperature,
                                    sample_densities])

            _dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                   t_final, constant_cfl, local_dt=local_dt)
            if local_dt:
                dt = make_obj_array([_dt, zeros, zeros])
            else:
                dt = _dt

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            collect_garbage = check_step(step=step, interval=ngarbage)

            if collect_garbage:
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    gc.collect()

            if do_health:
                dv = fluid_state.dv
                cv = fluid_state.cv
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)
                gc.freeze()

            if do_viz:                
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state)
                gc.freeze()

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")

            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state)
            raise

        return state, dt


    def my_rhs(t, state):
        cv, tseed, sample_densities = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed,
                                       material_densities=sample_densities)
        
        cv_rhs = ns_operator(dcoll, state=fluid_state, time=t,
                             boundaries=boundaries, gas_model=gas_model,
                             quadrature_tag=quadrature_tag)

#        sponge = sponge_func(cv=fluid_state.cv, cv_ref=ref_state.cv,
#                             sigma=sponge_sigma)

        darcy_flow = darcy_source_terms(fluid_state.cv, fluid_state.tv, fluid_state.wv)

        return make_obj_array([cv_rhs + darcy_flow, zeros, zeros])


    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer.start_sub_timer():
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections\n")
                gc.freeze()

        min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt
    
    ##########################################################################

    if local_dt:
        t = make_obj_array([current_t + zeros, zeros, zeros])
        dt = make_obj_array([current_dt, zeros, zeros])
    else:
        t = current_t
        dt = current_dt

    if rank == 0:
        logging.info("Stepping.")

    stepper_state = make_obj_array([current_state.cv, temperature_seed,
                                    sample_densities])

    (current_step, current_t, current_cv) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=stepper_state,
                      dt=dt, t_final=t_final, t=t,
                      istep=current_step, local_dt=local_dt, max_steps=500000,
                      force_eval=force_eval)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

#    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model)
#    ts_field, cfl, dt = my_get_timestep(current_t, current_dt, current_state)
#    my_write_viz(step=current_step, t=current_t, cv=current_state.cv,
#                   dv=current_state.dv, ref_cv=ref_state.cv)
#    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

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
    casename = "cylinder"
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
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile,
         lazy=args.lazy, casename=casename, rst_filename=restart_file)
