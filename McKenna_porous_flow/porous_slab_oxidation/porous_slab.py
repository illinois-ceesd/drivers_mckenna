"""mirgecom driver for the porous-slab flow demonstration."""

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
import cantera
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
from mirgecom.navierstokes import (
    grad_t_operator,
    grad_cv_operator,
    ns_operator
)
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
    AdiabaticNoslipWallBoundary,
    PrescribedFluidBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import PowerLawTransport #SimpleTransport
from mirgecom.eos import PyrometheusMixture
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    make_operator_fluid_states
)
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_add_cl_device_info, logmgr_set_time,
    logmgr_add_device_memory_usage
)
from mirgecom.wall_model import (
    PorousWallTransport, PorousFlowModel, PorousWallVars
)


class _FluidOpStatesTag:
    pass


class _FluidGradCVTag:
    pass


class _FluidGradTempTag:
    pass


class _FluidOperatorTag:
    pass


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
    mesh_filename = "grid_small-v2.msh"
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)

    return mesh


from mirgecom.materials.carbon_fiber import FiberEOS as OriginalFiberEOS
class FiberEOS(OriginalFiberEOS):
    """Inherits and modified the original carbon fiber."""

    def permeability(self, tau):
        virgin = 1.0e-5
        char = 1e+7
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

     # default i/o frequencies
    nviz = 2500
    nrestart = 10000
    nhealth = 1
    nstatus = 100
    ngarbage = 10

    # default timestepping control
    integrator = "compiled_lsrk45"
    t_final = 100.0

    constant_cfl = True
    current_cfl = 0.20
    current_dt = 0.0 #dummy if constant_cfl = True
    local_dt = True
    
    # discretization and model control
    order = 4
    use_overintegration = False

    mechanism_file = "air_3sp"

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

    u_x = 1.25e-3*10.0*10.0
    u_y = 0.0

    _temperature = 1000.0
    _pressure = 101325.0

    # {{{ Set up initial state using Cantera

    # Use Cantera for initialization
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    x_reference = np.zeros(nspecies,)
    x_reference[cantera_soln.species_index("O2")] = 0.21
    x_reference[cantera_soln.species_index("N2")] = 0.79

    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = _temperature, _pressure, x_reference
    y_reference = cantera_soln.Y

    # Import Pyrometheus EOS
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
                                cantera_soln, temperature_niter=3)(actx.np)

    species_names = pyrometheus_mechanism.species_names
    print(f"Pyrometheus mechanism species names {species_names}\n")

    temperature_seed = 1000.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    # }}}

#    mu = 1.5e-5*100.0
#    kappa = 1.5e-2
#    diff = 1.5e-4*100.0*np.ones(nspecies,)
#    base_transport = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
#                                     species_diffusivity=diff)
    base_transport=PowerLawTransport(beta=4.093e-7*10.0, lewis=np.ones(nspecies,))
    sample_transport = PorousWallTransport(base_transport=base_transport)

    # ~~~
    material_densities = 168.0/10.0 + zeros

    import mirgecom.materials.carbon_fiber as material_sample
    from mirgecom.materials.carbon_fiber import Y3_Oxidation_Model
    material = FiberEOS(dim=2, char_mass=0.0, virgin_mass=168.0/10.0,
                        anisotropic_direction=0)
    decomposition = Y3_Oxidation_Model(wall_material=material)

    # ~~~
    gas_model = PorousFlowModel(eos=eos, transport=sample_transport,
                                wall_eos=material)

#####################################################################

    def plug_region(x_vec, thickness):

        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context

        y0 = +0.01
        dy = actx.np.where(
        actx.np.less(ypos, y0-thickness*0.5),
            0.0, actx.np.where(actx.np.greater(ypos, y0+thickness*0.5),
                               1.0, (ypos-(y0-thickness*0.5))/thickness)
        )
        sponge_0 = 1.0-(-20.0*dy**7 + 70*dy**6 - 84*dy**5 + 35*dy**4)

        x0 = -0.03
        dx = actx.np.where(
        actx.np.less(xpos, x0-thickness*0.5),
            0.0, actx.np.where(actx.np.greater(xpos, x0+thickness*0.5),
                               1.0, (xpos-(x0-thickness*0.5))/thickness)
        )
        sponge_1 = (-20.0*dx**7 + 70*dx**6 - 84*dx**5 + 35*dx**4)

        x0 = +0.03
        dx = actx.np.where(
        actx.np.less(xpos, x0-thickness*0.5),
            0.0, actx.np.where(actx.np.greater(xpos, x0+thickness*0.5),
                               1.0, (xpos-(x0-thickness*0.5))/thickness)
        )
        sponge_2 = 1.0-(-20.0*dx**7 + 70*dx**6 - 84*dx**5 + 35*dx**4)

        return sponge_0*sponge_1*sponge_2

    plug = force_evaluation(actx, plug_region(x_vec=nodes, thickness=0.002))

    smoothing = 0.5*(1.0 + actx.np.tanh((nodes[0] + 0.045)/0.001))
    _species = make_obj_array([
        y_reference[0]*(1.0 - smoothing),
        zeros,
        (1.0 - y_reference[2])*smoothing + y_reference[2]
    ])

    init_temperature = 0.5*(1.0 - actx.np.tanh((nodes[0] + 0.045)/0.001))*700.0 + 300.0

    init_velocity = make_obj_array([u_x*0.5*(1.0 - actx.np.tanh((nodes[0] + 0.045)/0.001)),
                                    u_y])
    
    from mirgecom.materials.initializer import PorousWallInitializer
    flow_init = PorousWallInitializer(
        temperature=init_temperature, material_densities=material_densities,
        velocity=init_velocity, pressure=_pressure,
        porous_region=plug, species_mass_fractions=_species)

######################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
    
        temperature_seed = init_temperature
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

            current_cv = connection(restart_data["cv"])
            temperature_seed = connection(restart_data["temperature_seed"])
            sample_densities = connection(restart_data["sample_densities"])

#            # ------ filtering
#            filter_cutoff = restart_order
#            filter_order = 0
#            filter_alpha = -1.0*np.log(np.finfo(float).eps)

#            print("filter_cutoff = ", filter_cutoff)

#            if filter_cutoff < 0:
#                filter_cutoff = int(filter_frac * order)

#            if filter_cutoff >= order:
#                raise ValueError("Invalid setting for filter (cutoff >= order).")

#            from mirgecom.filter import (
#                exponential_mode_response_function as xmrfunc,
#                filter_modally
#            )
#            frfunc = partial(xmrfunc, alpha=filter_alpha,
#                             filter_order=filter_order)

#            def filter_field(field):
#                return filter_modally(dcoll, filter_cutoff, frfunc, field, dd=dd)

#            def filter_field(field):
#                # Compute cell averages of the state
#                def cancel_polynomials(grp):
#                    for mode_id in grp.mode_ids():
#                        print(mode_id)
#                    print(actx.from_numpy(np.asarray([0 if sum(mode_id) > restart_order
#                                                       else 1 for mode_id in grp.mode_ids()])))
#                    return actx.from_numpy(np.asarray([0 if sum(mode_id) > restart_order
#                                                       else 1 for mode_id in grp.mode_ids()]))

#                from grudge.dof_desc import DISCR_TAG_MODAL
#                from meshmode.transform_metadata import FirstAxisIsElementsTag

#                # map from nodal to modal
#                dd_nodal = dd
#                dd_modal = dd_nodal.with_discr_tag(DISCR_TAG_MODAL)

#                modal_map = dcoll.connection_from_dds(dd_nodal, dd_modal)
#                nodal_map = dcoll.connection_from_dds(dd_modal, dd_nodal)

#                modal_discr = dcoll.discr_from_dd(dd_modal)
#                modal_field = modal_map(field)

#                # cancel the ``high-order'' polynomials p > 0, and only the average remains
#                filtered_modal_field = DOFArray(
#                    actx,
#                    tuple(actx.einsum("ej,j->ej",
#                                      vec_i,
#                                      cancel_polynomials(grp),
#                                      arg_names=("vec", "filter"),
#                                      tagged=(FirstAxisIsElementsTag(),))
#                          for grp, vec_i in zip(modal_discr.groups, modal_field))
#                )

#                # convert back to nodal to have the average at all points
#                return nodal_map(filtered_modal_field)

#            current_cv = filter_field(current_cv)
#            temperature_seed = filter_field(temperature_seed)
#            sample_densities = filter_field(sample_densities)

        else:
            current_cv = restart_data["cv"]
            temperature_seed = restart_data["temperature_seed"]
            sample_densities = restart_data["sample_densities"]


    ##################################################

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_fluid_cv(cv, wv, pressure, temperature, dd=None):

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
        mass_lim = wv.void_fraction*eos.get_density(
            pressure=pressure, temperature=temperature,
            species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        ) + wv.density*gas_model.wall_eos.enthalpy(temperature, wv.tau)
        

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                              momentum=mass_lim*cv.velocity,
                              species_mass=mass_lim*spec_lim)

    # ~~~

    def _get_fluid_state(cv, sample_densities, temp_seed):
        return make_fluid_state(
            cv=cv, gas_model=gas_model, temperature_seed=temp_seed,
            limiter_func=_limit_fluid_cv, material_densities=sample_densities)

    get_fluid_state = actx.compile(_get_fluid_state)

    current_cv = force_evaluation(actx, current_cv)
    temperature_seed = force_evaluation(actx, temperature_seed)
    sample_densities = force_evaluation(actx, sample_densities)
    current_state = get_fluid_state(cv=current_cv, temp_seed=temperature_seed,
                                    sample_densities=sample_densities)
    
    ############################################################################

    inflow_init = PorousWallInitializer(
        temperature=_temperature, material_densities=0.0,
        velocity=make_obj_array([u_x, u_y]), pressure=_pressure,
        porous_region=0.0, species_mass_fractions=y_reference)

    inflow_nodes = actx.thaw(dcoll.nodes(dd.trace('inflow')))
    _inflow_cv, _ = inflow_init(x_vec=inflow_nodes, gas_model=gas_model)
    inflow_cv = force_evaluation(actx, _inflow_cv)
    inflow_samp_dens = force_evaluation(actx, inflow_nodes[0]*0.0)
    inflow_state = force_evaluation(
        actx, make_fluid_state(cv=inflow_cv, temperature_seed=_temperature,
                               gas_model=gas_model,
                               material_densities=inflow_samp_dens,
                               limiter_func=_limit_fluid_cv,
                               limiter_dd=dd.trace('inflow')))

    def _inflow_boundary_state_func(**kwargs):
        return inflow_state
   
    inflow_boundary  = PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state_func)

    side_boundary = AdiabaticNoslipWallBoundary()

    _mass = eos.get_density(_pressure, _temperature, y_reference)
    inflow_boundary = LinearizedInflowBoundary(
        free_stream_density=_mass, free_stream_velocity=make_obj_array([u_x, u_y]),
        free_stream_pressure=_pressure, free_stream_species_mass_fractions=y_reference)
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

    visualizer = make_visualizer(dcoll, vis_order=order)

    initname = "slab"
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

        if rank == 0:
            logging.info("Writing solution file.")

        viz_fields = [("CV", fluid_state.cv),
                      ("DV_U", fluid_state.cv.velocity[0]),
                      ("DV_V", fluid_state.cv.velocity[1]),
                      ("DV_P", fluid_state.dv.pressure),
                      ("DV_T", fluid_state.dv.temperature),
                      ("WV", fluid_state.wv),
                      ("dt", dt[0] if local_dt else None),
                      # ("plug", plug)
                      ]

        # species mass fractions
        viz_fields.extend((
            "Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
            for i in range(nspecies))
                   
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):

        if rank == 0:
            logging.info("Writing restart file.")

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

    import os
    import time
    t_start = time.time()
    t_shutdown = 720*60

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
                dt = make_obj_array([_dt, zeros, _dt])
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

            t_elapsed = time.time() - t_start
            if t_shutdown - t_elapsed < 300.0:
                my_write_restart(step, t, state)
                sys.exit()

            file_exists = os.path.exists("write_restart")
            if file_exists:
              os.system("rm write_restart")
              do_restart = True

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

        fluid_state = make_fluid_state(
            cv=cv, gas_model=gas_model, temperature_seed=tseed,
            limiter_func=_limit_fluid_cv, material_densities=sample_densities)
        
        # ~~~
        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries, dd=dd,
            comm_tag=_FluidOpStatesTag, limiter_func=_limit_fluid_cv)

        # fluid grad CV
        fluid_grad_cv = grad_cv_operator(
            dcoll, gas_model, boundaries, fluid_state, time=t, dd=dd,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradCVTag)

        # fluid grad T
        fluid_grad_temperature = grad_t_operator(
            dcoll, gas_model, boundaries, fluid_state, time=t, dd=dd,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradTempTag)

        fluid_rhs = ns_operator(
            dcoll, gas_model, fluid_state, boundaries,
            time=t, quadrature_tag=quadrature_tag, dd=dd,
            operator_states_quad=fluid_operator_states_quad,
            grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
            comm_tag=_FluidOperatorTag)

        darcy_flow = darcy_source_terms(fluid_state.cv, fluid_state.tv,
                                        fluid_state.wv)

        # ~~~
        idx_O2 = cantera_soln.species_index("O2")
        idx_CO2 = cantera_soln.species_index("CO2")
        sample_mass_rhs, sample_source_O2, sample_source_CO2 = \
            decomposition.get_source_terms(
                fluid_state.temperature, fluid_state.wv.tau,
                fluid_state.cv.species_mass_fractions[idx_O2])

        source_species = actx.np.zeros_like(fluid_state.cv.species_mass)
        source_species[idx_O2] = sample_source_O2
        source_species[idx_CO2] = sample_source_CO2

        reaction_rates = 0.10
        sample_heterogeneous_source = make_conserved(dim=2,
            mass=reaction_rates*sum(source_species),
            energy=zeros,
            momentum=make_obj_array([zeros, zeros]),
            species_mass=reaction_rates*source_species
        )
#        sample_heterogeneous_source = fluid_state.cv*0.0
#        sample_mass_rhs = zeros*0.0

        return make_obj_array([
            fluid_rhs + darcy_flow + sample_heterogeneous_source,
            zeros, reaction_rates*sample_mass_rhs])

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
        t = make_obj_array([current_t + zeros, zeros, current_t + zeros])
        dt = make_obj_array([current_dt, zeros, current_dt])
    else:
        t = current_t
        dt = current_dt

    if rank == 0:
        logging.info("Stepping.")

    stepper_state = make_obj_array([current_state.cv, temperature_seed,
                                    sample_densities])

    (current_step, current_t, current_cv) = \
        advance_state(
            rhs=my_rhs, timestepper=timestepper, state=stepper_state,
            pre_step_callback=my_pre_step, post_step_callback=my_post_step,
            dt=dt, t_final=t_final, t=t, istep=current_step,
            local_dt=local_dt, max_steps=500000, force_eval=force_eval)

#    # Dump the final data
#    if rank == 0:
#        logger.info("Checkpointing final state ...")

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
    casename = "slab"
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
