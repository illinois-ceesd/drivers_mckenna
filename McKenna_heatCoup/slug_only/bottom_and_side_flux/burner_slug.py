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
import os
import sys
import logging
import numpy as np
import pyopencl as cl
import cantera
from functools import partial
from dataclasses import dataclass, fields

from arraycontext import (
    dataclass_array_container, with_container_arithmetic,
    get_container_context_recursively
)

from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array

from meshmode.dof_array import DOFArray

from grudge import op
from grudge.reductions import integral
from grudge.trace_pair import TracePair, inter_volume_trace_pairs
from grudge.geometry.metrics import normal as normal_vector
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, BoundaryDomainTag, VolumeDomainTag
)

from mirgecom.utils import (
    force_evaluation as force_eval,
    project_from_base
)
from mirgecom.simutil import (
    check_step, get_sim_timestep, distribute_mesh, write_visfile,
    check_naninf_local, check_range_local, global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
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
from mirgecom.diffusion import (
    diffusion_operator,
    grad_operator as wall_grad_t_operator,
    NeumannDiffusionBoundary, PrescribedFluxDiffusionBoundary,
    grad_facial_flux_weighted, grad_facial_flux_central,
    diffusion_facial_flux_harmonic, diffusion_facial_flux_central
)
from mirgecom.wall_model import (
    SolidWallModel, SolidWallState, SolidWallConservedVars
)

#########################################################################

class _SampleMaskTag:
    pass


class _SolidGradTempTag:
    pass


class _SolidOperatorTag:
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
         use_tpe=False, use_profiling=False, casename=None, lazy=False,
         restart_file=None, user_input_file=False,
         force_wall_initialization=False):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=False,
                           use_einsum_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    # default i/o frequencies
    nviz = 1000
    nrestart = 50000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "ssprk43"
    t_final = 20.0
    niter = 4000001
    local_dt = False
    constant_cfl = False
    current_cfl = 0.4

    # discretization and model control
    order = 1

    current_dt = 10.0e-5
    mesh_filename = f"mesh_slug"

    temp_wall = 300.0
    wall_penalty_amount = 1.0

    restart_iterations = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # https://www.azom.com/article.aspx?ArticleID=2850
    # https://matweb.com/search/DataSheet.aspx?MatGUID=9aebe83845c04c1db5126fada6f76f7e&ckck=1
    wall_copper_rho = 8920.0
    wall_copper_cp = 385.0
    wall_copper_kappa = 391.1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import time
    t_start = time.time()
    t_shutdown = 720*60

# ############################################################################

    dim = 2

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    if integrator == "ssprk43":
        from mirgecom.integrators.ssprk import ssprk43_step
        timestepper = ssprk43_step
        force_eval_stepper = True

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        if constant_cfl is False:
            print(f"\tt_final = {t_final}")
        else:
            print(f"\tconstant_cfl = {constant_cfl}")
            print(f"\tcurrent_cfl = {current_cfl}")
            print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

# ############################################################################

    restart_step = None
    if restart_file is None:

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0

        if rank == 0:
            mesh_path = "mesh_slug-v2.msh"
#            os.system(f"rm -rf {omesh_path} {mesh_path}")
#            os.system(f"gmsh {geo_path} -2 -o {omesh_path}")
#            os.system(f"gmsh {omesh_path} -save -format msh2 -o {mesh_path}")

            print(f"Reading mesh from {mesh_path}")

        comm.Barrier()

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_path, force_ambient_dim=dim,
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
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        first_step = restart_step+0

        assert comm.Get_size() == restart_data["num_parts"]

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    if rank == 0:
        logger.info("Done making discretization")

    from grudge.dof_desc import DD_VOLUME_ALL
    dd_vol_solid = DD_VOLUME_ALL
    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))
    solid_zeros = force_eval(actx, solid_nodes[0]*0.0)

    # ~~~~~~~~~~
    from grudge.dt_utils import characteristic_lengthscales
    char_length_solid = force_eval(
        actx, characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid))

##########################################################################

    # {{{ Initialize wall model

#    def _solid_density_func():
#        return wall_copper_rho + solid_zeros

    def _solid_enthalpy_func(temperature, **kwargs):
        return wall_copper_cp*temperature

    def _solid_heat_capacity_func(temperature, **kwargs):
        return wall_copper_cp + solid_zeros

    def _solid_thermal_cond_func(temperature, **kwargs):
        return wall_copper_kappa + solid_zeros

    solid_wall_model = SolidWallModel(
        # density_func=_solid_density_func,
        enthalpy_func=_solid_enthalpy_func,
        heat_capacity_func=_solid_heat_capacity_func,
        thermal_conductivity_func=_solid_thermal_cond_func)

    # }}}

#############################################################################

    def _get_copper_solid_state(cv):
        wdv = solid_wall_model.dependent_vars(cv)
        return SolidWallState(cv=cv, dv=wdv)

    get_solid_state = actx.compile(_get_copper_solid_state)

##############################################################################

    if restart_file is None or force_wall_initialization:

        print("Starting the wall from scratch!!")

        wv_tseed = force_eval(actx, temp_wall + solid_zeros)

        from mirgecom.materials.initializer import SolidWallInitializer
        solid_init = SolidWallInitializer(temperature=300.0,
                                          material_densities=wall_copper_rho)
        solid_cv = solid_init(solid_nodes, solid_wall_model)

    else:
        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=restart_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            solid_connection = make_same_mesh_connection(
                actx, dcoll.discr_from_dd(dd_vol_solid),
                restart_dcoll.discr_from_dd(dd_vol_solid))
            solid_cv = wall_connection(restart_data["wv"])
        else:
            solid_cv = restart_data["wv"]

#########################################################################

    solid_cv = force_eval(actx, solid_cv)
    solid_state = get_solid_state(solid_cv)

#########################################################################

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"

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

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)
##############################################################################
    
    #dd_hot = dd_vol_solid.trace("bottom")
    #nodes_intfc = op.project(dcoll, dd_vol_solid, dd_hot, solid_nodes[0])
    #flux = PrescribedFluxDiffusionBoundary(-8.0*10000.0*nodes_intfc)
    flux_bot = PrescribedFluxDiffusionBoundary(-3.93*10000.0)
    flux_gap = PrescribedFluxDiffusionBoundary(-0.5*10.0*1000.0)
    wall_symmetry = NeumannDiffusionBoundary(0.0)
    solid_boundaries = {
        dd_vol_solid.trace("sym").domain_tag: wall_symmetry,
        dd_vol_solid.trace("top").domain_tag: wall_symmetry,
        dd_vol_solid.trace("right").domain_tag: wall_symmetry,
        dd_vol_solid.trace("gap").domain_tag: flux_gap,
        dd_vol_solid.trace("bottom").domain_tag: flux_bot
    }

##############################################################################

    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

#########################################################################

    def my_write_viz(
            step, t, dt, solid_state, grad_t_solid=None):

        wv = solid_state.cv
        wdv = solid_state.dv
        solid_viz_fields = [
            # ("wv_energy", wv.energy),
            # ("cfl", solid_zeros),  # FIXME
            # ("wall_kappa", wdv.thermal_conductivity),
            # ("wall_progress", wdv.tau),
            ("wall_temperature", wdv.temperature),
            # ("wall_grad_t", grad_t_solid),
        ]

        solid_viz_fields.append(("wv_mass", wv.mass))

        print("Writing solution file...")
        write_visfile(
            dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        if rank == 0:
            print("Writing restart file...")

        wv = state

        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_file:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "wv": wv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

#########################################################################

    def my_health_check(temperature):
        health_error = False

        if check_naninf_local(dcoll, "vol", temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

##############################################################################

    from arraycontext import outer
    from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
    from meshmode.discretization.connection import FACE_RESTR_ALL

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    off_axis_x = 1e-7
    solid_nodes_are_off_axis = actx.np.greater(solid_nodes[0], off_axis_x)

    # ~~~~~~~
    def axisym_source_solid(actx, dcoll, solid_state, grad_t):
        dkappadr = 0.0*solid_nodes[0]

        temperature = solid_state.dv.temperature
        kappa = solid_state.dv.thermal_conductivity
        
        qr = - (kappa*grad_t)[0]
#        d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0], axisym_wall_boundaries, dd_vol_solid,  "symmetry")[0]
#        dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)
                
        source_mass = solid_state.cv.mass*0.0

        source_rhoE_dom = - qr
        source_rhoE_sng = 0.0 #- dqrdr
        source_rhoE = actx.np.where( solid_nodes_are_off_axis,
                          source_rhoE_dom/solid_nodes[0], source_rhoE_sng)

        return SolidWallConservedVars(mass=source_mass, energy=source_rhoE)

    compiled_axisym_source_solid = actx.compile(axisym_source_solid)

##############################################################################

    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        _wv = force_eval(actx, state)
        solid_state = get_solid_state(_wv)
        wv = solid_state.cv
        wdv = solid_state.dv

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            t_elapsed = time.time() - t_start
            if t_shutdown - t_elapsed < 60.0:
                my_write_restart(step, t, state)
                logmgr.close()
                sys.exit()

            ngarbage = 10
            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            if do_health:
                health_errors = global_reduce(my_health_check(wdv.temperature), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            file_exists = os.path.exists("write_solution")
            if file_exists:
              os.system("rm write_solution")
              do_viz = True
        
            file_exists = os.path.exists("write_restart")
            if file_exists:
              os.system("rm write_restart")
              do_restart = True

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, solid_state=solid_state)

                # garbage is getting out of control without this
                gc.freeze()

            if do_restart:
                my_write_restart(step, t, state)

                # garbage is getting out of control without this
                gc.freeze()

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, solid_state=solid_state)
            raise

        return state, dt

    def my_rhs(t, state):
        
        # construct wall state
        solid_state = _get_copper_solid_state(state)
        wv = solid_state.cv
        wdv = solid_state.dv

        solid_grad_t = wall_grad_t_operator(
            dcoll, wdv.thermal_conductivity, solid_boundaries, wdv.temperature)

        solid_energy_rhs = diffusion_operator(
            dcoll, wdv.thermal_conductivity, solid_boundaries,
            wdv.temperature,
            penalty_amount=wall_penalty_amount,
            dd=dd_vol_solid,
            grad_u=solid_grad_t,
            comm_tag=_SolidOperatorTag)

        solid_sources = axisym_source_solid(
            actx, dcoll, solid_state, solid_grad_t)

        solid_rhs = SolidWallConservedVars(mass=solid_zeros,
                                           energy=solid_energy_rhs)

        return solid_rhs + solid_sources

    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer.start_sub_timer():
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

    stepper_state = solid_state.cv

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt, t=current_t,
                      t_final=t_final,
                      force_eval=force_eval_stepper, state=stepper_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    logging.basicConfig(
        #format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        format="%(message)s",
        level=logging.INFO)

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
    parser.add_argument("--esdg", action="store_true",
        help="use flux-differencing/entropy stable DG for inviscid computations.")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--tpe", action="store_true",
        help="use quadrilateral elements.")
    parser.add_argument("--init_wall", action="store_true", default=False,
        help="Force wall initialization from scratch.")

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

    from warnings import warn
    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    if args.init_wall:
        print("Starting the wall from scratch!!")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    main(actx_class, use_logmgr=args.log, casename=casename, use_tpe=args.tpe,
         restart_file=restart_file, user_input_file=input_file,
         force_wall_initialization=args.init_wall)
