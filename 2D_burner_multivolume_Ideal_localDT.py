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
import pyopencl.tools as cl_tools
from functools import partial
from dataclasses import dataclass, fields

#from arraycontext import thaw, freeze
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.dof_desc import BoundaryDomainTag
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.trace_pair import inter_volume_trace_pairs
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, VolumeDomainTag
)

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    IsothermalWallBoundary,
    #PressureOutflowBoundary
    OutflowBoundary,
    SymmetryBoundary,
    PrescribedFluidBoundary,
    #MyPrescribedBoundary_v3,
    AdiabaticNoslipWallBoundary,
    #LinearizedBoundary
)

from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient,
    make_conserved
)

from mirgecom.transport import (
    PowerLawTransport,
    ArtificialViscosityTransport,
    #MixtureAveragedTransport
)
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

from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    coupled_grad_t_operator,
    coupled_ns_heat_operator
)

from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary
)

#########################################################################

class Burner2D_Ideal:

    def __init__(
            self, *, dim=2,
                     sigma=0.001,
                     pressure,
                     temperature,
                     speedup_factor):

        self._dim = dim
        self._sigma = sigma
        self._pres = pressure
        self._speedup_factor = speedup_factor
        self._temp = temperature

    def __call__(self, x_vec, eos, flow_rate, state_minus=None, init=False):

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

        #~~~ temperature
        temp = (self._temp)*(1.0-upper_atm) + 300.0*upper_atm
        temperature = temp*core + 300.0*(1.0 - core)

        #~~~ velocity
        v_inlet = flow_rate/20.0*(0.117892*self._speedup_factor)
        v_shroud = 20.0/20.0*(0.117892*self._speedup_factor)*2.769230769230767
        smoothY = ((self._temp/300.0)*v_inlet)*(1.0-upper_atm)
        u_x = 0.0*x_vec[0]
        u_y = core*smoothY + shroud*v_shroud*(1.0-upper_atm)
        velocity = make_obj_array([u_x, u_y])

        #~~~ 

#        B1 = -actx.np.tanh(1.0/(_sigma)*(x_vec[0] - ext_diam))
#        B2 = 0.5*(1.0 - actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - self._flaLoc)))
#        burner = 0.5*(B1 + actx.np.abs(B1))*B2

        #pressure = (self._pres - 101325.0)*burner + 101325.0
        pressure = self._pres + 0.0*x_vec[0]
        mass = eos.get_density(pressure, temperature)

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



def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return DOFArray(actx, tuple(group_arrays))


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallVars:
    mass: DOFArray
    energy: DOFArray
    ox_mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name)
                                    for f in fields(WallVars)))


class WallModel:
    """Model for calculating wall quantities."""
    def __init__(
            self,
            insert_mask, surround_mask,
            insert_heat_capacity, surround_heat_capacity,
            insert_mass_to_thermal_conductivity, surround_thermal_conductivity,
            insert_mass_to_effective_surface_area,
            insert_oxygen_diffusivity, *,
            gas_const, mw_o2, mw_co):
        self._insert_mask = insert_mask
        self._surround_mask = surround_mask
        self._heat_capacity = (
            insert_heat_capacity * insert_mask
            + surround_heat_capacity * surround_mask)
        self._insert_mass_to_thermal_conductivity = \
            insert_mass_to_thermal_conductivity
        self._surround_thermal_conductivity = surround_thermal_conductivity
        self._insert_mass_to_effective_surface_area = \
            insert_mass_to_effective_surface_area
        self._oxygen_diffusivity = insert_oxygen_diffusivity * insert_mask
        self._gas_const = gas_const
        self._mw_o2 = mw_o2
        self._mw_co = mw_co

    @property
    def heat_capacity(self):
        return self._heat_capacity

    def mass_loss_rate(self, wv):
        actx = wv.mass.array_context
        temperature = wv.energy/(wv.mass * self.heat_capacity)
        alpha = (
            (0.00143+0.01*actx.np.exp(-1450.0/temperature))
            / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
        k = alpha * actx.np.sqrt(
            (self._gas_const*temperature)/(2.0*np.pi*self._mw_o2))
        eff_surf_area = self._insert_mass_to_effective_surface_area(wv.mass)
        return (self._mw_co/self._mw_o2 - 0.5)*wv.ox_mass*k*eff_surf_area

    def thermal_conductivity(self, mass):
        return (
            self._insert_mask * self._insert_mass_to_thermal_conductivity(mass)
            + self._surround_mask * self._surround_thermal_conductivity)

    def thermal_diffusivity(self, mass, thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(mass)
        return thermal_conductivity/(mass * self.heat_capacity)

    @property
    def oxygen_diffusivity(self):
        return self._oxygen_diffusivity



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

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    restart_pattern = rst_path + "{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 250
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 1.0e-8
    t_final = 2.0

    niter = 100001
    
    # discretization and model control
    order = 2

    local_dt = True
    constant_cfl = True
    current_cfl = 0.2

    speedup_factor = 15.0

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

    rho_atmosphere = 101325.0/(eos.gas_const()*300.0)

##########################################################################

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    # for graphite
    wall_insert_rho = 1625
    wall_insert_cp = 770
    wall_insert_kappa = 247.5

    wall_surround_rho = 1625
    wall_surround_cp = 770
    wall_surround_kappa = 247.5

    # wall stuff
    wall_penalty_amount = 25
    wall_time_scale = speedup_factor

    temp_wall = 300

    wall_insert_ox_diff = 1e-4

##############################################################################

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

##################################

    flow_init = Burner2D_Ideal(dim=dim, sigma=0.00020, temperature=2100,
         pressure=101325.0, speedup_factor=speedup_factor)

    ref_state = Burner2D_Ideal(dim=dim, sigma=0.00020, temperature=2100,
         pressure=101325.0, speedup_factor=speedup_factor)

##############################################################################

    mesh_filename = "mesh_01m_ReactionRate-v2.msh"
    restart_step = None
    if restart_file is None:        
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "fluid": ["fluid"],
                "solid": ["wall_insert", "wall_surround"]
                }
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

    local_nelements = (
          volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["solid"][0].nelements)

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
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_wall = DOFDesc(VolumeDomainTag("solid"), DISCR_TAG_BASE)

    wall_vol_discr = dcoll.discr_from_dd(dd_vol_wall)
    wall_tag_to_elements = volume_to_local_mesh_data["solid"][1]
    wall_insert_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["wall_insert"])
    wall_surround_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["wall_surround"])

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    wall_nodes = actx.thaw(dcoll.nodes(dd_vol_wall))

#####################################################################################

    def _create_wall_derived(wv):
        wall_kappa = wall_model.thermal_conductivity(wv.mass)
        wall_temperature = wv.energy/(wv.mass * wall_model.heat_capacity)
        return make_obj_array([wall_kappa, wall_temperature])

    create_wall_derived_compiled = actx.compile(_create_wall_derived)

    def _get_wall_kappa(wv):
        return wall_model.thermal_conductivity(wv.mass)

    get_wall_kappa_compiled = actx.compile(_get_wall_kappa)

    def _get_wv(wv):
        return wv

    get_wv = actx.compile(_get_wv)

    def _get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            #limiter_func=_limit_fluid_cv, limiter_dd=dd_vol_fluid,
        )

    get_fluid_state = actx.compile(_get_fluid_state)

#####################################################################################

    def vol_min(dd_vol, x):
        return actx.to_numpy(nodal_min(dcoll, dd_vol, x,
                                       initial=np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(nodal_max(dcoll, dd_vol, x,
                                       initial=-np.inf))[()]

#########################################################################

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                               
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
        
#################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = flow_init(fluid_nodes, eos, flow_rate=60.0)

        wall_mass = (
            wall_insert_rho * wall_insert_mask
            + wall_surround_rho * wall_surround_mask)
        wall_cp = (
            wall_insert_cp * wall_insert_mask
            + wall_surround_cp * wall_surround_mask)
        current_wv = WallVars(
            mass=wall_mass,
            energy=wall_mass * wall_cp * temp_wall,
            ox_mass=0*wall_mass)

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
        else:
            current_cv = restart_data["state"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)


    current_cv = force_evaluation(actx, current_cv)
    current_wv = force_evaluation(actx, current_wv)

    current_state = get_fluid_state(current_cv)

#####################################################################################

    from mirgecom.boundary import PrescribedFluidBoundary
    class LinearizedBoundary(PrescribedFluidBoundary):
        r"""Characteristics outflow BCs for linearized Euler equations.

        Implement non-reflecting outflow based on characteristic variables for
        the Euler equations assuming small perturbations based on [Giles_1988]_.
        The equations assume an uniform, steady flow and linerize the Euler eqs.
        in this reference state, yielding a linear equation in the form

        .. math::
            \frac{\partial U}{\partial t} + A \frac{\partial U}{\partial x} +
            B \frac{\partial U}{\partial y} = 0

        where where U is the vector of perturbation (primitive) variables and
        the coefficient matrices A and B are constant matrices based on the
        uniform, steady variables.

        Using the linear hyperbolic system theory, this equation can be further
        simplified by ignoring the y-axis terms (tangent) such that wave propagation
        occurs only along the x-axis direction (normal). Then, the eigendecomposition
        results in a orthogonal system where the wave have characteristic directions
        of propagations and enable the creation of non-reflecting outflow boundaries.

        This can also be applied for Navier-Stokes equations in regions where
        viscous effects are not dominant, such as the far-field.
        """

        def __init__(self, dim, free_stream_state=None,
                     free_stream_density=None,
                     free_stream_velocity=None,
                     free_stream_pressure=None,
                     free_stream_temperature=None,
                     free_stream_species_mass_fractions=None):
            """Initialize the boundary condition object."""
            self._ref_state = free_stream_state
            self._mass = free_stream_density
            self._velocity = free_stream_velocity
            self._pressure = free_stream_pressure
            self._temperature = free_stream_temperature
            self._Y = free_stream_species_mass_fractions

            PrescribedFluidBoundary.__init__(
                self, boundary_state_func=self.outflow_state
            )

        def outflow_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            """Non-reflecting outflow."""
            if self._ref_state is None:
                ref_mass = self._mass
                ref_velocity = self._velocity
                ref_pressure = self._pressure
                species_mass_fractions = self._Y
            else:
                ref_mass = self._ref_state.cv.mass
                ref_velocity = self._ref_state.velocity
                ref_pressure = self._ref_state.pressure
                species_mass_fractions = self._ref_state.cv.species_mass_fractions

            actx = state_minus.array_context
            nhat = actx.thaw(dcoll.normal(dd_bdry))

            rtilde = state_minus.cv.mass - ref_mass
            utilde = state_minus.velocity[0] - ref_velocity[0]
            vtilde = state_minus.velocity[1] - ref_velocity[1]
            ptilde = state_minus.dv.pressure - ref_pressure

            un_tilde = +utilde*nhat[0] + vtilde*nhat[1]
            ut_tilde = -utilde*nhat[1] + vtilde*nhat[0]

            a = state_minus.speed_of_sound

            c1 = -rtilde*a**2 + ptilde
            c2 = ref_mass*a*ut_tilde
            c3 = ref_mass*a*un_tilde + ptilde
            c4 = 0.0  # zero-out the last characteristic variable
            r_tilde_bnd = 1.0/(a**2)*(-c1 + 0.5*c3 + 0.5*c4)
            un_tilde_bnd = 1.0/(ref_mass*a)*(0.5*c3 - 0.5*c4)
            ut_tilde_bnd = 1.0/(ref_mass*a)*c2
            p_tilde_bnd = 0.5*c3 + 0.5*c4

            mass = r_tilde_bnd + ref_mass
            u_x = ref_velocity[0] + (nhat[0]*un_tilde_bnd - nhat[1]*ut_tilde_bnd)
            u_y = ref_velocity[1] + (nhat[1]*un_tilde_bnd + nhat[0]*ut_tilde_bnd)
            pressure = p_tilde_bnd + ref_pressure

            kin_energy = 0.5*mass*(u_x**2 + u_y**2)
            if state_minus.is_mixture:
                gas_const = gas_model.eos.gas_const(state_minus.cv)
                temperature = ref_pressure/(ref_mass*gas_const)
                int_energy = mass*gas_model.eos.get_internal_energy(
                    temperature, species_mass_fractions)
            else:
                int_energy = pressure/(gas_model.eos.gamma() - 1.0)

            boundary_cv = (
                make_conserved(dim=2, mass=mass,
                               energy=kin_energy + int_energy,
                               momentum=make_obj_array([u_x*mass, u_y*mass]),
                               species_mass=state_minus.cv.species_mass)
            )

            return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                    temperature_seed=state_minus.temperature)

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
        inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
        inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos, flow_rate=60.0)
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
            #limiter_func=_limit_fluid_cv, limiter_dd=dd_bdry
        )

    wall_bnd = AdiabaticNoslipWallBoundary()
    inflow_bnd = PrescribedFluidBoundary(boundary_state_func=inlet_bnd_state_func)
    outflow_bnd = OutflowBoundary(boundary_pressure=101325.0)
    symmmetry_bnd = SymmetryBoundary()
    linear_bnd = LinearizedBoundary(dim=2, free_stream_density=rho_atmosphere,
        free_stream_pressure=101325.0, free_stream_velocity=np.zeros(shape=(dim,)))
    
    wall_symmetry = DirichletDiffusionBoundary(temp_wall)

    fluid_boundaries = {
        dd_vol_fluid.trace("inlet").domain_tag: inflow_bnd,
        dd_vol_fluid.trace("symmetry").domain_tag: symmmetry_bnd,
        dd_vol_fluid.trace("linear").domain_tag: linear_bnd,
        dd_vol_fluid.trace("outlet").domain_tag: outflow_bnd,
        dd_vol_fluid.trace("wall").domain_tag: wall_bnd}

    wall_boundaries = {
        dd_vol_wall.trace("wall_sym").domain_tag: wall_symmetry
    }

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

    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=fluid_nodes))
    ref_cv = force_evaluation(actx, ref_state(fluid_nodes, eos, flow_rate=60.0))

##############################################################################

    def insert_mass_to_kappa(mass):
        # TODO
        # mass_loss_frac = (wall_insert_rho - mass)/wall_insert_rho
        # ...
        return wall_insert_kappa

    def insert_mass_to_effective_surface_area(mass):
        # TODO
        # mass_loss_frac = (wall_insert_rho - mass)/wall_insert_rho
        # ...
        return 1

    mw_o2 = 15.999*2
    mw_co = 28.010
    gas_const = eos.gas_const()
    wall_model = WallModel(
        wall_insert_mask, wall_surround_mask,
        wall_insert_cp, wall_surround_cp,
        insert_mass_to_kappa, wall_surround_kappa,
        insert_mass_to_effective_surface_area,
        wall_insert_ox_diff,
        gas_const=gas_const, mw_o2=mw_o2, mw_co=mw_co)

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    wall_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_wall)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=current_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=current_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

#########################################################################

    from grudge.dt_utils import characteristic_lengthscales, dt_geometric_factors
    def my_write_viz(step, t, dt, fluid_state, wv, wall_kappa, wall_temperature):

        fluid_viz_fields = [
            ("CV_rho", fluid_state.cv.mass),
            ("CV_rhoU", fluid_state.cv.momentum),
            ("CV_rhoE", fluid_state.cv.energy),
            ("DV_P", fluid_state.pressure),
            ("DV_T", fluid_state.temperature),
            ("DV_U", fluid_state.velocity[0]),
            ("DV_V", fluid_state.velocity[1]),
            ("dt", dt[0]),
            ("sponge", sponge_sigma)
        ]

        cell_alpha = wall_model.thermal_diffusivity(wv.mass, wall_kappa)
        wall_viz_fields = [
            ("wv", wv),
            ("wall_kappa", wall_kappa),
            ("wall_temperature", wall_temperature),
            ("wall_alpha", cell_alpha),
            ("dt", dt[1]),
            #("dt" if constant_cfl else "cfl", ts_field_wall)
        ]
                      
        print('Writing solution file...')
        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t,
            overwrite=True, comm=comm)
        write_visfile(
            dcoll, wall_viz_fields, wall_visualizer,
            vizname=vizname+"-wall", step=step, t=t,
            overwrite=True, comm=comm)

    from mirgecom.restart import write_restart_file
    def my_write_restart(step, t, state):
        if rank == 0:
            print('Writing restart file...')

        cv, wv = state
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "wv": wv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

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
          
    from mirgecom.flux import num_flux_central
    from grudge.dof_desc import as_dofdesc

    def coupled_gradient_operator(dcoll, gas_model, fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries, fluid_field, wall_field, wall_kappa,
        *, time=0., fluid_numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE, _kappa_inter_vol_tpairs=None,
        _temperature_inter_vol_tpairs=None, _fluid_operator_states_quad=None):

        fluid_boundaries = {
            as_dofdesc(bdtag).domain_tag: bdry
            for bdtag, bdry in fluid_boundaries.items()}
        wall_boundaries = {
            as_dofdesc(bdtag).domain_tag: bdry
            for bdtag, bdry in wall_boundaries.items()}

        fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
            get_interface_boundaries(dcoll, gas_model, fluid_dd, wall_dd,
                fluid_state, wall_kappa, wall_temp,
                _kappa_inter_vol_tpairs=_kappa_inter_vol_tpairs,
                _temperature_inter_vol_tpairs=_temperature_inter_vol_tpairs)

        fluid_all_boundaries_no_grad = {}
        fluid_all_boundaries_no_grad.update(fluid_boundaries)
        fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

        wall_all_boundaries_no_grad = {}
        wall_all_boundaries_no_grad.update(wall_boundaries)
        wall_all_boundaries_no_grad.update(wall_interface_boundaries_no_grad)

        return (
            my_derivative_function(actx, dcoll, fluid_field, fluid_all_boundaries_no_grad, dd_vol_fluid, 'replicate'), #XXX
            my_derivative_function(actx, dcoll,  wall_field,  wall_all_boundaries_no_grad,  dd_vol_wall, 'replicate')  #XXX
        )

    def _coupled_grad_t_operator(dcoll, fluid_boundaries, wall_boundaries, fluid_state, wall_kappa, wall_temperature):
        return coupled_grad_t_operator(dcoll, gas_model, dd_vol_fluid, dd_vol_wall,
            fluid_boundaries, wall_boundaries, fluid_state, wall_kappa, wall_temperature)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.boundary import DummyBoundary
    from mirgecom.diffusion import DiffusionBoundary, diffusion_facial_flux
    class DummyDiffusionBoundary(DiffusionBoundary):
        def get_grad_flux(self, dcoll, dd_bdry, kappa_minus, u_minus):
            return None
        def get_diffusion_flux(self, dcoll, dd_bdry, kappa_minus, u_minus,
            grad_u_minus, lengthscales_minus, *, penalty_amount=None):
            return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from arraycontext import outer
    from grudge.trace_pair import TracePair, interior_trace_pairs, tracepair_with_discr_tag
    from grudge import op
    from meshmode.discretization.connection import FACE_RESTR_ALL

    from grudge.trace_pair import inter_volume_trace_pairs
    class _MyGradTag:
        pass

    fluid_field = fluid_nodes[0]*0.0
    wall_field = wall_nodes[0]*0.0
    pairwise_field = {
        (dd_vol_fluid, dd_vol_wall): (fluid_field, wall_field)}
    pairwise_field_tpairs = inter_volume_trace_pairs(
        dcoll, pairwise_field, comm_tag=_MyGradTag)
    field_tpairs_F = pairwise_field_tpairs[dd_vol_wall, dd_vol_fluid]
    field_tpairs_W = pairwise_field_tpairs[dd_vol_fluid, dd_vol_wall]

    axisym_fluid_boundaries = {}
    axisym_fluid_boundaries.update(fluid_boundaries)
    axisym_fluid_boundaries.update({
            tpair.dd.domain_tag: DummyBoundary()
            for tpair in field_tpairs_F})

    axisym_wall_boundaries = {}
    axisym_wall_boundaries.update(wall_boundaries)
    axisym_wall_boundaries.update({
            tpair.dd.domain_tag: DummyDiffusionBoundary()
            for tpair in field_tpairs_W})

    def my_derivative_function(actx, dcoll, field, field_bounds, dd_vol, bnd_cond, verbose=False):    

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

            if bnd_cond == 'symmetry' and bdtag  == '-0':
                ext_soln_quad = 0.0*int_soln_quad
            else:
                ext_soln_quad = 1.0*int_soln_quad

            bnd_tpair = TracePair(bdtag, interior=int_soln_quad,
                                         exterior=ext_soln_quad)
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.fluid import velocity_gradient, species_mass_fraction_gradient

    off_axis_x = 1e-7
    fluid_nodes_are_off_axis = actx.np.greater(fluid_nodes[0], off_axis_x)  # noqa
    wall_nodes_are_off_axis  = actx.np.greater( wall_nodes[0], off_axis_x)  # noqa
   
    def axisym_source_fluid(actx, dcoll, state, grad_cv, grad_t):      
        cv = state.cv
        dv = state.dv
        
        mu = state.tv.viscosity
        beta = transport_model.volume_viscosity(cv, dv, eos)
        kappa = state.tv.thermal_conductivity
        #d_ij = state.tv.species_diffusivity
        
        grad_v = velocity_gradient(cv,grad_cv)
        grad_y = species_mass_fraction_gradient(cv, grad_cv)

        u = state.velocity[0]
        v = state.velocity[1]

        dudr = grad_v[0][0]
        dudy = grad_v[0][1]
        dvdr = grad_v[1][0]
        dvdy = grad_v[1][1]
        
        drhoudr = (grad_cv.momentum[0])[0]

        d2udr2   = my_derivative_function(actx, dcoll,  dudr, fluid_boundaries, dd_vol_fluid, 'replicate')[0] #XXX
        d2vdr2   = my_derivative_function(actx, dcoll,  dvdr, fluid_boundaries, dd_vol_fluid, 'replicate')[0] #XXX
        d2udrdy  = my_derivative_function(actx, dcoll,  dudy, fluid_boundaries, dd_vol_fluid, 'replicate')[0] #XXX
                
        dmudr    = my_derivative_function(actx, dcoll,    mu, fluid_boundaries, dd_vol_fluid, 'replicate')[0]
        dbetadr  = my_derivative_function(actx, dcoll,  beta, fluid_boundaries, dd_vol_fluid, 'replicate')[0]
        dbetady  = my_derivative_function(actx, dcoll,  beta, fluid_boundaries, dd_vol_fluid, 'replicate')[1]
       #dkappadr = my_derivative_function(actx, dcoll, kappa, fluid_boundaries, dd_vol_fluid, 'replicate')[0]
        
        #dyidr = grad_y[:,0]
        #dyi2dr2 = my_derivative_function(actx, dcoll,     dyidr, 'replicate')[:,0]   #XXX
        
        qr = - kappa*grad_t[0]
       #d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0], axisym_fluid_boundaries, dd_vol_fluid, 'replicate')[0]
        dqrdr = 0.0 #- (dkappadr*grad_t[0] + kappa*d2Tdr2) #XXX
        
        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              fluid_nodes_are_off_axis, u/fluid_nodes[0], dudr )

        #dtaurydr = my_derivative_function(actx, dcoll, tau_ry)[0]
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
        
        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                       momentum=make_obj_array([source_rhoU, source_rhoV]),
        )

    def axisym_source_wall(actx, dcoll, kappa, temperature, grad_t):              
        dkappadr = 0.0*wall_nodes[0]
        
        kappa = 1.0 #FIXME there is soemthing odd here.. Check the equations!
        qr = - kappa*grad_t[0]
#        d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0], axisym_wall_boundaries, dd_vol_wall,  'symmetry')[0]
#        dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)
        
        source_mass = 0.0*wall_nodes[0]
        source_ox_mass = 0.0*wall_nodes[0]

        source_rhoE_dom = - qr
        source_rhoE_sng = 0.0  # - dqrdr #XXX
        source_rhoE = actx.np.where( wall_nodes_are_off_axis,
                          source_rhoE_dom/wall_nodes[0], source_rhoE_sng )

        source = WallVars(mass=source_mass,
                energy=wall_mass * wall_cp * source_rhoE, ox_mass=source_ox_mass)

        return source
    
    def gravity_source_terms(cv):
        """Gravity."""
        delta_rho = cv.mass - rho_atmosphere
        return make_conserved(dim=2, mass=cv.mass*0.0,
            energy=delta_rho*cv.velocity[1]*-9.80665,
            momentum=make_obj_array([cv.mass*0.0, delta_rho*-9.80665]),
            species_mass=cv.species_mass*0.0
        )

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        cv, wv = state
        cv = force_evaluation(actx, cv)
        wv = force_evaluation(actx, wv)

        #TODO drop order

        fluid_state = get_fluid_state(cv)
        wall_kappa, wall_temperature = create_wall_derived_compiled(wv)

        if local_dt:
            t = force_evaluation(actx, t)
            dt_fluid = force_evaluation(actx, 
                 get_sim_timestep(dcoll, current_state, current_t,
                         current_dt, current_cfl, gas_model,
                         constant_cfl=constant_cfl, local_dt=local_dt, fluid_dd=dd_vol_fluid)
            )
            dt_wall = force_evaluation(actx, current_dt + wall_nodes[0]*0.0) #FIXME
            dt = make_obj_array([dt_fluid, dt_wall])
        else:
            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                      t_final, constant_cfl, local_dt, dd_vol_fluid)

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            state = make_obj_array([cv, wv])
            wv = get_wv(wv)

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
                my_write_restart(step, t, state)

            if do_viz:               
                my_write_viz(step, t, dt, fluid_state, wv, wall_kappa, wall_temperature)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step, t, dt, fluid_state, wv, wall_kappa, wall_temperature)
            raise

        return make_obj_array([fluid_state.cv, wv]), dt


    def my_rhs(t, state):

        #XXX Do the interface for axisymmetric depends on ( 1/r dT/dr ) or the terms cancel out?

        cv, wv = state

        #TODO drop order

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
        wall_kappa, wall_temperature = _create_wall_derived(wv)

        # update wall model
        wall_kappa = wall_model.thermal_conductivity(wv.mass)

        #~~~~~~~~~~~~~

        fluid_rhs, wall_energy_rhs, grad_cv_fluid, grad_t_fluid = \
            coupled_ns_heat_operator(
                dcoll, gas_model, dd_vol_fluid, dd_vol_wall, fluid_boundaries, wall_boundaries,
                fluid_state, wall_kappa, wall_temperature, time=t, quadrature_tag=quadrature_tag,
                return_fluid_gradients=True
            )

        #~~~~~~~~~~~~~

        coupled_grad_t = _coupled_grad_t_operator(dcoll, fluid_boundaries, wall_boundaries, fluid_state, wall_kappa, wall_temperature)
        grad_t_fluid = coupled_grad_t[0]
        grad_t_wall = coupled_grad_t[1]

        #~~~~~~~~~~~~~

        fluid_sources = (
            sponge_func(cv=cv, cv_ref=ref_cv, sigma=sponge_sigma)
            + gravity_source_terms(cv)
            + axisym_source_fluid(actx, dcoll, fluid_state, grad_cv_fluid, grad_t_fluid)
        )

        #~~~~~~~~~~~~~

        wall_mass_rhs = -wall_model.mass_loss_rate(wv)
        wall_energy_rhs = wall_time_scale * wall_energy_rhs #FIXME
        fluid_ox_mass = 0.0 #cv.species_mass[cantera_soln.species_index("O2")]

        pairwise_ox = {
            (dd_vol_fluid, dd_vol_wall): (fluid_ox_mass, wv.ox_mass)}
        pairwise_ox_tpairs = inter_volume_trace_pairs(
            dcoll, pairwise_ox, comm_tag=_OxCommTag)
        ox_tpairs = pairwise_ox_tpairs[dd_vol_fluid, dd_vol_wall]

        wall_ox_boundaries = {
            dd_vol_wall.trace("wall_sym").domain_tag:
                DirichletDiffusionBoundary(0)}
        wall_ox_boundaries.update({
            tpair.dd.domain_tag: DirichletDiffusionBoundary(tpair.ext)
            for tpair in ox_tpairs})

        wall_ox_mass_rhs = wall_time_scale * diffusion_operator(
            dcoll, wall_model.oxygen_diffusivity, wall_ox_boundaries, wv.ox_mass,
            penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag, dd=dd_vol_wall)

        wall_rhs = WallVars(mass=wall_mass_rhs, energy=wall_energy_rhs,
            ox_mass=wall_ox_mass_rhs)

        #~~~~~~~~~~~~~

        fluid_rhs = fluid_rhs + fluid_sources
        wall_rhs = wall_rhs + axisym_source_wall(actx, dcoll, wall_kappa, wall_temperature, grad_t_wall)

        return make_obj_array([fluid_rhs, wall_rhs])

    class _OxCommTag:
        pass

#    def my_post_step(step, t, dt, state):
#        min_dt = np.min(actx.to_numpy(dt))
#        if logmgr:
#            set_dt(logmgr, min_dt)         
#            logmgr.tick_after()

#        return state, dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, 1.0)         
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([current_state.cv, current_wv])

    if local_dt == True:
        dt_fluid = force_evaluation(actx, 
             get_sim_timestep(dcoll, current_state, current_t,
                     current_dt, current_cfl, gas_model,
                     constant_cfl=constant_cfl, local_dt=local_dt, fluid_dd=dd_vol_fluid)
        )
        dt_wall = force_evaluation(actx, current_dt +wall_nodes[0]*0.0)
        t_fluid = force_evaluation(actx, current_t + dt_fluid*0.0)
        t_wall = force_evaluation(actx, current_t + dt_wall*0.0)
        dt = make_obj_array([dt_fluid, dt_wall])
        t = make_obj_array([dt_fluid, dt_wall])
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt,
                      t=t, t_final=t_final,
                      max_steps=niter, local_dt=local_dt,
                      force_eval=force_eval,
                      state=stepper_state)

    current_cv, current_wv = stepper_state
    current_fluid_state = create_fluid_state(current_cv)
    current_wall_kappa, current_wall_temperature = \
        create_wall_derived_compiled(current_wv)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_state = make_fluid_state(current_cv, gas_model)

    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

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
         lazy=args.lazy, restart_filename=restart_file)
