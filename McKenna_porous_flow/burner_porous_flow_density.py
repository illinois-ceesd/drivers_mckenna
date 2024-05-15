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
import gc
import sys
import logging
import numpy as np
import pyopencl as cl
import cantera
from functools import partial
# from dataclasses import dataclass, fields
from warnings import warn

# from arraycontext import (
#     dataclass_array_container, with_container_arithmetic,
#     get_container_context_recursively
# )

from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array

from meshmode.dof_array import DOFArray

from grudge.trace_pair import TracePair  # , inter_volume_trace_pairs
# from grudge.geometry.metrics import normal as normal_vector
from grudge.geometry import normal as normal_vector
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, DISCR_TAG_BASE, VolumeDomainTag
)

from mirgecom.utils import (
    force_evaluation as force_eval,
    normalize_boundaries
)
from mirgecom.simutil import (
    check_step, distribute_mesh, write_visfile,
    check_naninf_local, get_sim_timestep
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    # IsothermalWallBoundary,
    AdiabaticSlipBoundary,
    PrescribedFluidBoundary,
    AdiabaticNoslipWallBoundary,
    LinearizedOutflowBoundary
)
from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient, make_conserved
)
from mirgecom.transport import MixtureAveragedTransport
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import make_fluid_state, make_operator_fluid_states
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
    NeumannDiffusionBoundary,
)
from mirgecom.wall_model import (
    SolidWallModel, SolidWallState, SolidWallConservedVars
)
from mirgecom.wall_model import PorousWallTransport, PorousFlowModel

#########################################################################

density_scaling = 10.0

class _SampleMaskTag:
    pass


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


class _RadiationTag:
    pass


class Burner2D_Reactive:  # noqa

    def __init__(self, *, sigma, sigma_flame, flame_loc, pressure,
                 temperature, speedup_factor, mass_rate_burner, mass_rate_shroud,
                 species_unburn, species_burned, species_shroud, species_atm,
                 porous_domain, material_densities):

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

        self._plug = porous_domain
        self._solid_mass = material_densities

        self._mass_rate_burner = mass_rate_burner
        self._mass_rate_shroud = mass_rate_shroud

    def __call__(self, x_vec, gas_model, flow_rate,
                 state_minus=None, prescribe_species=False, boundary=False):

        actx = x_vec[0].array_context

        _ya = self._ya
        _yb = self._yb
        _ys = self._ys
        _yu = self._yu

        int_diam = 2.38*25.4/2000.0  # radius, actually
        ext_diam = 2.89*25.4/2000.0  # radius, actually

        # ~~~ after combustion products
        upper_bnd = 0.15

        sigma_factor = 12.0 - 11.0*(upper_bnd - x_vec[1])**2/(upper_bnd - 0.10)**2
        _sigma = self._sigma*(
            actx.np.where(actx.np.less(x_vec[1], upper_bnd),
                          actx.np.where(actx.np.greater(x_vec[1], .10),
                                        sigma_factor, 1.0),
                          12.0)
        )

        upper_atm = 0.5*(1.0 + actx.np.tanh(1.0/(12.0*self._sigma)*(x_vec[1] - upper_bnd)))

        # ~~~ shroud
        shroud_1 = 0.5*(1.0 + actx.np.tanh(1.0/(_sigma)*(x_vec[0] - int_diam)))
        shroud_2 = actx.np.where(actx.np.greater(x_vec[0], ext_diam),
                                 0.0, - actx.np.tanh(1.0/(_sigma)*(x_vec[0] - ext_diam)))
        shroud = shroud_1*shroud_2

        # ~~~ flame ignition
        core = 0.5*(1.0 - actx.np.tanh(1.0/_sigma*(x_vec[0] - int_diam)))

        # ~~~ atmosphere
        atmosphere = 1.0 - (shroud + core)

        # ~~~ flame ignition
        flame = actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - 0.1))

        # ~~~ species
        if prescribe_species:
            yf = (flame*_yb + (1.0-flame)*_yu)*(1.0-upper_atm) + _ya*upper_atm
            ys = _ya*upper_atm + (1.0 - upper_atm)*_ys
            y = atmosphere*_ya + shroud*ys + core*yf
            # y = (atmosphere*_ya + shroud*ys + core*yf)*(1.0 - self._plug) + self._plug*_yb
        else:
            y = state_minus.cv.species_mass_fractions

        # ~~~ temperature and EOS

        upper_bnd = 0.11

        sigma_factor = 10.0 - 9.0*(upper_bnd - x_vec[1])**2/(upper_bnd - 0.10)**2
        _sigma = self._sigma*(
            actx.np.where(actx.np.less(x_vec[1], upper_bnd),
                          actx.np.where(actx.np.greater(x_vec[1], .10),
                                        sigma_factor, 1.0),
                          10.0)
        )
        upper_atm = 0.5*(1.0 + actx.np.tanh(1.0/(2.5*self._sigma)*(x_vec[1] - upper_bnd)))

        temp = (flame*self._temp + (1.0-flame)*300.0)*(1.0-upper_atm) + 300.0*upper_atm
        temperature = temp*core + 300.0*(1.0 - core)

        if state_minus is None:
            pressure = self._pres + 0.0*x_vec[0]
            mass = gas_model.eos.get_density(pressure, temperature,
                                             species_mass_fractions=y)
        else:
            mass = state_minus.cv.mass

        if boundary is False:
            mass = mass*((1.0 - self._plug)*0.12 + 0.88)

        # ~~~ velocity and/or momentum
        mom_burner = self._mass_rate_burner*self._speedup_factor
        mom_shroud = self._mass_rate_shroud*self._speedup_factor
        smoothY = mom_burner*(1.0-upper_atm) + 0.0*upper_atm  # noqa
        mom_x = 0.0*x_vec[0]
        mom_y = core*smoothY + shroud*mom_shroud*(1.0-upper_atm)
        momentum = make_obj_array([mom_x, mom_y])
        velocity = momentum/mass

        # ~~~
        specmass = mass * y

        # ~~~
        # FIXME prescribe temperature or use internal value?
        internal_energy = gas_model.eos.get_internal_energy(temperature, species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        solid_energy = mass*0.0
        if boundary is False:
            material_dens = self._plug * self._solid_mass + x_vec[0]*0.0
            eps_rho_solid = gas_model.solid_density(material_dens)
            tau = gas_model.decomposition_progress(material_dens)
            solid_energy = eps_rho_solid * gas_model.wall_eos.enthalpy(temperature, tau)
        energy = mass * (internal_energy + kinetic_energy) + solid_energy

        return make_conserved(dim=2, mass=mass, energy=energy,
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

        return actx.np.maximum(sponge_x, sponge_y)


class PorousMaterial:

    def __init__(self, x_min, y_min, thickness):
        self._x_min = x_min
        self._y_min = y_min
        self._thickness = thickness

    def __call__(self, x_vec):
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context

        sponge_x = xpos*0.0
        sponge_y = xpos*0.0

        y0 = self._y_min
        dy = actx.np.where(
            actx.np.less(ypos, y0),
            0.5, actx.np.where(actx.np.greater(ypos, y0+self._thickness*0.5),
                               1.0, (ypos-(y0-self._thickness*0.5))/self._thickness)
            )
        sponge_y = actx.np.absolute(-1. + 2.*(-20.*dy**7 + 70.*dy**6
                                              - 84.*dy**5 + 35.*dy**4))
        sponge_y = actx.np.where(actx.np.greater(ypos, 0.14405+0.01), 0.0, sponge_y)

        x0 = self._x_min
        dx = actx.np.where(
            actx.np.less(xpos, x0-self._thickness*0.5),
            0.0, actx.np.where(actx.np.greater(xpos, x0),
                               0.5, (xpos-(x0-self._thickness*0.5))/self._thickness)
            )
        sponge_x = 1. - actx.np.absolute(2.*(-20.*dx**7 + 70.*dx**6
                                             - 84.*dx**5 + 35.*dx**4))

        return sponge_x*sponge_y

#        radius = actx.np.sqrt((xpos-(x0-self._thickness*0.5))**2 + (ypos-(y0 + self._thickness*0.5))**2)
#        sponge = actx.np.where(actx.np.less(radius, self._thickness*0.5), 0.5*radius/(self._thickness*0.5), 0.5)
#        circle = 1.0 - 2.0*(-20.0*sponge**7 + 70*sponge**6 - 84*sponge**5 + 35*sponge**4)

#        weight = \
#            actx.np.where(actx.np.less(xpos, x0 - self._thickness*0.5),
#                sponge_x*sponge_y,
#                actx.np.where(actx.np.greater(ypos, y0 + self._thickness*0.5), sponge_x*sponge_y, circle))

#        return weight


from mirgecom.materials.carbon_fiber import FiberEOS as OriginalFiberEOS  # noqa E402
class FiberEOS(OriginalFiberEOS):  # noqa E302
    """Inherits and modified the original carbon fiber."""

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the carbon fiber."""
        virgin = 2.0e-11*density_scaling
        char = 1e+9
        return virgin*tau + char*(1.0 - tau)


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
         restart_file=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=use_tpe,
                           use_einsum_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~
    my_material = "fiber"

    width = 0.015

    flame_grid_spacing = 100

    # ~~~~~~~~~~~~~~~~~~
    # default i/o frequencies
    nviz = 25000
    nrestart = 50000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "ssprk43"
    t_final = 2.0
    niter = 4000001
    local_dt = True
    constant_cfl = True
    current_cfl = 0.4

    # discretization and model control
    order = 4

    chem_rate = 1.0
    speedup_factor = 1.0

    equiv_ratio = 0.7
    total_flow_rate = 17.0
    # air_flow_rate = 18.8
    # shroud_rate = 11.85
    prescribe_species = True

    width_mm = str("%02i" % (width*1000)) + "mm"
    flame_grid_um = str("%03i" % flame_grid_spacing) + "um"

    current_dt = 1.0e-8
    wall_time_scale = speedup_factor  # wall speed-up
    mechanism_file = "uiuc_7sp"
    solid_domains = ["wall_alumina", "wall_graphite"]

    if use_tpe:
        mesh_filename = \
            f"mesh_v3_{width_mm}_{flame_grid_um}_porous_coarse_quads-v2.msh"
    else:
        mesh_filename = \
            f"mesh_v3_{width_mm}_{flame_grid_um}_porous_coarse"

    wall_penalty_amount = 1.0
    use_radiation = False

    restart_iterations = False
    # restart_iterations = True

    # Average from https://www.azom.com/properties.aspx?ArticleID=52 for alumina
    wall_alumina_rho = 3500.0
    wall_alumina_cp = 700.0
    wall_alumina_kappa = 25.00

    # Average from https://www.azom.com/article.aspx?ArticleID=1630 for graphite
    # TODO There is a table with the temperature-dependent data for graphite
    wall_graphite_rho = 1625.0
    wall_graphite_cp = 770.0
    wall_graphite_kappa = 200.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import time
    t_start = time.time()
    t_shutdown = 720*60

    x0_sponge = 0.150
    sponge_amp = 400.0
    theta_factor = 0.02

# ############################################################################

    dim = 2

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:09d}-{rank:04d}.pkl"

    if integrator == "ssprk43":
        from mirgecom.integrators.ssprk import ssprk43_step
        timestepper = ssprk43_step
        force_eval_stepper = True
    else:
        sys.exit()

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

        if use_tpe:
            mesh2_path = mesh_filename
        else:
            if rank == 0:
                local_path = os.path.dirname(os.path.abspath(__file__)) + "/"
                geo_path = local_path + mesh_filename + ".geo"
                mesh2_path = local_path + mesh_filename + "-v2.msh"
                mesh1_path = local_path + mesh_filename + "-v1.msh"

                os.system(f"rm -rf {mesh1_path} {mesh2_path}")
                os.system(f"gmsh {geo_path} -2 -o {mesh1_path}")
                os.system(f"gmsh {mesh1_path} -save -format msh2 -o {mesh2_path}")

                os.system(f"rm -rf {mesh1_path}")

        print(f"Reading mesh from {mesh2_path}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh2_path, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {"fluid": ["fluid", "sample"],
                              "solid": solid_domains}
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
        + volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["solid"][0].nelements)

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order,
        tensor_product_elements=use_tpe)

    quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("solid"), DISCR_TAG_BASE)

    from mirgecom.utils import mask_from_elements
    # wall_vol_discr = dcoll.discr_from_dd(dd_vol_solid)
    wall_tag_to_elements = volume_to_local_mesh_data["solid"][1]
    wall_alumina_mask = mask_from_elements(
        dcoll, dd_vol_solid, actx, wall_tag_to_elements["wall_alumina"])
    wall_graphite_mask = mask_from_elements(
        dcoll, dd_vol_solid, actx, wall_tag_to_elements["wall_graphite"])

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))

    fluid_zeros = force_eval(actx, fluid_nodes[0]*0.0)
    solid_zeros = force_eval(actx, solid_nodes[0]*0.0)

    # ~~~~~~~~~~
    # from grudge.dt_utils import characteristic_lengthscales
    # char_length_fluid = force_eval(
    #     actx, characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid))
    # char_length_solid = force_eval(
    #     actx, characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid))

    # ~~~~~~~~~~
    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

##########################################################################

    # {{{ Set up initial state using Cantera

    # Use Cantera for initialization
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    ct_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = ct_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.

    air = "O2:0.21,N2:0.79"
    fuel = "C2H4:1"
    ct_soln.set_equivalence_ratio(phi=equiv_ratio,
                                       fuel=fuel, oxidizer=air)

    temp_unburned = 300.0
    pres_unburned = 101325.0/density_scaling
    ct_soln.TP = temp_unburned, pres_unburned
    x_reference = ct_soln.X
    y_reference = ct_soln.Y
    rho_int = ct_soln.density

    r_int = 2.38*25.4/2000.0
    r_ext = 2.89*25.4/2000.0

    try:
        total_flow_rate
    except NameError:
        total_flow_rate = None

    try:
        air_flow_rate
    except NameError:
        air_flow_rate = None

    if total_flow_rate is not None:
        flow_rate = total_flow_rate*1.0
    else:
        idx_fuel = ct_soln.species_index("C2H4")
        flow_rate = air_flow_rate/(sum(x_reference) - x_reference[idx_fuel])

    A_int = np.pi*r_int**2  # noqa
    lmin_to_m3s = 1.66667e-5
    u_int = flow_rate*lmin_to_m3s/A_int*density_scaling
    rhoU_int = rho_int*u_int  # noqa

    # mass_shroud = shroud_rate*1.0
    A_ext = np.pi*(r_ext**2 - r_int**2)  # noqa
    u_ext = u_int  # mass_shroud*lmin_to_m3s/A_ext
    rho_ext = pres_unburned/((cantera.gas_constant/ct_soln.molecular_weights[-1])*300.)
    # mdot_ext = rho_ext*u_ext*A_ext
    rhoU_ext = rho_ext*u_ext  # noqa

    print("width=", width_mm, "(mm)")
    print("V_dot=", flow_rate, "(L/min)")
    print(f"{A_int= }", "(m^2)")
    print(f"{A_ext= }", "(m^2)")
    print(f"{u_int= }", "(m/s)")
    print(f"{u_ext= }", "(m/s)")
    print(f"{rho_int= }", "(kg/m^3)")
    print(f"{rho_ext= }", "(kg/m^3)")
    print(f"{rhoU_int= }")
    print(f"{rhoU_ext= }")
    print("ratio=", u_ext/u_int, "\n")

    # Set Cantera internal gas temperature, pressure, and mole fractios
    ct_soln.TPX = temp_unburned, pres_unburned, x_reference

    if prescribe_species:

        ct_soln.TPX = temp_unburned, 101325.0, x_reference

        # Pull temperature, density, mass fractions, and pressure from Cantera
        # set the mass flow rate at the inlet
        sim = cantera.ImpingingJet(gas=ct_soln, width=width)
        sim.inlet.mdot = rhoU_int
        sim.set_refine_criteria(ratio=2, slope=0.1, curve=0.1, prune=0.0)
        sim.set_initial_guess(products="equil")
        sim.solve(loglevel=0, refine_grid=True, auto=True)

        # ~~~ Reactants
        assert np.absolute(sim.density[0]*sim.velocity[0] - rhoU_int) < 1e-11
        rho_unburned = sim.density[0]/density_scaling
        y_unburned = sim.Y[:, 0]

        # ~~~ Products
        index_burned = np.argmax(sim.T)
        temp_burned = sim.T[index_burned]
        y_burned = sim.Y[:, index_burned]
        rho_burned = sim.density[index_burned]/density_scaling

    else:
        # ~~~ Reactants
        y_unburned = y_reference*1.0
        _, rho_unburned, y_unburned = ct_soln.TDY

        # ~~~ Products

        ct_soln.TPY = 2400.0, pres_unburned, y_unburned
        ct_soln.equilibrate("TP")
        temp_burned, rho_burned, y_burned = ct_soln.TDY

    idx_CO2 = ct_soln.species_index("CO2")
    idx_O2 = ct_soln.species_index("O2")
    idx_N2 = ct_soln.species_index("N2")

    # ~~~ Atmosphere
    x = np.zeros(nspecies)
    x[idx_O2] = 0.21
    x[idx_N2] = 0.79
    ct_soln.TPX = temp_unburned, pres_unburned, x

    y_atmosphere = np.zeros(nspecies)
    dummy, rho_atmosphere, y_atmosphere = ct_soln.TDY
    temp_atmosphere, rho_atmosphere, y_atmosphere = ct_soln.TDY

    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_shroud = y_atmosphere*0.0
    y_shroud[idx_N2] = 1.0

    ct_soln.TPY = 300.0, pres_unburned, y_shroud
    temp_shroud, rho_shroud = ct_soln.TD

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
        ct_soln, temperature_niter=3)(actx.np)

    temperature_seed = 1234.56789
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    base_transport = MixtureAveragedTransport(pyrometheus_mechanism,
                                              lewis=np.ones(nspecies,),
                                              factor=speedup_factor)

    transport_model = PorousWallTransport(base_transport=base_transport)

    # {{{

    if my_material == "fiber":

        fiber_density = 168.0/1000.0
        material_densities = fiber_density + fluid_zeros

        import mirgecom.materials.carbon_fiber as material_sample
        material = FiberEOS(
            dim=dim, char_mass=0.0, virgin_mass=fiber_density,
            anisotropic_direction=1, timescale=speedup_factor)
        decomposition = material_sample.Y3_Oxidation_Model(
            wall_material=material, arrhenius=speedup_factor*1e5,
            activation_energy=-120000.0)

    plug_region = PorousMaterial(x_min=0.015875, y_min=0.115, thickness=0.005)

    plug = force_eval(actx, plug_region(x_vec=fluid_nodes))

    # }}}

    gas_model = PorousFlowModel(eos=eos, transport=transport_model,
                                wall_eos=material)

    # {{{

    species_names = pyrometheus_mechanism.species_names
    print(f"Pyrometheus mechanism species names {species_names}\n")
    print("Unburned:")
    print(f"T = {temp_unburned}")
    print(f"D = {rho_unburned}")
    print(f"Y = {y_unburned}\n")
    print("Burned:")
    print(f"T = {temp_burned}")
    print(f"D = {rho_burned}")
    print(f"Y = {y_burned}\n")
    print("Atmosphere:")
    print(f"T = {temp_atmosphere}")
    print(f"D = {rho_atmosphere}")
    print(f"Y = {y_atmosphere}\n")
    print("Shroud:")
    print(f"T = {temp_shroud}")
    print(f"D = {rho_shroud}")
    print(f"Y = {y_shroud}\n")

    # }}}

    # {{{ Initialize wall model

    from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
        add_interface_boundaries,
        add_interface_boundaries_no_grad
    )

    wall_densities = (
        + wall_alumina_rho * wall_alumina_mask
        + wall_graphite_rho * wall_graphite_mask)

    def _get_solid_densities():
        return wall_densities

    def _get_solid_enthalpy(temperature):
        wall_alumina_h = wall_alumina_cp * temperature
        wall_graphite_h = wall_graphite_cp * temperature
        return (wall_alumina_h * wall_alumina_mask
                + wall_graphite_h * wall_graphite_mask)

    def _get_solid_heat_capacity(temperature):
        return (wall_alumina_cp * wall_alumina_mask
                + wall_graphite_cp * wall_graphite_mask)

    def _get_solid_thermal_conductivity(temperature):
        return (wall_alumina_kappa * wall_alumina_mask
                + wall_graphite_kappa * wall_graphite_mask)

    def _get_solid_decomposition_progress(mass):
        return (1.0 * wall_alumina_mask
                + 1.0 * wall_graphite_mask)

    def _get_solid_emissivity(temperature=None, tau=None):
        return speedup_factor*(
            0.00 * wall_alumina_mask
            + 0.85 * wall_graphite_mask)

    wall_emissivity = _get_solid_emissivity()

    solid_wall_model = SolidWallModel(
        density_func=_get_solid_densities,
        enthalpy_func=_get_solid_enthalpy,
        heat_capacity_func=_get_solid_heat_capacity,
        thermal_conductivity_func=_get_solid_thermal_conductivity,
        # decomposition_func=_get_solid_decomposition_progress
    )

    # }}}

#############################################################################

    def reaction_damping(dcoll, nodes, **kwargs):
        """Region where chemistry is frozen."""
        ypos = nodes[1]

        y_max = 0.25
        y_thickness = 0.10

        y0 = (y_max - y_thickness)
        dy = +((ypos - y0)/y_thickness)
        return actx.np.where(
            actx.np.greater(ypos, y0),
            actx.np.where(actx.np.greater(ypos, y_max),
                          0.0, 1.0 - (3.0*dy**2 - 2.0*dy**3)),
            1.0)

#############################################################################

    def smoothness_region(dcoll, nodes):
        """Region where numerical viscosity is added for smooth outflow."""
        ypos = nodes[1]

        y_max = 0.55
        y_thickness = 0.20

        y0 = (y_max - y_thickness)
        dy = +((ypos - y0)/y_thickness)
        return actx.np.where(
            actx.np.greater(ypos, y0),
            actx.np.where(actx.np.greater(ypos, y_max),
                          1.0, 3.0*dy**2 - 2.0*dy**3),
            0.0)

##############################################################################

    from mirgecom.limiter import bound_preserving_limiter

    from grudge.discretization import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_MODAL
    from meshmode.transform_metadata import FirstAxisIsElementsTag

    def _limit_fluid_cv(cv, wv, temperature_seed, gas_model, dd=None):
        temperature = gas_model.get_temperature(cv=cv, wv=wv, tseed=temperature_seed)
        pressure = gas_model.get_pressure(cv, wv, temperature)

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
        mass_lim = wv.void_fraction*eos.get_density(pressure=pressure,
                                                    temperature=temperature,
                                                    species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        ) + wv.density*gas_model.wall_eos.enthalpy(temperature, wv.tau)

        # make a new CV with the limited variables
        lim_cv = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                                momentum=mass_lim*cv.velocity,
                                species_mass=mass_lim*spec_lim)

        return make_obj_array([lim_cv, pressure, temperature])

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

        # cancel the ``high-order"" polynomials p > 0 and keep the average
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
        return cv

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

    fluid_init = Burner2D_Reactive(sigma=0.00020,
        sigma_flame=0.000200, temperature=temp_burned, pressure=pres_unburned,
        flame_loc=0.10025, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned,
        porous_domain=plug, material_densities=material_densities)

    ref_state = Burner2D_Reactive(sigma=0.00020,
        sigma_flame=0.00001, temperature=temp_burned, pressure=pres_unburned,
        flame_loc=0.1050, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned,
        porous_domain=plug, material_densities=material_densities)

###############################################################################

    smooth_region = force_eval(actx, smoothness_region(dcoll, fluid_nodes))

    reaction_rates_damping = force_eval(actx, reaction_damping(dcoll, fluid_nodes))

    def _get_fluid_state(cv, sample_densities, temp_seed):
        return make_fluid_state(
            cv=cv, gas_model=gas_model, temperature_seed=temp_seed,
            material_densities=sample_densities,
            limiter_func=_limit_fluid_cv, limiter_dd=dd_vol_fluid,
        )

    get_fluid_state = actx.compile(_get_fluid_state)

    def _get_solid_state(cv):
        wdv = solid_wall_model.dependent_vars(cv)
        return SolidWallState(cv=cv, dv=wdv)

    get_solid_state = actx.compile(_get_solid_state)

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")

        # ~~~ FLUID
        fluid_cv = fluid_init(
            x_vec=fluid_nodes, gas_model=gas_model, flow_rate=flow_rate,
            prescribe_species=prescribe_species)
        tseed = force_eval(actx, 300.0 + fluid_zeros)

        # ~~~ SAMPLE
        sample_densities = plug*material_densities
        del material_densities

        # ~~~ HOLDER
        wall_mass = solid_wall_model.solid_density(wall_densities)

        wall_alumina_h = wall_alumina_cp*300.0 + solid_zeros
        wall_graphite_h = wall_graphite_cp*300.0 + solid_zeros
        wall_enthalpy = (wall_alumina_h * wall_alumina_mask
                         + wall_graphite_h * wall_graphite_mask)

        wall_energy = wall_mass * wall_enthalpy

        solid_cv = SolidWallConservedVars(mass=wall_densities,
                                          energy=wall_energy)

        last_stored_step = -1.0
        my_file = open("temperature_file.dat", "w")
        my_file.close()

    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if np.isscalar(current_t) is False:
            current_t = np.min(actx.to_numpy(current_t[2]))

        if restart_iterations:
            current_t = 0.0
            current_step = 0

        last_stored_step = -1.0  # sometimes the file only has 1 line...
        if os.path.isfile("temperature_file.dat"):
            data = np.genfromtxt("temperature_file.dat", delimiter=",")
            if data.shape == 2:
                last_stored_step = data[-1, 1]

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
                actx, dcoll.discr_from_dd(dd_vol_fluid),
                restart_dcoll.discr_from_dd(dd_vol_fluid))
            solid_connection = make_same_mesh_connection(
                actx, dcoll.discr_from_dd(dd_vol_solid),
                restart_dcoll.discr_from_dd(dd_vol_solid))
            fluid_cv = fluid_connection(restart_data["cv"])
            tseed = fluid_connection(restart_data["temperature_seed"])
            solid_cv = solid_connection(restart_data["wall_cv"])
            sample_densities = solid_connection(restart_data["sample_densities"])
        else:
            fluid_cv = restart_data["cv"]
            tseed = restart_data["temperature_seed"]
            solid_cv = restart_data["wall_cv"]
            sample_densities = restart_data["sample_densities"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

#########################################################################

    tseed = force_eval(actx, tseed)
    fluid_cv = force_eval(actx, fluid_cv)
    sample_densities = force_eval(actx, sample_densities)
    fluid_state = get_fluid_state(fluid_cv, sample_densities, tseed)

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

    # initialize the sponge field
    sponge_x_thickness = 0.055
    sponge_y_thickness = 0.055

    xMaxLoc = x0_sponge + sponge_x_thickness
    yMinLoc = 0.04

    sponge_init = InitSponge(amplitude=sponge_amp,
                             x_max=xMaxLoc, y_min=yMinLoc,
                             x_thickness=sponge_x_thickness,
                             y_thickness=sponge_y_thickness)

    sponge_sigma = force_eval(actx, sponge_init(x_vec=fluid_nodes))

    ref_cv = ref_state(fluid_nodes, gas_model, flow_rate,
                       state_minus=fluid_state,
                       prescribe_species=prescribe_species)

    ref_cv = force_eval(actx, ref_cv)

##############################################################################

    inflow_nodes = force_eval(actx, dcoll.nodes(dd_vol_fluid.trace("inlet")))
    inflow_temperature = inflow_nodes[0]*0.0 + 300.0

    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return inflow_temperature

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_cv_cond = ref_state(
            x_vec=inflow_nodes, gas_model=gas_model,
            flow_rate=flow_rate, state_minus=state_minus,
            prescribe_species=prescribe_species, boundary=True)
        return make_fluid_state(
            cv=inflow_cv_cond, gas_model=gas_model,
            material_densities=state_minus.wv.material_densities,
            temperature_seed=300.0)

    from mirgecom.inviscid import inviscid_flux
    from mirgecom.flux import num_flux_central
    from mirgecom.viscous import viscous_flux
    from mirgecom.flux import num_flux_lfr

    """ """
    class MyPrescribedBoundary(PrescribedFluidBoundary):
        r"""Prescribed my boundary function."""

        def __init__(self, bnd_state_func, temperature_func):
            """Initialize the boundary condition object."""
            self.bnd_state_func = bnd_state_func
            PrescribedFluidBoundary.__init__(
                self,
                boundary_state_func=bnd_state_func,
                inviscid_flux_func=self.inviscid_wall_flux,
                viscous_flux_func=self.viscous_wall_flux,
                boundary_temperature_func=temperature_func,
                boundary_gradient_cv_func=self.grad_cv_bc)

        def prescribed_state_for_advection(
                self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            state_plus = self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                             state_minus, **kwargs)

            mom_x = -state_minus.cv.momentum[0]
            mom_y = 2.0*state_plus.cv.momentum[1] - state_minus.cv.momentum[1]
            mom_plus = make_obj_array([mom_x, mom_y])

            kin_energy_ref = 0.5*np.dot(state_plus.cv.momentum, state_plus.cv.momentum)/state_plus.cv.mass
            kin_energy_mod = 0.5*np.dot(mom_plus, mom_plus)/state_plus.cv.mass
            energy_plus = state_plus.cv.energy - kin_energy_ref + kin_energy_mod
            # no need for solid energy at the boundary

            cv = make_conserved(dim=dim, mass=state_plus.cv.mass,
                                energy=energy_plus, momentum=mom_plus,
                                species_mass=state_plus.cv.species_mass)

            return make_fluid_state(
                cv=cv, gas_model=gas_model, temperature_seed=300.0,
                material_densities=state_minus.wv.material_densities)

        def prescribed_state_for_diffusion(
                self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            return self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                       state_minus, **kwargs)

        def inviscid_wall_flux(
                self, dcoll, dd_bdry, gas_model, state_minus,
                numerical_flux_func, **kwargs):

            state_plus = self.prescribed_state_for_advection(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, **kwargs)

            state_pair = TracePair(dd_bdry, interior=state_minus,
                                   exterior=state_plus)

            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            # FIXME use centered scheme?
            actx = state_pair.int.array_context
            lam = actx.np.maximum(state_pair.int.wavespeed,
                                  state_pair.ext.wavespeed)
            return num_flux_lfr(
                f_minus_normal=inviscid_flux(state_pair.int, gas_model)@normal,
                f_plus_normal=inviscid_flux(state_pair.ext, gas_model)@normal,
                q_minus=state_pair.int.cv,
                q_plus=state_pair.ext.cv, lam=lam)

        def grad_cv_bc(
                self, state_plus, state_minus, grad_cv_minus, normal, **kwargs):
            """Return grad(CV) for boundary calculation of viscous flux."""
            if prescribe_species:
                return grad_cv_minus
            else:
                normal_velocity = state_minus.cv.velocity@normal

                inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos,
                                           flow_rate=flow_rate,
                                           prescribe_species=True)
                y_reference = inflow_cv_cond.species_mass_fractions

                grad_y_bc = 0.*grad_cv_minus.species_mass
                grad_species_mass_bc = 0.*grad_cv_minus.species_mass
                for i in range(nspecies):
                    delta_y = state_minus.cv.species_mass_fractions[i] - y_reference[i]
                    dij = state_minus.tv.species_diffusivity[i]
                    grad_y_bc[i] = + (normal_velocity*delta_y/dij) * normal
                    grad_species_mass_bc[i] = (
                        state_minus.mass_density*grad_y_bc[i]
                        + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

                return grad_cv_minus.replace(
                    species_mass=grad_species_mass_bc)

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
        free_stream_density=rho_atmosphere, free_stream_pressure=pres_unburned,
        free_stream_velocity=np.zeros(shape=(dim,)),
        free_stream_species_mass_fractions=y_atmosphere)

    my_presc_bnd = MyPrescribedBoundary(bnd_state_func=inlet_bnd_state_func,
                                        temperature_func=bnd_temperature_func)

    fluid_boundaries = {
        dd_vol_fluid.trace("inlet").domain_tag: my_presc_bnd,
        dd_vol_fluid.trace("symmetry").domain_tag: AdiabaticSlipBoundary(),
        dd_vol_fluid.trace("burner").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("linear").domain_tag: linear_bnd}

    wall_symmetry = NeumannDiffusionBoundary(0.0)
    solid_boundaries = {
        dd_vol_solid.trace("wall_sym").domain_tag: wall_symmetry}

##############################################################################

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

        fluid_viz_fields = [
            ("CV_rho", fluid_state.cv.mass),
            ("CV_rhoU", fluid_state.cv.momentum),
            ("CV_rhoE", fluid_state.cv.energy),
            ("DV_P", fluid_state.pressure),
            ("DV_T", fluid_state.temperature),
            ("DV_U", fluid_state.velocity[0]),
            ("DV_V", fluid_state.velocity[1]),
            ("WV", fluid_state.wv),
            ("dt", dt[0] if local_dt else None),
            ("plug", plug),
            ("sponge", sponge_sigma),
            ("reactions", reaction_rates_damping),
            ("smoothness", 1.0 - theta_factor*smoothness),
            # ("grad_epsilon", grad_epsilon),
            # ("f_phi", f_phi),
        ]

        # species mass fractions
        fluid_viz_fields.extend((
            "Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        wall_cv = solid_state.cv
        wdv = solid_state.dv
        solid_viz_fields = [
            ("wv_energy", wall_cv.energy),
            ("cfl", solid_zeros),  # FIXME
            ("wall_kappa", wdv.thermal_conductivity),
            # ("wall_progress", wdv.tau),
            ("wall_temperature", wdv.temperature),
            ("wall_grad_t", grad_t_solid),
            ("dt", dt[3] if local_dt else None),
        ]

        solid_viz_fields.append(("wv_mass", wall_cv.mass))

        if rank == 0:
            logger.info("Writing solution file...")
        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t,
            overwrite=True, comm=comm)
        write_visfile(
            dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        if rank == 0:
            logger.info("Writing restart file...")

        cv, tseed, sample_densities, wall_cv = state
        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_file:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "sample_densities": sample_densities,
                "nspecies": nspecies,
                "wall_cv": wall_cv,
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
        pressure = force_eval(actx, dv.pressure)
        temperature = force_eval(actx, dv.temperature)

        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_naninf_local(dcoll, "vol", temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

##############################################################################

    from arraycontext import outer
    from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
    from grudge import op
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
                    field_bounds.items()))))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    off_axis_x = 1e-7
    fluid_nodes_are_off_axis = actx.np.greater(fluid_nodes[0], off_axis_x)
    solid_nodes_are_off_axis = actx.np.greater(solid_nodes[0], off_axis_x)

    def axisym_source_fluid(actx, dcoll, state, grad_cv, grad_t):
        cv = state.cv
        dv = state.dv
        wv = state.wv

        mu = state.tv.viscosity
        beta = transport_model.volume_viscosity(cv, dv, wv, gas_model.eos)
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

        # d2udr2 = my_derivative_function(
        #    actx, dcoll, dudr, fluid_boundaries,
        #    dd_vol_fluid, "replicate", _MyGradTag_1)[0]  # XXX
        d2vdr2 = my_derivative_function(
            actx, dcoll, dvdr, fluid_boundaries,
            dd_vol_fluid, "replicate", _MyGradTag_2)[0]  # XXX
        d2udrdy = my_derivative_function(
            actx, dcoll, dudy, fluid_boundaries,
            dd_vol_fluid, "replicate", _MyGradTag_3)[0]  # XXX

        dmudr = my_derivative_function(actx, dcoll, mu, fluid_boundaries,
                                       dd_vol_fluid, "replicate", _MyGradTag_4)[0]
        dbetadr = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                         dd_vol_fluid, "replicate", _MyGradTag_5)[0]
        dbetady = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                         dd_vol_fluid, "replicate", _MyGradTag_6)[1]

        qr = - (kappa*grad_t)[0]  # FIXME add species enthalpy term
        dqrdr = 0.0  # - (dkappadr*grad_t[0] + kappa*d2Tdr2)  #XXX

        dyidr = grad_y[:, 0]
        # dyi2dr2 = \
        #     my_derivative_function(actx, dcoll, dyidr, "replicate")[:,0]  # XXX

        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        # tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = (
            beta*(dudr + dvdy) + 2.0*mu*actx.np.where(fluid_nodes_are_off_axis,
                                                      u/fluid_nodes[0], dudr))

        dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

        """
        """
        source_mass_dom = - cv.momentum[0]

        source_rhoU_dom = (  # noqa N806
            - cv.momentum[0]*u
            + tau_rr - tau_tt
            + u*dbetadr + beta*dudr
            + beta*actx.np.where(fluid_nodes_are_off_axis,
                                 -u/fluid_nodes[0], -dudr))

        source_rhoV_dom = (  # noqa N806
            - cv.momentum[0]*v
            + tau_ry
            + u*dbetady + beta*dudy)

        source_rhoE_dom = (  # noqa N806
            - ((cv.energy+dv.pressure)*u + qr)
            + u*tau_rr + v*tau_ry
            + u**2*dbetadr + beta*2.0*u*dudr
            + u*v*dbetady + u*beta*dvdy + v*beta*dudy)

        source_spec_dom = - cv.species_mass*u + d_ij*dyidr
        """
        """

        source_mass_sng = - drhoudr
        source_rhoU_sng = 0.0  # noqa N806  # mu*d2udr2 + 0.5*beta*d2udr2  #XXX
        source_rhoV_sng = (  # noqa N806
            - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady)
        source_rhoE_sng = ( # noqa N806
            -((cv.energy+dv.pressure)*dudr + dqrdr)
            + tau_rr*dudr + v*dtaurydr
            + 2.0*beta*dudr**2
            + beta*dudr*dvdy
            + v*dudr*dbetady
            + v*beta*d2udrdy)
        source_spec_sng = - cv.species_mass*dudr  # + d_ij*dyi2dr2

        """
        """
        source_mass = (
            actx.np.where(fluid_nodes_are_off_axis,
                          source_mass_dom/fluid_nodes[0], source_mass_sng))
        source_rhoU = (  # noqa N806
            actx.np.where(fluid_nodes_are_off_axis,
                          source_rhoU_dom/fluid_nodes[0], source_rhoU_sng))
        source_rhoV = (  # noqa N806
            actx.np.where(fluid_nodes_are_off_axis,
                          source_rhoV_dom/fluid_nodes[0], source_rhoV_sng))
        source_rhoE = (  # noqa N806
            actx.np.where(fluid_nodes_are_off_axis,
                          source_rhoE_dom/fluid_nodes[0], source_rhoE_sng))
        source_spec = make_obj_array([
            actx.np.where(fluid_nodes_are_off_axis,
                          source_spec_dom[i]/fluid_nodes[0], source_spec_sng[i])
            for i in range(nspecies)])

        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                       momentum=make_obj_array([source_rhoU, source_rhoV]),
                       species_mass=source_spec)

    # ~~~~~~~
    def axisym_source_solid(actx, dcoll, solid_state, grad_t):

        # temperature = solid_state.dv.temperature

        kappa = solid_state.dv.thermal_conductivity
        # dkappadr = 0.0*solid_nodes[0]

        qr = - (kappa*grad_t)[0]
        # d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0],
        #                                  axisym_wall_boundaries,
        #                                  dd_vol_solid, "symmetry")[0]
        # dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)

        source_mass = solid_state.cv.mass*0.0

        source_rhoE_dom = - qr  # noqa N806
        source_rhoE_sng = 0.0  # noqa N806  #- dqrdr
        source_rhoE = actx.np.where(solid_nodes_are_off_axis,  # noqa N806
                                    source_rhoE_dom/solid_nodes[0], source_rhoE_sng)

        return SolidWallConservedVars(mass=source_mass, energy=source_rhoE)

    # compiled_axisym_source_solid = actx.compile(axisym_source_solid)

    # ~~~~~~~
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
        """Chemistry."""
        return density_scaling*chem_rate*speedup_factor*reaction_rates_damping*(
            eos.get_species_source_terms(cv, temperature))

    # ~~~~~~~
    def darcy_source_terms(state):
        """Source term to mimic Darcy's law."""
        momentum = -1.0 * state.tv.viscosity/state.wv.permeability * (
            state.wv.void_fraction * state.cv.velocity)
        energy = -1.0 * state.tv.viscosity/state.wv.permeability * (
            state.wv.void_fraction**2 * np.dot(state.cv.velocity, state.cv.velocity))
        return make_conserved(
            dim=dim,
            mass=fluid_zeros,
            energy=energy,
            momentum=momentum,
            species_mass=state.cv.species_mass*0.0)

    # ~~~~~~~
    def radiation_sink_terms(boundaries, temperature, epsilon):
        """Radiation sink term"""
        radiation_boundaries = normalize_boundaries(boundaries)
        grad_epsilon = my_derivative_function(
            actx, dcoll, epsilon, radiation_boundaries, dd_vol_fluid,
            "replicate", _RadiationTag)
        epsilon_0 = 1.0
        f_phi = actx.np.sqrt(grad_epsilon[0]**2 + grad_epsilon[1]**2)

        # this already includes wall/fluid speed-up
        emissivity = material.emissivity(temperature, tau=None)

        return - emissivity*5.67e-8*(1.0/epsilon_0*f_phi)*(temperature**4 - 300.0**4)

    # ~~~~~~~
    def oxidation_source_terms(state):
        """Oxidation source terms."""
        sample_mass_rhs, m_dot_o2, m_dot_co2 = \
            decomposition.get_source_terms(
                temperature=state.temperature, tau=state.wv.tau,
                rhoY_o2=density_scaling*state.cv.species_mass[idx_O2])

        species_sources = state.cv.species_mass*0.0
        species_sources[idx_CO2] = m_dot_co2
        species_sources[idx_O2] = m_dot_o2

        cv = make_conserved(dim=dim, mass=m_dot_o2+m_dot_co2,
                            energy=fluid_zeros, momentum=state.cv.momentum*0.0,
                            species_mass=species_sources)

        return cv, sample_mass_rhs

##############################################################################

    from grudge.op import nodal_min, nodal_max

    def vol_min(dd_vol, x):
        return actx.to_numpy(nodal_min(dcoll, dd_vol, x, initial=+np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(nodal_max(dcoll, dd_vol, x, initial=-np.inf))[()]

    def _my_get_timestep_wall(solid_state, t, dt):
        return current_dt + solid_zeros
        # wall_diffusivity = solid_wall_model.thermal_diffusivity(solid_state)
        # return current_cfl*char_length_solid**2/(wall_diffusivity)

    # from mirgecom.wall_model import get_porous_flow_timestep
    def _my_get_timestep_fluid(fluid_state, t, dt):
        # return get_porous_flow_timestep(dcoll, gas_model, fluid_state,
        #                                 current_cfl, dd_vol_fluid)
        return get_sim_timestep(
            dcoll, fluid_state, t, dt, current_cfl, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd_vol_fluid)

    my_get_timestep_wall = actx.compile(_my_get_timestep_wall)
    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed, sample_densities, wall_cv = state

        cv = force_eval(actx, cv)
        tseed = force_eval(actx, tseed)

        # include both outflow and sponge in the damping region
        smoothness = force_eval(actx, smooth_region + sponge_sigma/sponge_amp)

        # damp the outflow
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        # fluid_state = get_fluid_state(cv, sample_wv, tseed)
        fluid_state = get_fluid_state(cv, sample_densities, tseed)
        cv = fluid_state.cv

        # wall variables
        wall_cv = force_eval(actx, wall_cv)
        solid_state = get_solid_state(wall_cv)
        wall_cv = solid_state.cv
        # wdv = solid_state.dv

        t = force_eval(actx, t)
        dt_fluid = force_eval(actx,
                              my_get_timestep_fluid(fluid_state, t[0], dt[0]))
#        dt_fluid = force_eval(actx, actx.np.minimum(
#            current_dt, my_get_timestep_fluid(fluid_state, t[0], dt[0])))

        dt_solid = force_eval(actx,
                              my_get_timestep_wall(solid_state, t[2], dt[2]))
#        dt_solid = force_eval(actx, actx.np.minimum(
#            1.0e-8, my_get_timestep_wall(solid_state, t[2], dt[2])))
#        dt_solid = force_eval(actx, current_dt + solid_zeros)

        dt = make_obj_array([dt_fluid, fluid_zeros, fluid_zeros, dt_solid])

        try:
            state = make_obj_array([
                fluid_state.cv, fluid_state.dv.temperature,
                fluid_state.wv.material_densities, solid_state.cv])

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            t_elapsed = time.time() - t_start
            if t_shutdown - t_elapsed < 300.0:
                my_write_restart(step, t, state)
                sys.exit()

            ngarbage = 10
            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    warn("Running gc.collect() to work around memory growth issue ")
                    gc.collect()

            if step % 1000 == 0:
                dd_centerline = dd_vol_solid.trace("wall_sym")
                temperature_centerline = op.project(
                    dcoll, dd_vol_solid, dd_centerline, solid_state.dv.temperature)
                min_temp_center = vol_min(dd_centerline, temperature_centerline)
                max_temp_center = vol_max(dd_centerline, temperature_centerline)
                max_temp = vol_max(dd_vol_solid, solid_state.dv.temperature)

                wall_time = np.max(actx.to_numpy(t[2]))
                if step > last_stored_step:
                    my_file = open("temperature_file.dat", "a")
                    my_file.write(f"{wall_time:.8f}, {step}, {min_temp_center:.8f}, {max_temp_center:.8f}, {max_temp:.8f} \n")
                    my_file.close()

                gc.freeze()

            if do_health:
                # FIXME warning in lazy compilation
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
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
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                             solid_state=solid_state, smoothness=smoothness)
                gc.freeze()

            if do_restart:
                my_write_restart(step, t, state)
                gc.freeze()

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                         solid_state=solid_state, smoothness=smoothness)
            raise

        return state, dt

    def _get_rhs(t, state):

        fluid_state, solid_state = state

        cv = fluid_state.cv

        # wall_cv = solid_state.cv
        wdv = solid_state.dv

        # ~~~~~~~~~~~~~
        fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
            add_interface_boundaries_no_grad(
                dcoll, gas_model,
                dd_vol_fluid, dd_vol_solid,
                fluid_state, wdv.thermal_conductivity, wdv.temperature,
                fluid_boundaries, solid_boundaries,
                interface_noslip=True, interface_radiation=use_radiation)

        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_fluid, comm_tag=_FluidOpStatesTag,
            limiter_func=_limit_fluid_cv)

        # fluid grad CV
        fluid_grad_cv = grad_cv_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=t, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradCVTag)

        # fluid grad T
        fluid_grad_temperature = grad_t_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=t, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradTempTag)

        # solid grad T
        solid_grad_temperature = wall_grad_t_operator(
            dcoll, wdv.thermal_conductivity,
            solid_all_boundaries_no_grad, wdv.temperature,
            quadrature_tag=quadrature_tag, dd=dd_vol_solid,
            # numerical_flux_func=grad_facial_flux_weighted,
            comm_tag=_SolidGradTempTag)

        fluid_all_boundaries, solid_all_boundaries = \
            add_interface_boundaries(
                dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
                fluid_state, wdv.thermal_conductivity,
                wdv.temperature,
                fluid_grad_temperature, solid_grad_temperature,
                fluid_boundaries, solid_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                wall_emissivity=wall_emissivity, sigma=5.67e-8,
                ambient_temperature=300.0,
                wall_penalty_amount=wall_penalty_amount)

        fluid_rhs = ns_operator(
            dcoll, gas_model, fluid_state, fluid_all_boundaries,
            time=t, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
            comm_tag=_FluidOperatorTag, inviscid_terms_on=True)

        # ~~~~~~~~~~~~~
        oxidation, sample_mass_rhs = oxidation_source_terms(fluid_state)

        energy_radiation = radiation_sink_terms(
            fluid_all_boundaries_no_grad, fluid_state.temperature,
            fluid_state.wv.void_fraction)

        radiation = make_conserved(
            dim=dim, mass=fluid_zeros, momentum=cv.momentum*0.0,
            energy=energy_radiation, species_mass=cv.species_mass*0.0)

        # ~~~~
        fluid_sources = (
            chemical_source_term(fluid_state.cv, fluid_state.temperature)
            + sponge_func(cv=fluid_state.cv, cv_ref=ref_cv, sigma=sponge_sigma)
            + gravity_source_terms(fluid_state.cv)
            + axisym_source_fluid(actx, dcoll, fluid_state, fluid_grad_cv,
                                  fluid_grad_temperature)
            + darcy_source_terms(fluid_state)
            + radiation
            + oxidation
        )

        # ~~~~~~~~~~~~~
        solid_energy_rhs = diffusion_operator(
            dcoll, wdv.thermal_conductivity, solid_all_boundaries,
            wdv.temperature,
            penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag,
            dd=dd_vol_solid,
            grad_u=solid_grad_temperature,
            comm_tag=_SolidOperatorTag,
        )

        solid_sources = wall_time_scale * axisym_source_solid(
            actx, dcoll, solid_state, solid_grad_temperature)

        solid_rhs = wall_time_scale * SolidWallConservedVars(
            mass=solid_zeros, energy=solid_energy_rhs)

        # ~~~~~~~~~~~~~

        return make_obj_array([fluid_rhs + fluid_sources,
                               fluid_zeros,
                               sample_mass_rhs,
                               solid_rhs + solid_sources])

    get_rhs_compiled = actx.compile(_get_rhs)

    def my_rhs(t, state):
        cv, tseed, sample_densities, wall_cv = state

        cv = force_eval(actx, cv)
        tseed = force_eval(actx, tseed)

        t = force_eval(actx, t)
        smoothness = force_eval(actx, smooth_region + sponge_sigma/sponge_amp)

        # apply outflow damping
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, sample_densities, tseed)
        fluid_state = force_eval(actx, fluid_state)
        cv = fluid_state.cv

        # construct wall state
        wall_cv = force_eval(actx, wall_cv)
        solid_state = get_solid_state(wall_cv)

        solid_state = force_eval(actx, solid_state)
        wall_cv = solid_state.cv
        # wdv = solid_state.dv

        actual_state = make_obj_array([fluid_state, solid_state])

        return get_rhs_compiled(t, actual_state)

    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer.start_sub_timer():
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

    dt_fluid = force_eval(
        actx, my_get_timestep_fluid(
            fluid_state,
            force_eval(actx, current_t + fluid_zeros),
            force_eval(actx, current_dt + fluid_zeros)))
    dt_solid = force_eval(actx, current_dt + solid_zeros)

    t_fluid = force_eval(actx, current_t + fluid_zeros)
    t_solid = force_eval(actx, current_t + solid_zeros)

    stepper_state = make_obj_array([fluid_state.cv, fluid_state.dv.temperature,
                                    sample_densities,
                                    solid_state.cv])

    dt = make_obj_array([dt_fluid, fluid_zeros, fluid_zeros, dt_solid])
    t = make_obj_array([t_fluid, t_fluid, t_fluid, t_solid])

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt, t=t, t_final=t_final,
                      max_steps=niter, local_dt=local_dt,
                      force_eval=force_eval_stepper, state=stepper_state,
                      compile_rhs=False)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    sys.exit()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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

    args = parser.parse_args()

    # for writing output
    casename = "burner_mix"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if args.input_file:
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    main(actx_class, use_logmgr=args.log, casename=casename, use_tpe=args.tpe,
         restart_file=restart_file)
