""" Sun 03 Sep 2023 07:02:39 PM CDT """

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
from mirgecom.utils import force_evaluation, mask_from_elements
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
    LinearizedOutflowBoundary,
)
from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient, make_conserved
)
from mirgecom.transport import (
    SimpleTransport,
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
from mirgecom.materials.initializer import (
    SolidWallInitializer,
    PorousWallInitializer
)
from mirgecom.wall_model import (
    PorousWallTransport,
    PorousFlowModel,
    SolidWallModel,
    SolidWallState,
    SolidWallDependentVars,
    SolidWallConservedVars,
)

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array

#########################################################################

class _MyGradTag_FH:
    pass

class _MyGradTag_FS:
    pass

class _MyGradTag_SH:
    pass

class _MyGradTag_F1:
    pass

class _MyGradTag_F2:
    pass

class _MyGradTag_F3:
    pass

class _MyGradTag_F4:
    pass

class _MyGradTag_F5:
    pass

class _MyGradTag_F6:
    pass

class _MyGradTag_S1:
    pass

class _MyGradTag_S2:
    pass

class _MyGradTag_S3:
    pass

class _MyGradTag_S4:
    pass

class _MyGradTag_S5:
    pass

class _MyGradTag_S6:
    pass


class _FluidGradCVTag:
    pass

class _FluidGradTempTag:
    pass

class _SampleGradCVTag:
    pass

class _SampleGradTempTag:
    pass

class _HolderGradTempTag:
    pass

class _FluidOperatorTag:
    pass

class _SampleOperatorTag:
    pass

class _HolderOperatorTag:
    pass

class _FluidOpStatesTag:
    pass

class _WallOpStatesTag:
    pass

class Burner2D_Reactive:

    def __init__(self, *, dim=2, nspecies, sigma, sigma_flame,
        flame_loc, pressure, temperature, speedup_factor,
        mass_rate_burner, mass_rate_shroud,
        species_unburn, species_burned, species_shroud, species_atm):

        self._dim = dim
        #self._nspecies = nspecies 
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

    def __call__(self, actx, x_vec, eos, solve_the_flame=True,
                 state_minus=None, time=None):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

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

        int_diam = 2.38*25.4/2000.0 #radius, actually
        ext_diam = 2.89*25.4/2000.0 #radius, actually

        #~~~ shroud
        S1 = 0.5*(1.0 + actx.np.tanh(1.0/(_sigma)*(x_vec[0] - int_diam)))
        S2 = actx.np.where(actx.np.greater(x_vec[0], ext_diam),
                 0.0, - actx.np.tanh(1.0/(_sigma)*(x_vec[0] - ext_diam))
             )
        shroud = S1*S2

        #~~~ flame ignition
        core = 0.5*(1.0 - actx.np.tanh(1.0/_sigma*(x_vec[0] - int_diam)))

        #~~~ atmosphere
        atmosphere = 1.0 - (shroud + core)
             
        #~~~ after combustion products
        upper_atm = 0.5*(1.0 + actx.np.tanh( 1.0/(2.0*self._sigma)*(x_vec[1] - upper_bnd)))

        if solve_the_flame:

            #~~~ flame ignition
            flame = 0.5*(1.0 + actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - self._flaLoc)))

            #~~~ species
            yf = (flame*_yb + (1.0-flame)*_yu)*(1.0-upper_atm) + _ya*upper_atm
            ys = _ya*upper_atm + (1.0 - upper_atm)*_ys
            y = atmosphere*_ya + shroud*ys + core*yf

            #~~~ temperature and EOS
            temp = (flame*self._temp + (1.0-flame)*cool_temp)*(1.0-upper_atm) + 300.0*upper_atm
            temperature = temp*core + 300.0*(1.0 - core)
            
            if state_minus is None:
                pressure = self._pres + 0.0*x_vec[0]
                mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
            else:
                mass = state_minus.cv.mass

            #~~~ velocity and/or momentum
            mom_burner = self._mass_rate_burner*self._speedup_factor
            mom_shroud = self._mass_rate_shroud*self._speedup_factor
            smoothY = mom_burner*(1.0-upper_atm) + 0.0*upper_atm
            mom_x = 0.0*x_vec[0]
            mom_y = core*smoothY + shroud*mom_shroud*(1.0-upper_atm)
            momentum = make_obj_array([mom_x, mom_y])
            velocity = momentum/mass

            #~~~ 
            specmass = mass * y

            #~~~ 
            internal_energy = eos.get_internal_energy(
                temperature, species_mass_fractions=y)
            kinetic_energy = 0.5 * np.dot(velocity, velocity)
            energy = mass * (internal_energy + kinetic_energy)

#        else:

#            #~~~ flame ignition
#            flame = 1.0

#            #~~~ species
#            yf = (flame*_yb + (1.0-flame)*_yu)*(1.0-upper_atm) + _ya*upper_atm
#            ys = _ya*upper_atm + (1.0 - upper_atm)*_ys
#            y = atmosphere*_ya + shroud*ys + core*yf

#            #~~~ temperature and EOS
#            temp = self._temp*(1.0-upper_atm) + 300.0*upper_atm
#            temperature = temp*core + 300.0*(1.0 - core)

#            if state_minus is None:
#                pressure = self._pres + 0.0*x_vec[0]
#                mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
#            else:
#                mass = state_minus.cv.mass

#            #~~~ velocity
#            mom_inlet = self._mass_rate_burner*self._speedup_factor
#            mom_shroud = self._mass_rate*self._speedup_factor
#            smoothY = mom_inlet*(1.0-upper_atm) + 0.0*upper_atm
#            mom_x = 0.0*x_vec[0]
#            mom_y = core*smoothY + shroud*mom_shroud*(1.0-upper_atm)
#            momentum = make_obj_array([mom_x, mom_y])
#            velocity = momentum/mass

#            #~~~ 
#            specmass = mass * y

#            #~~~ 
#            internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
#            kinetic_energy = 0.5 * np.dot(velocity, velocity)
#            energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                momentum=mass*velocity, species_mass=specmass)


#class Burner2D_Reactive:

#    def __init__(self, flame_loc, pressure, temperature, speedup_factor,
#        mixture_velocity, shroud_velocity,
#        species_unburn, species_burned, species_shroud, species_atm, *,
#        sigma=1e-3, sigma_flame=2e-4):

#        self._sigma = sigma
#        self._sigma_flame = sigma_flame
#        self._pres = pressure
#        self._speedup_factor = speedup_factor
#        self._v_mixture = mixture_velocity
#        self._v_shroud = shroud_velocity
#        self._yu = species_unburn
#        self._yb = species_burned
#        self._ys = species_shroud
#        self._ya = species_atm
#        self._temp = temperature
#        self._flaLoc = flame_loc

#    def __call__(self, x_vec, gas_model, solve_the_flame=True,
#                 state_minus=None, time=None):

#        actx = x_vec[0].array_context
#        eos = gas_model.eos

#        cool_temp = 300.0

#        upper_bnd = 0.105

#        sigma_factor = 12.0 - 11.0*(upper_bnd - x_vec[1])**2/(upper_bnd - 0.10)**2
#        _sigma = self._sigma*(
#            actx.np.where(actx.np.less(x_vec[1], upper_bnd),
#                              actx.np.where(actx.np.greater(x_vec[1], .10),
#                                            sigma_factor, 1.0),
#                              12.0)
#        )
##        _sigma = self._sigma

#        int_diam = 2.38*25.4/2000 #radius, actually
#        ext_diam = 2.89*25.4/2000 #radius, actually

#        #~~~ shroud
#        S1 = 0.5*(1.0 + actx.np.tanh(1.0/(_sigma)*(x_vec[0] - int_diam)))
#        S2 = actx.np.where(actx.np.greater(x_vec[0], ext_diam),
#                 0.0, - actx.np.tanh(1.0/(_sigma)*(x_vec[0] - ext_diam))
#             )
#        shroud = S1*S2

#        #~~~ flame ignition
#        core = 0.5*(1.0 - actx.np.tanh(1.0/_sigma*(x_vec[0] - int_diam)))

#        #~~~ atmosphere
#        atmosphere = 1.0 - (shroud + core)
#             
#        #~~~ after combustion products
#        upper_atm = 0.5*(1.0 + actx.np.tanh( 1.0/(2.0*self._sigma)*(x_vec[1] - upper_bnd)))

#        if solve_the_flame:

#            #~~~ flame ignition
#            flame = 0.5*(1.0 + actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - self._flaLoc)))

#            #~~~ species
#            yf = (flame*self._yb + (1.0-flame)*self._yu)*(1.0-upper_atm) + self._ya*upper_atm
#            ys = self._ya*upper_atm + (1.0 - upper_atm)*self._ys
#            y = atmosphere*self._ya + shroud*ys + core*yf

#            #~~~ temperature and EOS
#            temp = (flame*self._temp + (1.0-flame)*cool_temp)*(1.0-upper_atm) + 300.0*upper_atm
#            temperature = temp*core + 300.0*(1.0 - core)
#            
#            if state_minus is None:
#                pressure = self._pres + 0.0*x_vec[0]
#                mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
#            else:
#                mass = state_minus.cv.mass

#            #~~~ velocity and/or momentum
#            mom_inlet = mass*self._v_mixture*self._speedup_factor
#            smoothY = mom_inlet*(1.0-upper_atm) + 0.0*upper_atm
#            mom_shroud = mass*self._v_shroud*self._speedup_factor
#            momentum = make_obj_array([
#                0.0*x_vec[0], core*smoothY + shroud*mom_shroud*(1.0-upper_atm)])
#            velocity = momentum/mass

#            #~~~ 
#            specmass = mass * y

#            #~~~ 
#            internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
#            kinetic_energy = 0.5 * np.dot(velocity, velocity)
#            energy = mass * (internal_energy + kinetic_energy)

#        else:

#            #~~~ flame ignition
#            flame = 1.0

#            #~~~ species
#            yf = (flame*self._yb + (1.0-flame)*self._yu)*(1.0-upper_atm) + self._ya*upper_atm
#            ys = self._ya*upper_atm + (1.0 - upper_atm)*self._ys
#            y = atmosphere*self._ya + shroud*ys + core*yf

#            #~~~ temperature and EOS
#            temp = self._temp*(1.0-upper_atm) + 300.0*upper_atm
#            temperature = temp*core + 300.0*(1.0 - core)

#            if state_minus is None:
#                pressure = self._pres + 0.0*x_vec[0]
#                mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
#            else:
#                mass = state_minus.cv.mass

#            #~~~ velocity
#            mom_inlet = mass*self._v_mixture*self._speedup_factor
#            smoothY = mom_inlet*(1.0-upper_atm) + 0.0*upper_atm
#            mom_shroud = mass*self._v_shroud*self._speedup_factor
#            momentum = make_obj_array([
#                0.0*x_vec[0], core*smoothY + shroud*mom_shroud*(1.0-upper_atm)])
#            velocity = momentum/mass

#            #~~~ 
#            specmass = mass * y

#            #~~~ 
#            internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
#            kinetic_energy = 0.5 * np.dot(velocity, velocity)
#            energy = mass * (internal_energy + kinetic_energy)

#        return make_conserved(dim=2, mass=mass, energy=energy,
#                momentum=mass*velocity, species_mass=specmass)




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

    sys.setrecursionlimit(5000)

    mesh_filename = "mesh_11m_10mm_020um_3domains-v2.msh"

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 25000
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
    use_soln_filter = False
    use_rhs_filter = False

    x0_sponge = 0.150
    sponge_amp = 400.0
    theta_factor = 0.02
    speedup_factor = 7.5

    mechanism_file = "uiuc_7sp"
#    mechanism_file = "uiuc_8sp_phenol"
    equiv_ratio = 0.7
    chem_rate = 0.3 #keep it between 0.0 and 1.0
    flow_rate = 25.0
    shroud_rate = 11.85
    Twall = 300.0
    T_products = 2000.0
    solve_the_flame = True

#    transport = "Mixture"
    transport = "PowerLaw"

    # wall stuff
    ignore_wall = False
    my_material = "fiber"
#    my_material = "composite"
    temp_wall = 300

    wall_penalty_amount = 1.0
    wall_time_scale = speedup_factor*25.0

    emissivity = 0.85*speedup_factor

    restart_iterations = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                "fluid": ["fluid"],
                "sample": ["wall_sample"],
                "holder": ["wall_alumina", "wall_graphite"]
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
        + volume_to_local_mesh_data["sample"][0].nelements
        + volume_to_local_mesh_data["holder"][0].nelements)

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
    dd_vol_holder = DOFDesc(VolumeDomainTag("holder"), DISCR_TAG_BASE)

    fluid_nodes = force_evaluation(actx, dcoll.nodes(dd_vol_fluid))
    sample_nodes = force_evaluation(actx, dcoll.nodes(dd_vol_sample))
    holder_nodes = force_evaluation(actx, dcoll.nodes(dd_vol_holder))

    fluid_zeros = actx.np.zeros_like(fluid_nodes[0])
    sample_zeros = actx.np.zeros_like(sample_nodes[0])
    holder_zeros = actx.np.zeros_like(holder_nodes[0])

    #~~~~~~~~~~
    wall_tag_to_elements = volume_to_local_mesh_data["holder"][1]
    wall_alumina_mask = mask_from_elements(
        dcoll, dd_vol_holder, actx, wall_tag_to_elements["wall_alumina"])
    wall_graphite_mask = mask_from_elements(
        dcoll, dd_vol_holder, actx, wall_tag_to_elements["wall_graphite"])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = T_products

    air = "O2:0.21,N2:0.79"
    fuel = "C2H4:1"
    cantera_soln.set_equivalence_ratio(phi=equiv_ratio,
                                       fuel=fuel, oxidizer=air)
    x_unburned = cantera_soln.X
    pres_unburned = cantera.one_atm

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

    mmw_N2 = cantera_soln.molecular_weights[cantera_soln.species_index("N2")]
    rho_ext = 101325.0/((8314.46/mmw_N2)*300.0)
    mdot_ext = rho_ext*u_ext*A_ext
    rhoU_ext = rho_ext*u_ext

    print("V_dot=",mass_react,"(L/min)")
    print(f"{A_int= }","(m^2)")
    print(f"{A_ext= }","(m^2)")
    print(f"{u_int= }","(m/s)")
    print(f"{u_ext= }","(m/s)")
    print(f"{rho_int= }","(kg/m^3)")
    print(f"{rho_ext= }","(kg/m^3)")
    print(f"{rhoU_int= }")
    print(f"{rhoU_ext= }")
    print("ratio=",u_ext/u_int)

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x_unburned}")

    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x_unburned

    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY
    mmw_unburned = cantera_soln.mean_molecular_weight

    cantera_soln.TPX = temp_ignition, pres_unburned, x_unburned
    cantera_soln.equilibrate("TP")
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P
    mmw_burned = cantera_soln.mean_molecular_weight

    # Pull temperature, density, mass fractions, and pressure from Cantera
    x = np.zeros(nspecies)
    x[cantera_soln.species_index("O2")] = 0.21
    x[cantera_soln.species_index("N2")] = 0.79
    cantera_soln.TPX = temp_unburned, pres_unburned, x

    y_atmosphere = np.zeros(nspecies)
    dummy, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    cantera_soln.equilibrate("TP")
    temp_atmosphere, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    pres_atmosphere = cantera_soln.P
    mmw_atmosphere = cantera_soln.mean_molecular_weight
    
    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_shroud = y_atmosphere*0.0
    y_shroud[cantera_soln.species_index("N2")] = 1.0

    cantera_soln.TPY = 300.0, 101325.0, y_shroud
    temp_shroud, rho_shroud = cantera_soln.TD
    mmw_shroud = cantera_soln.mean_molecular_weight

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
    print(f"Unburned:")
    print(f"T = {temp_unburned}")
    print(f"D = {rho_unburned}")
    print(f"Y = {y_unburned}")
    print(f"W = {mmw_unburned}")
    print(f" ")
    print(f"Burned:")
    print(f"T = {temp_burned}")
    print(f"D = {rho_burned}")
    print(f"Y = {y_burned}")
    print(f"W = {mmw_burned}")
    print(f" ")
    print(f"Atmosphere:")
    print(f"T = {temp_atmosphere}")
    print(f"D = {rho_atmosphere}")
    print(f"Y = {y_atmosphere}")
    print(f"W = {mmw_atmosphere}")
    print(f" ")
    print(f"Shroud:")
    print(f"T = {temp_shroud}")
    print(f"D = {rho_shroud}")
    print(f"Y = {y_shroud}")
    print(f"W = {mmw_shroud}")

    # }}}
    
    # {{{ Initialize transport model

    # ~~~ fluid domain
    if transport == "Mixture":
        physical_transport = MixtureAveragedTransport(
            pyrometheus_mechanism, lewis=np.ones(nspecies,),
            factor=speedup_factor)
    else:
        if transport == "PowerLaw":
            physical_transport = PowerLawTransport(lewis=np.ones(nspecies,),
               beta=4.093e-7*speedup_factor)
        else:
            print('No transport class defined..')
            print('Use one of "Mixture" or "PowerLaw"')
            sys.exit()

    # ~~~ porous media
    # do not scale the transport model because the RHS will be scaled
    base_transport = PowerLawTransport(lewis=np.ones(nspecies,), beta=4.093e-7)
    sample_transport = PorousWallTransport(base_transport=base_transport)

    # }}}

    # ~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    if my_material == "fiber":
        import mirgecom.materials.carbon_fiber as material_sample
        material = material_sample.FiberEOS(char_mass=0.0,
                                            virgin_mass=160.0)
        decomposition = material_sample.Y3_Oxidation_Model(wall_material=material)

    if my_material == "composite":
        import mirgecom.materials.tacot as material_sample
        material = material_sample.TacotEOS(char_mass=220.0,
                                            virgin_mass=280.0)
        decomposition = material_sample.Pyrolysis()   

    # }}}

    # ~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    # Averaging from https://www.azom.com/properties.aspx?ArticleID=52 for alumina
    wall_alumina_rho = 3500.0
    wall_alumina_cp = 700.0
    wall_alumina_kappa = 25.00

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630 for graphite
    # TODO There is a table with the temperature-dependent data for graphite
    wall_graphite_rho = 1625.0
    wall_graphite_cp = 770.0
    wall_graphite_kappa = 50.0

#    def _get_holder_density():
#        return (wall_alumina_rho * wall_alumina_mask
#                + wall_graphite_rho * wall_graphite_mask)

    def _get_holder_enthalpy(temperature, **kwargs):
        wall_alumina_h = wall_alumina_cp * temperature
        wall_graphite_h = wall_graphite_cp * temperature
        return (wall_alumina_h * wall_alumina_mask
                + wall_graphite_h * wall_graphite_mask)

    def _get_holder_heat_capacity(temperature, **kwargs):
        return (wall_alumina_cp * wall_alumina_mask
                + wall_graphite_cp * wall_graphite_mask)

    def _get_holder_thermal_conductivity(temperature, **kwargs):
        return (wall_alumina_kappa * wall_alumina_mask
                + wall_graphite_kappa * wall_graphite_mask)

    holder_wall_model = SolidWallModel(
        #density_func=_get_holder_density,
        enthalpy_func=_get_holder_enthalpy,
        heat_capacity_func=_get_holder_heat_capacity,
        thermal_conductivity_func=_get_holder_thermal_conductivity)

    # }}}

    gas_model_fluid = GasModel(eos=eos, transport=physical_transport)

    gas_model_sample = PorousFlowModel(eos=eos, transport=sample_transport,
                                       wall_eos=material)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # initialize the sponge field
    sponge_x_thickness = 0.055
    sponge_y_thickness = 0.055

    xMaxLoc = x0_sponge + sponge_x_thickness
    yMinLoc = 0.04

    sponge_init = InitSponge(amplitude=sponge_amp,
        x_max=xMaxLoc, y_min=yMinLoc,
        x_thickness=sponge_x_thickness,
        y_thickness=sponge_y_thickness)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # --- Filtering settings ---
    # ------ Solution filtering
    soln_filter_frac = 0.5
    # soln_filter_cutoff = -1 => filter_frac*order)
    soln_filter_cutoff = 1 #-1
    soln_filter_order = 2
    soln_filter_alpha = -1.0*np.log(np.finfo(float).eps)

    # ------ RHS filtering
    rhs_filter_frac = 0.5
    rhs_filter_cutoff = 1 #-1
    rhs_filter_order = 2
    rhs_filter_alpha = -1.0*np.log(np.finfo(float).eps)

    from mirgecom.filter import (
        exponential_mode_response_function as xmrfunc,
        filter_modally
    )

    if use_soln_filter:
        if soln_filter_cutoff < 0:
            soln_filter_cutoff = int(soln_filter_frac * order)

        if soln_filter_cutoff >= order:
            raise ValueError("Invalid setting for solution filter (cutoff >= order).")

        soln_frfunc = partial(xmrfunc, alpha=soln_filter_alpha,
                              filter_order=soln_filter_order)

        logger.info("Solution filtering settings:")
        logger.info(f" - filter alpha  = {soln_filter_alpha}")
        logger.info(f" - filter cutoff = {soln_filter_cutoff}")
        logger.info(f" - filter order  = {soln_filter_order}")

        def filter_sample_cv(cv):
            return filter_modally(dcoll, soln_filter_cutoff, soln_frfunc, cv,
                                  dd=dd_vol_sample)

        filter_sample_cv_compiled = actx.compile(filter_sample_cv)

    if use_rhs_filter:

        if rhs_filter_cutoff < 0:
            rhs_filter_cutoff = int(rhs_filter_frac * order)

        if rhs_filter_cutoff >= order:
            raise ValueError("Invalid setting for RHS filter (cutoff >= order).")

        rhs_frfunc = partial(xmrfunc, alpha=rhs_filter_alpha,
                             filter_order=rhs_filter_order)

        logger.info("RHS filtering settings:")
        logger.info(f" - filter alpha  = {rhs_filter_alpha}")
        logger.info(f" - filter cutoff = {rhs_filter_cutoff}")
        logger.info(f" - filter order  = {rhs_filter_order}")

        def filter_sample_rhs(rhs):
            return filter_modally(dcoll, rhs_filter_cutoff, rhs_frfunc, rhs,
                                  dd=dd_vol_sample)

        filter_sample_rhs_compiled = actx.compile(filter_sample_rhs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = actx.np.zeros_like(cv.mass)
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

    def _limit_sample_cv(cv, wv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = actx.np.zeros_like(cv.mass)
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        #XXX recompute density with pressure forced to be 1atm
        mass_lim = wv.void_fraction*gas_model_sample.eos.get_density(
            pressure=actx.np.zeros_like(pressure) + 101325.0,
            temperature=temperature, species_mass_fractions=spec_lim)

        # ignore velocity inside the wall
        velocity = actx.np.zeros_like(cv.velocity)

        # recompute energy
        energy_gas = mass_lim*(gas_model_sample.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(velocity, velocity)
        )

        energy_solid = \
            wv.density*gas_model_sample.wall_eos.enthalpy(temperature, wv.tau)

        energy = energy_gas + energy_solid

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy,
            momentum=mass_lim*velocity, species_mass=mass_lim*spec_lim)

    # ~~~~~~~~~~

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_fluid,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

    get_fluid_state = actx.compile(_get_fluid_state)

    def _get_sample_state(cv, wv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_sample,
            material_densities=wv, temperature_seed=temp_seed,
            limiter_func=_limit_sample_cv, limiter_dd=dd_vol_sample
        )

    get_sample_state = actx.compile(_get_sample_state)

    def _get_holder_state(wv):
        dep_vars = holder_wall_model.dependent_vars(wv)
        return SolidWallState(cv=wv, dv=dep_vars)

    get_holder_state = actx.compile(_get_holder_state)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_init = Burner2D_Reactive(dim=dim, nspecies=nspecies, sigma=0.00020,
        sigma_flame=0.00005, temperature=temp_ignition, pressure=101325.0,
        flame_loc=0.10025, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

    ref_state = Burner2D_Reactive(dim=dim, nspecies=nspecies, sigma=0.00020,
        sigma_flame=0.00001, temperature=temp_ignition, pressure=101325.0,
        flame_loc=0.1050, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

#    fluid_init = Burner2D_Reactive(sigma=0.00020, sigma_flame=0.00005,
#        temperature=temp_ignition, pressure=101325.0, flame_loc=0.10025,
#        speedup_factor=speedup_factor,
#        mixture_velocity=u_int, shroud_velocity=u_ext,
#        species_shroud=y_shroud, species_atm=y_atmosphere,
#        species_unburn=y_unburned, species_burned=y_burned)

#    ref_state = Burner2D_Reactive(sigma=0.00020, sigma_flame=0.00001,
#        temperature=temp_ignition, pressure=101325.0, flame_loc=0.1050,
#        speedup_factor=speedup_factor,
#        mixture_velocity=u_int, shroud_velocity=u_ext,
#        species_shroud=y_shroud, species_atm=y_atmosphere,
#        species_unburn=y_unburned, species_burned=y_burned)

    if my_material == "fiber":
        sample_density = 0.1*1600.0 + sample_zeros

    if my_material == "composite":
        # soln setup and init
        sample_density = np.empty((3,), dtype=object)

        sample_density[0] = 30.0 + sample_zeros
        sample_density[1] = 90.0 + sample_zeros
        sample_density[2] = 160. + sample_zeros

    sample_init = PorousWallInitializer(
        pressure=101325.0, temperature=300.0,
        species=y_atmosphere, material_densities=sample_density)
    
    holder_mass = (wall_alumina_rho * wall_alumina_mask
                   + wall_graphite_rho * wall_graphite_mask)
    holder_init = SolidWallInitializer(temperature=300.0,
                                       material_densities=holder_mass)

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")

        fluid_tseed = temperature_seed*1.0
        sample_tseed = temp_wall*1.0

#        fluid_cv = fluid_init(fluid_nodes, gas_model_fluid,
#                              solve_the_flame=solve_the_flame)
        fluid_cv = fluid_init(actx, fluid_nodes, eos, solve_the_flame=solve_the_flame)

        sample_cv = sample_init(2, sample_nodes, gas_model_sample)

        holder_cv = holder_init(holder_nodes, holder_wall_model)

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
            holder_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_holder),
                restart_dcoll.discr_from_dd(dd_vol_holder)
            )
            fluid_cv = fluid_connection(restart_data["fluid_cv"])
            fluid_tseed = fluid_connection(restart_data["fluid_temperature_seed"])
            sample_cv = sample_connection(restart_data["sample_cv"])
            sample_tseed = sample_connection(restart_data["wall_temperature_seed"])
            sample_density = sample_connection(restart_data["sample_density"])
            holder_cv = holder_connection(restart_data["holder_cv"])
        else:
            fluid_cv = restart_data["fluid_cv"]
            fluid_tseed = restart_data["fluid_temperature_seed"]
            sample_cv = restart_data["sample_cv"]
            sample_tseed = restart_data["sample_temperature_seed"]
            sample_density = restart_data["sample_density"]
            holder_cv = restart_data["holder_cv"]
            #holder_cv_dummy = restart_data["holder_cv"]
            #holder_cv = SolidWallConservedVars(mass=holder_cv_dummy.mass,
            #                                   energy=holder_cv_dummy.energy)

    #XXX dummy piece of code to switch from fiber to phenolic
    if False:
        sample_density = np.empty((3,), dtype=object)

        sample_density[0] = 30.0 + sample_zeros
        sample_density[1] = 90.0 + sample_zeros
        sample_density[2] = 160. + sample_zeros

        sample_init = PorousWallInitializer(pressure=101325.0, temperature=300.0,
                species=y_atmosphere, material_densities=sample_density)
        sample_cv = sample_init(actx, sample_nodes, gas_model_sample)

##############################################################################

    fluid_cv = force_evaluation(actx, fluid_cv)
    fluid_tseed = force_evaluation(actx, fluid_tseed)
    fluid_state = get_fluid_state(fluid_cv, fluid_tseed)

    sample_cv = force_evaluation(actx, sample_cv)
    sample_tseed = force_evaluation(actx, sample_tseed)
    sample_density = force_evaluation(actx, sample_density)
    sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)

    holder_cv = force_evaluation(actx, holder_cv)
    holder_state = get_holder_state(holder_cv)  

    # ~~~~
    smooth_region = force_evaluation(
        actx, smoothness_region(dcoll, fluid_nodes))

    # ~~~~
    reaction_rates_damping = force_evaluation(
        actx, reaction_damping(dcoll, fluid_nodes))

    # ~~~~
    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=fluid_nodes))
#    ref_cv = force_evaluation(actx,
#        ref_state(fluid_nodes, gas_model_fluid, solve_the_flame))
    ref_cv = force_evaluation(actx,
        ref_state(actx, fluid_nodes, eos, solve_the_flame))

    # ~~~~
    # track mass loss
    from grudge.reductions import integral

    eps_rho_solid = gas_model_sample.solid_density(sample_density)
    initial_mass = integral(dcoll, dd_vol_sample,
                            np.pi*sample_nodes[0]*eps_rho_solid + sample_zeros)

    initial_mass = force_evaluation(actx, initial_mass)
    if restart_file is None:
        if rank == 0:
            logging.info(f"{initial_mass=}")    

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing logmgr.")

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

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing BC setup")

    inflow_nodes = force_evaluation(actx,
                                    dcoll.nodes(dd_vol_fluid.trace('inlet')))
    inflow_temperature = actx.np.zeros_like(inflow_nodes[0]) + 300.0
    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return inflow_temperature

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
#        inflow_cv_cond = ref_state(x_vec=inflow_nodes, gas_model=gas_model,
#            solve_the_flame=solve_the_flame, state_minus=state_minus)
        inflow_cv_cond = ref_state(actx=actx, x_vec=inflow_nodes, eos=eos,
            solve_the_flame=solve_the_flame, state_minus=state_minus)
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
            temperature_seed=300.0)

# #XXX overintegration:

#    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
##        inflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
##        inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
#        inflow_zeros = actx.np.zeros_like(state_minus.cv.mass)
#        return inflow_zeros + 300.0

#    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
#        inflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
#        inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
#        inflow_cv_cond = ref_state(x_vec=inflow_nodes, gas_model=gas_model,
#            solve_the_flame=solve_the_flame, state_minus=state_minus)
#        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
#            temperature_seed=300.0)

    from mirgecom.inviscid import inviscid_flux
    from mirgecom.flux import num_flux_central
    from mirgecom.viscous import viscous_flux

    class MyPrescribedBoundary(PrescribedFluidBoundary):
        r"""My prescribed boundary function. """

        def __init__(self, bnd_state_func, temperature_func):
            """Initialize the boundary condition object."""
            self.bnd_state_func = bnd_state_func
            PrescribedFluidBoundary.__init__(self,
            boundary_state_func=bnd_state_func,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_temperature_func=temperature_func,
            boundary_gradient_cv_func=self.grad_cv_bc)

        def prescribed_state_for_advection(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
            state_plus = self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                             state_minus, **kwargs)

            mom_x = - state_minus.cv.momentum[0]
            mom_y = - state_minus.cv.momentum[1] + 2.0*state_plus.cv.momentum[1]
            mom_plus = make_obj_array([mom_x, mom_y])

            kin_energy_ref = 0.5/state_plus.cv.mass*np.dot(state_plus.cv.momentum, state_plus.cv.momentum)
            kin_energy_mod = 0.5/state_plus.cv.mass*np.dot(mom_plus, mom_plus)
            energy_plus = state_plus.cv.energy - kin_energy_ref + kin_energy_mod

            cv = make_conserved(dim=2, mass=state_plus.cv.mass,
                energy=energy_plus, momentum=mom_plus,
                species_mass=state_plus.cv.species_mass)

            return make_fluid_state(cv=cv, gas_model=gas_model,
                                    temperature_seed=300.0)

        def prescribed_state_for_diffusion(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
            return self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                       state_minus, **kwargs)

        def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                numerical_flux_func, **kwargs):

            state_plus = self.prescribed_state_for_advection(dcoll=dcoll,
                dd_bdry=dd_bdry, gas_model=gas_model, 
                state_minus=state_minus,**kwargs)

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

        def grad_cv_bc(self, state_plus, state_minus, grad_cv_minus, normal,
                       **kwargs):
            """Return grad(CV) for boundary calculation of viscous flux."""
            return grad_cv_minus

        def viscous_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
            grad_cv_minus, grad_t_minus, numerical_flux_func, **kwargs):
            """Return the boundary flux for viscous flux."""
            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            state_plus = self.prescribed_state_for_diffusion(dcoll=dcoll,
                dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, **kwargs)

            grad_cv_plus = self.grad_cv_bc(state_plus=state_plus,
                state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                normal=normal, **kwargs)

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
        dd_vol_fluid.trace("inlet").domain_tag:
            MyPrescribedBoundary(bnd_state_func=inlet_bnd_state_func, 
                                 temperature_func=bnd_temperature_func),
        dd_vol_fluid.trace("symmetry").domain_tag: AdiabaticSlipBoundary(),
        dd_vol_fluid.trace("burner").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("linear").domain_tag: linear_bnd,
        dd_vol_fluid.trace("outlet").domain_tag:
            PressureOutflowBoundary(boundary_pressure=101325.0),
    }

    # ~~~~~~~~~~
    sample_boundaries = {
        dd_vol_sample.trace("sample_sym").domain_tag: AdiabaticSlipBoundary()
    }

    # ~~~~~~~~~~
    holder_boundaries = {
        dd_vol_holder.trace("holder_sym").domain_tag: NeumannDiffusionBoundary(0.0)
    }

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    sample_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_sample)
    holder_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_holder)

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
        step, t, dt, fluid_state, sample_state, holder_state, smoothness=None, rhs=None):
        if rank == 0:
            print('Writing solution file...')

        fluid_viz_fields = [
            ("rho_g", fluid_state.cv.mass),
            ("rhoU_g", fluid_state.cv.momentum),
            ("rhoE_g", fluid_state.cv.energy),
            ("pressure", fluid_state.pressure),
            ("temperature", fluid_state.temperature),
            ("Vx", fluid_state.velocity[0]),
            ("Vy", fluid_state.velocity[1]),
            ("dt", dt[0] if local_dt else None),
#            ("sponge", sponge_sigma),
#            ("smoothness", 1.0 - theta_factor*smoothness),
#            ("RR", chem_rate*reaction_rates_damping),
#            ("heat_rls", heat_rls),
        ]

        # species mass fractions
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        write_visfile(dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t, overwrite=True, comm=comm)

        # ~~~ Sample
        wv = sample_state.wv 
        wall_mass = gas_model_sample.solid_density(sample_state.wv.material_densities)
        wall_conductivity = material.thermal_conductivity(
            temperature=sample_state.dv.temperature, tau=wv.tau)
        wall_heat_capacity = material.heat_capacity(
            temperature=sample_state.dv.temperature, tau=wv.tau)
        wall_heat_diffusivity = wall_conductivity/(wall_mass * wall_heat_capacity)            
        
        if my_material == "fiber":
            sample_mass_rhs, sample_source_O2, sample_source_CO2 = \
                decomposition.get_source_terms(
                    sample_state.temperature, wv.tau,
                    sample_state.cv.species_mass[cantera_soln.species_index("O2")])

            source_species = make_obj_array([sample_zeros for i in range(nspecies)])
            source_species[cantera_soln.species_index("O2")] = sample_source_O2
            source_species[cantera_soln.species_index("CO2")] = sample_source_CO2

            sample_source_gas = sample_source_O2 + sample_source_CO2

        if my_material == "composite":
            sample_mass_rhs = decomposition.get_source_terms(
                temperature=sample_state.temperature,
                chi=sample_state.wv.material_densities)
            
            source_species = make_obj_array([sample_zeros for i in range(nspecies)])
            source_species[cantera_soln.species_index("X2")] = -sum(sample_mass_rhs)

            sample_source_gas = -sum(sample_mass_rhs)

        sample_viz_fields = [
            ("rho_g", sample_state.cv.mass),
            ("rhoU_g", sample_state.cv.momentum),
            ("rhoE_b", sample_state.cv.energy),
            ("pressure", sample_state.pressure),
            ("temperature", sample_state.temperature),
            ("solid_mass", sample_state.wv.material_densities),
            ("viscosity", sample_state.tv.viscosity),
            ("kappa", sample_state.thermal_conductivity),
            ("alpha", wall_heat_diffusivity),
            ("ox_diff", sample_state.tv.species_diffusivity[1]),
            ("Vx", sample_state.velocity[0]),
            ("Vy", sample_state.velocity[1]),
            ("progress", wv.tau),
            ("dt", dt[2] if local_dt else None),
            ("source_solid", sample_mass_rhs),
#            ("source_O2", sample_source_O2),
#            ("source_CO2", sample_source_CO2),
        ]

        # species mass fractions
        sample_viz_fields.extend(
            ("Y_"+species_names[i], sample_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        write_visfile(dcoll, sample_viz_fields, sample_visualizer,
            vizname=vizname+"-sample", step=step, t=t, overwrite=True, comm=comm)

        # ~~~ Holder
        holder_viz_fields = [
            ("solid_mass", holder_state.cv.mass),
            ("rhoE_s", holder_state.cv.energy),
            ("temperature", holder_state.dv.temperature),
            ("kappa", holder_state.dv.thermal_conductivity),
            ("dt", dt[5] if local_dt else None),
        ]
        write_visfile(dcoll, holder_viz_fields, holder_visualizer,
            vizname=vizname+"-holder", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, fluid_state, sample_state, holder_state):
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
                "sample_density": sample_state.wv.material_densities,
                "sample_temperature_seed": sample_state.dv.temperature,
                "holder_cv": holder_state.cv,
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

    # create a list for the boundaries to be used for axisym sources
    from mirgecom.boundary import DummyBoundary

    fluid_field = fluid_zeros
    sample_field = sample_zeros
    holder_field = holder_zeros

    pairwise_field = {
        (dd_vol_fluid, dd_vol_holder): (fluid_field, holder_field)}
    pairwise_field_tpairs = inter_volume_trace_pairs(
        dcoll, pairwise_field, comm_tag=_MyGradTag_FH)
    field_tpairs_HF = pairwise_field_tpairs[dd_vol_holder, dd_vol_fluid]
#    field_tpairs_FH = pairwise_field_tpairs[dd_vol_fluid, dd_vol_holder]

    pairwise_field = {
        (dd_vol_fluid, dd_vol_sample): (fluid_field, sample_field)}
    pairwise_field_tpairs = inter_volume_trace_pairs(
        dcoll, pairwise_field, comm_tag=_MyGradTag_FS)
    field_tpairs_SF = pairwise_field_tpairs[dd_vol_sample, dd_vol_fluid]
    field_tpairs_FS = pairwise_field_tpairs[dd_vol_fluid, dd_vol_sample]

    pairwise_field = {
        (dd_vol_holder, dd_vol_sample): (holder_field, sample_field)}
    pairwise_field_tpairs = inter_volume_trace_pairs(
        dcoll, pairwise_field, comm_tag=_MyGradTag_SH)
#    field_tpairs_SH = pairwise_field_tpairs[dd_vol_sample, dd_vol_holder]
    field_tpairs_HS = pairwise_field_tpairs[dd_vol_holder, dd_vol_sample]

    axisym_fluid_boundaries = {}
    axisym_fluid_boundaries.update(fluid_boundaries)
    axisym_fluid_boundaries.update({
            tpair.dd.domain_tag: DummyBoundary()
            for tpair in field_tpairs_HF})
    axisym_fluid_boundaries.update({
            tpair.dd.domain_tag: DummyBoundary()
            for tpair in field_tpairs_SF})

    axisym_sample_boundaries = {}
    axisym_sample_boundaries.update(sample_boundaries)
    axisym_sample_boundaries.update({
            tpair.dd.domain_tag: DummyBoundary()
            for tpair in field_tpairs_FS})
    axisym_sample_boundaries.update({
            tpair.dd.domain_tag: DummyBoundary()
            for tpair in field_tpairs_HS})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from arraycontext import outer
    from grudge.trace_pair import (
        interior_trace_pairs, tracepair_with_discr_tag)
    from meshmode.discretization.connection import FACE_RESTR_ALL

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

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad,
                              flux_int)

        def boundary_flux(bdtag, bdry):
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = actx.thaw(dcoll.normal(dd_bdry_quad)) 
            int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field)

            if bnd_cond == 'symmetry' and bdtag  == '-0':
                ext_soln_quad = 0.0*int_soln_quad
            else:
                ext_soln_quad = 1.0*int_soln_quad

            bnd_tpair = TracePair(dd_bdry_quad,
                interior=int_soln_quad, exterior=ext_soln_quad)
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    off_axis_x = 1e-7
    fluid_nodes_are_off_axis = actx.np.greater(fluid_nodes[0], off_axis_x)
    sample_nodes_are_off_axis = actx.np.greater(sample_nodes[0], off_axis_x)
    holder_nodes_are_off_axis = actx.np.greater(holder_nodes[0], off_axis_x)
   
    def axisym_source_fluid(actx, dcoll, state, grad_cv, grad_t):
        cv = state.cv
        dv = state.dv
        
        mu = state.tv.viscosity
        beta = physical_transport.volume_viscosity(cv, dv, gas_model_fluid.eos)
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

        d2udr2   = my_derivative_function(actx, dcoll, dudr, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_F1)[0] #XXX
        d2vdr2   = my_derivative_function(actx, dcoll, dvdr, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_F2)[0] #XXX
        d2udrdy  = my_derivative_function(actx, dcoll, dudy, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_F3)[0] #XXX
                
        dmudr    = my_derivative_function(actx, dcoll,   mu, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_F4)[0]
        dbetadr  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_F5)[0]
        dbetady  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, 'replicate', _MyGradTag_F6)[1]
        
        qr_temp = - kappa*grad_t[0]
        qr_spec = + cv.mass*np.dot(dv.species_enthalpies*d_ij,grad_y)[0]
        dqrdr = 0.0 #- (dkappadr*grad_t[0] + kappa*d2Tdr2) #XXX
        
        dyidr = grad_y[:,0]
        #dyi2dr2 = my_derivative_function(actx, dcoll, dyidr, 'replicate')[:,0]   #XXX
        
        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              fluid_nodes_are_off_axis, u/fluid_nodes[0], dudr )

        dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

        #~~~~~~
        source_mass_dom = - cv.momentum[0]

        source_rhoU_dom = - cv.momentum[0]*u \
                          + tau_rr - tau_tt \
                          + u*dbetadr + beta*dudr \
                          + beta*actx.np.where(
                              fluid_nodes_are_off_axis, -u/fluid_nodes[0], -dudr )

        source_rhoV_dom = - cv.momentum[0]*v \
                          + tau_ry \
                          + u*dbetady + beta*dudy

        source_rhoE_dom = -( (cv.energy+dv.pressure)*u + (qr_temp + qr_spec) ) \
                          + u*tau_rr + v*tau_ry \
                          + u**2*dbetadr + beta*2.0*u*dudr \
                          + u*v*dbetady + u*beta*dvdy + v*beta*dudy

        source_spec_dom = - cv.species_mass*u + cv.mass*d_ij*dyidr

        #~~~~~~
        source_mass_sng = - drhoudr
        source_rhoU_sng = 0.0  # mu*d2udr2 + 0.5*beta*d2udr2  #XXX
        source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady

        # FIXME add species diffusion term
        source_rhoE_sng = -( (cv.energy+dv.pressure)*dudr + dqrdr ) \
                                + tau_rr*dudr + v*dtaurydr \
                                + 2.0*beta*dudr**2 \
                                + beta*dudr*dvdy \
                                + v*dudr*dbetady \
                                + v*beta*d2udrdy
        source_spec_sng = - cv.species_mass*dudr #+ cv.mass*d_ij*dyi2dr2

        #~~~~~~
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

    # ~~~~~~~~~~~~~~~
    def axisym_source_sample(actx, dcoll, state, grad_cv, grad_t):
        cv = state.cv
        dv = state.dv

        mu = state.tv.viscosity
        beta = physical_transport.volume_viscosity(cv, dv, gas_model_sample.eos)
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

        d2udr2   = my_derivative_function(actx, dcoll, dudr, sample_boundaries,
                                          dd_vol_sample, 'replicate', _MyGradTag_S1)[0] #XXX
        d2vdr2   = my_derivative_function(actx, dcoll, dvdr, sample_boundaries,
                                          dd_vol_sample, 'replicate', _MyGradTag_S2)[0] #XXX
        d2udrdy  = my_derivative_function(actx, dcoll, dudy, sample_boundaries,
                                          dd_vol_sample, 'replicate', _MyGradTag_S3)[0] #XXX

        dmudr    = my_derivative_function(actx, dcoll,   mu, sample_boundaries,
                                          dd_vol_sample, 'replicate', _MyGradTag_S4)[0]
        dbetadr  = my_derivative_function(actx, dcoll, beta, sample_boundaries,
                                          dd_vol_sample, 'replicate', _MyGradTag_S5)[0]
        dbetady  = my_derivative_function(actx, dcoll, beta, sample_boundaries,
                                          dd_vol_sample, 'replicate', _MyGradTag_S6)[1]

        qr_temp = - kappa*grad_t[0]
        qr_spec = + cv.mass*np.dot(dv.species_enthalpies*d_ij,grad_y)[0]
        dqrdr = 0.0 #- (dkappadr*grad_t[0] + kappa*d2Tdr2) #XXX

        dyidr = grad_y[:,0]
        #dyi2dr2 = my_derivative_function(actx, dcoll, dyidr, 'replicate')[:,0]   #XXX

        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              sample_nodes_are_off_axis, u/sample_nodes[0], dudr )

        dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

        #~~~~~~
        source_mass_dom = - cv.momentum[0]

        source_rhoU_dom = - cv.momentum[0]*u \
                          + tau_rr - tau_tt \
                          + u*dbetadr + beta*dudr \
                          + beta*actx.np.where(
                              sample_nodes_are_off_axis, -u/sample_nodes[0], -dudr )

        source_rhoV_dom = - cv.momentum[0]*v \
                          + tau_ry \
                          + u*dbetady + beta*dudy

        source_rhoE_dom = -( (cv.energy+dv.pressure)*u + (qr_temp + qr_spec) ) \
                          + u*tau_rr + v*tau_ry \
                          + u**2*dbetadr + beta*2.0*u*dudr \
                          + u*v*dbetady + u*beta*dvdy + v*beta*dudy

        source_spec_dom = - cv.species_mass*u + cv.mass*d_ij*dyidr

        #~~~~~~
        source_mass_sng = - drhoudr
        source_rhoU_sng = 0.0  # mu*d2udr2 + 0.5*beta*d2udr2  #XXX
        source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady

        # FIXME add species diffusion term
        source_rhoE_sng = -( (cv.energy+dv.pressure)*dudr + dqrdr ) \
                                + tau_rr*dudr + v*dtaurydr \
                                + 2.0*beta*dudr**2 \
                                + beta*dudr*dvdy \
                                + v*dudr*dbetady \
                                + v*beta*d2udrdy
        source_spec_sng = - cv.species_mass*dudr #+ cv.mass*d_ij*dyi2dr2
        
        #~~~~~~
        source_mass = actx.np.where( sample_nodes_are_off_axis,
                          source_mass_dom/sample_nodes[0], source_mass_sng )
        source_rhoU = actx.np.where( sample_nodes_are_off_axis,
                          source_rhoU_dom/sample_nodes[0], source_rhoU_sng )
        source_rhoV = actx.np.where( sample_nodes_are_off_axis,
                          source_rhoV_dom/sample_nodes[0], source_rhoV_sng )
        source_rhoE = actx.np.where( sample_nodes_are_off_axis,
                          source_rhoE_dom/sample_nodes[0], source_rhoE_sng )
        source_spec = make_obj_array([
                      actx.np.where( sample_nodes_are_off_axis,
                          source_spec_dom[i]/sample_nodes[0], source_spec_sng[i] )
                      for i in range(nspecies)])

        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                       momentum=make_obj_array([source_rhoU, source_rhoV]),
                       species_mass=source_spec)

#    compiled_axisym_source_sample = actx.compile(axisym_source_sample)

    # ~~~~~~~
    def axisym_source_holder(actx, dcoll, state, grad_t):
        dkappadr = 0.0*holder_nodes[0]
        
        kappa = state.dv.thermal_conductivity

        qr = - kappa*grad_t[0]
#        d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0], 
#                       axisym_wall_boundaries, dd_vol_holder, 'symmetry')[0]
#        dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)
                
        source_mass = 0.0*holder_nodes[0]

        source_rhoE_dom = - qr
        source_rhoE_sng = 0.0 #- dqrdr
        source_rhoE = actx.np.where( holder_nodes_are_off_axis,
                          source_rhoE_dom/holder_nodes[0], source_rhoE_sng )

        return SolidWallConservedVars(mass=source_mass, energy=source_rhoE)

#    compiled_axisym_source_holder = actx.compile(axisym_source_holder)

    # ~~~~~~~
    def gravity_source_terms(cv):
        """Gravity."""
        gravity = - 9.80665 * speedup_factor 
        delta_rho = cv.mass - rho_atmosphere
        zeros = actx.np.zeros_like(cv.mass)
        return make_conserved(dim=2,
            mass=zeros,
            energy=delta_rho * cv.velocity[1] * gravity,
            momentum=make_obj_array([zeros, delta_rho*gravity]),
            species_mass=make_obj_array([zeros for i in range(nspecies)])
        )

    # ~~~~~~~
    def chemical_source_term(cv, temperature):
        if solve_the_flame:
            return chem_rate * speedup_factor * reaction_rates_damping*(
                eos.get_species_source_terms(cv, temperature))
        else:
            return zeros

    # ~~~~~~

    from grudge.discretization import filter_part_boundaries
    from grudge.reductions import integral
    sample_dd_list = filter_part_boundaries(dcoll, volume_dd=dd_vol_sample,
                                           neighbor_volume_dd=dd_vol_fluid)
    interface_nodes = force_evaluation(actx, dcoll.nodes(sample_dd_list[0]))
    interface_zeros = actx.np.zeros_like(interface_nodes[0])

    integral_volume = integral(dcoll, dd_vol_sample, 2.0*np.pi*sample_nodes[0])
    integral_surface = integral(dcoll, sample_dd_list[0], 2.0*np.pi*interface_nodes[0])

    radius = 0.015875
    height = 0.01905

    volume = np.pi*radius**2*height
    area = np.pi*radius**2 + 2.0*np.pi*radius*height

#    assert integral_volume - volume < 1e-9
#    assert integral_surface - area < 1e-9

    print(integral_volume - volume)
    print(integral_surface - area)

    def blowing_velocity(cv, source):

        # volume integral of the source terms
        # dV = r dz dr dO, with O = 0 to 2pi = 2 pi r dz dr
        dV = 2.0*np.pi*sample_nodes[0]
        integral_volume_source = \
            integral(dcoll, dd_vol_sample, source*dV)

        # FIXME generalize this for multiple returned arguments in the list
        dd_list = filter_part_boundaries(dcoll, volume_dd=dd_vol_sample,
                                         neighbor_volume_dd=dd_vol_fluid)

        # restrict to coupled surface 
        surface_density = op.project(
            dcoll, dd_vol_sample, dd_list[0], cv.mass)                                   

        # surface integral of the density
        # dS = 2 pi r dx
        dS = 2.0*np.pi*interface_nodes[0]
        integral_surface_density = \
            integral(dcoll, dd_list[0], surface_density*dS)

        return integral_volume_source/integral_surface_density + interface_zeros

##############################################################################

    from grudge.dt_utils import characteristic_lengthscales
    char_length_fluid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)
    char_length_sample = characteristic_lengthscales(actx, dcoll, dd=dd_vol_sample)
    char_length_holder = characteristic_lengthscales(actx, dcoll, dd=dd_vol_holder)

    #~~~~~~~~~~

    from mirgecom.viscous import get_local_max_species_diffusivity
    from grudge.dof_desc import DD_VOLUME_ALL
    def _my_get_timestep_sample(state):
        return maximum_solid_dt + actx.np.zeros_like(state.cv.energy)

#        wv = state.wv
#        wall_mass = gas_model_sample.solid_density(state.wv.material_densities)
#        wall_conductivity = material.thermal_conductivity(
#            temperature=state.dv.temperature, tau=wv.tau)
#        wall_heat_capacity = material.heat_capacity(
#            temperature=state.dv.temperature, tau=wv.tau)

#        wall_heat_diffusivity = wall_conductivity/(wall_mass * wall_heat_capacity)
#        wall_spec_diffusivity = get_local_max_species_diffusivity(actx, state.tv.species_diffusivity)
#        wall_diffusivity = actx.np.maximum(wall_heat_diffusivity, wall_spec_diffusivity)

#        timestep = char_length_sample**2/(wall_time_scale * wall_diffusivity)

#        if not constant_cfl:
#            return dt

#        if local_dt:
#            mydt = maximum_cfl*timestep
#        else:
#            if constant_cfl:
#                mydt = actx.to_numpy(nodal_min(
#                    dcoll, dd_vol_sample, maximum_cfl*timestep, initial=np.inf))[()]

#        return mydt

    def _my_get_timestep_holder(state):
        return maximum_solid_dt + actx.np.zeros_like(state.cv.energy)

#        dv = state.dv

#        wall_mass = holder_wall_model.density()
#        wall_conductivity = holder_wall_model.thermal_conductivity(dv.temperature)
#        wall_heat_capacity = holder_wall_model.heat_capacity(dv.temperature)

#        wall_heat_diffusivity = wall_conductivity/(wall_mass * wall_heat_capacity)

#        timestep = char_length_holder**2/(wall_time_scale * wall_heat_diffusivity)

#        if not constant_cfl:
#            return dt

#        if local_dt:
#            mydt = maximum_cfl*timestep
#        else:
#            if constant_cfl:
#                mydt = actx.to_numpy(nodal_min(
#                    dcoll, dd_vol_holder, maximum_cfl*timestep, initial=np.inf))[()]

#        return mydt

    def _my_get_timestep_fluid(fluid_state, t, dt):

        if not constant_cfl:
            return dt

        return actx.np.minimum(
            maximum_fluid_dt,
            get_sim_timestep(dcoll, fluid_state, t, dt,
            maximum_cfl, gas_model_fluid, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd_vol_fluid)
        )

    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)
    my_get_timestep_sample = actx.compile(_my_get_timestep_sample)
    my_get_timestep_holder = actx.compile(_my_get_timestep_holder)

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

#        fluid_cv, fluid_tseed, \
#            sample_cv, sample_tseed, sample_density, \
#            holder_cv, _ = state
        fluid_cv, fluid_tseed, \
            sample_cv, sample_tseed, sample_density, holder_cv = state

        fluid_cv = force_evaluation(actx, fluid_cv)
        fluid_tseed = force_evaluation(actx, fluid_tseed)
        sample_cv = force_evaluation(actx, sample_cv)
        sample_tseed = force_evaluation(actx, sample_tseed)
        holder_cv = force_evaluation(actx, holder_cv)

        # ~~~
        # include both outflow and sponge in the outflow damping region
        smoothness = force_evaluation(actx,
            smooth_region + sponge_sigma/sponge_amp)
        fluid_cv = drop_order_cv(fluid_cv, smoothness, theta_factor)

        # get species-limited fluid state
        fluid_state = get_fluid_state(fluid_cv, fluid_tseed)
        fluid_cv = fluid_state.cv

        # ~~~
        # get filtered then species-limited solid state
        if use_soln_filter:
            sample_cv = filter_sample_cv_compiled(sample_cv)

        sample_state = get_sample_state(sample_cv, sample_density,
                                        sample_tseed)
        sample_cv = sample_state.cv

        # ~~~
        # construct species-limited solid state
        holder_state = get_holder_state(holder_cv)

#        # ~~~
#        if my_material == "fiber":
#            boundary_velocity = interface_zeros

#        if my_material == "composite":
#            sample_mass_rhs = decomposition.get_source_terms(
#                temperature=sample_state.temperature,
#                chi=sample_state.wv.material_densities)

#            sample_source_gas = -sum(sample_mass_rhs)

#            # the normal points towards the wall but the gas must leave the wall
#            # so we flip the sign here..
#            boundary_velocity = \
#                -1.0 * speedup_factor * blowing_velocity(sample_cv, sample_source_gas)

        if local_dt:
            t = force_evaluation(actx, t)

            dt_fluid = my_get_timestep_fluid(fluid_state, t[0], dt[0])
            dt_sample = my_get_timestep_sample(sample_state)
            dt_holder = my_get_timestep_holder(holder_state)

            dt = make_obj_array([dt_fluid, fluid_zeros,
                                 dt_sample, sample_zeros, dt_sample,
#                                 dt_holder, interface_zeros])
                                 dt_holder])
        else:
            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, maximum_cfl,
                                      t_final, constant_cfl, local_dt, dd_vol_fluid)

        try:
            state = make_obj_array([
                fluid_cv, fluid_state.temperature,
                sample_cv, sample_state.temperature, sample_state.wv.material_densities,
#                holder_cv, boundary_velocity])
                holder_cv])

            do_garbage = check_step(step=step, interval=ngarbage)
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

            if do_garbage:
                with gc_timer:
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

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
                #rhs = my_rhs(t, state)
                rhs = None
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    sample_state=sample_state, holder_state=holder_state,
                    smoothness=smoothness, rhs=rhs)

            if do_restart:
                my_write_restart(step, t, fluid_state, sample_state, holder_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                sample_state=sample_state, holder_state=holder_state,
                smoothness=smoothness)
            raise

        return state, dt

    def unfiltered_rhs(time, state):

        fluid_cv, fluid_tseed, \
        sample_cv, sample_tseed, sample_density, \
        holder_cv = state

#        fluid_cv, fluid_tseed, \
#        sample_cv, sample_tseed, sample_density, \
#        holder_cv, boundary_velocity = state

        # include both outflow and sponge in the damping region
        # apply outflow damping
        smoothness = smooth_region + sponge_sigma/sponge_amp
        fluid_cv = _drop_order_cv(fluid_cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model_fluid,
            temperature_seed=fluid_tseed,
            limiter_func=_limit_fluid_cv, limiter_dd=dd_vol_fluid
        )
        fluid_cv = fluid_state.cv

        # construct species-limited solid state
        if use_soln_filter:
            sample_cv = filter_sample_cv(sample_cv)

        sample_state = make_fluid_state(cv=sample_cv, gas_model=gas_model_sample,
            material_densities=sample_density, temperature_seed=sample_tseed,
            limiter_func=_limit_sample_cv, limiter_dd=dd_vol_sample
        )
        sample_cv = sample_state.cv

        # construct species-limited solid state
        holder_state = _get_holder_state(holder_cv)
        holder_cv = holder_state.cv

        # wall degradation
        if ignore_wall:
            sample_mass_rhs = sample_zeros*0.0

        else:
            tau = gas_model_sample.wall_eos.decomposition_progress(
                sample_state.wv.material_densities)

            if my_material == "fiber":
                sample_mass_rhs, sample_source_O2, sample_source_CO2 = \
                    decomposition.get_source_terms(
                        sample_state.temperature, tau,
                        sample_state.cv.species_mass[cantera_soln.species_index("O2")])

                source_species = make_obj_array([sample_zeros for i in range(nspecies)])
                source_species[cantera_soln.species_index("O2")] = sample_source_O2
                source_species[cantera_soln.species_index("CO2")] = sample_source_CO2

                sample_source_gas = sample_source_O2 + sample_source_CO2

            if my_material == "composite":
                sample_mass_rhs = decomposition.get_source_terms(
                    temperature=sample_state.temperature,
                    chi=sample_state.wv.material_densities)
                
                source_species = make_obj_array([sample_zeros for i in range(nspecies)])
                source_species[cantera_soln.species_index("X2")] = -sum(sample_mass_rhs)

                sample_source_gas = -sum(sample_mass_rhs)

            sample_heterogeneous_source = make_conserved(dim=2,
                mass=sample_source_gas,
                energy=sample_zeros,
                momentum=make_obj_array([sample_zeros, sample_zeros]),
                species_mass=source_species
            )

        #~~~~~~~~~~~~~

        #TODO: Wall emissivity as a function of the material

        #~~~~~~~~~~~~~

        if True:

            fluid_all_bnds_no_grad, sample_all_bnds_no_grad = \
                add_multiphysics_interface_boundaries_no_grad(
                    dcoll, dd_vol_fluid, dd_vol_sample,
                    gas_model_fluid, gas_model_sample,
                    fluid_state, sample_state,
                    fluid_boundaries, sample_boundaries,
                    limiter_func_fluid=_limit_fluid_cv,
                    limiter_func_wall=_limit_sample_cv,
                    interface_noslip=True,
                    interface_radiation=True,
                    # boundary_velocity=boundary_velocity
                    )

            fluid_all_bnds_no_grad, holder_all_bnds_no_grad = \
                add_thermal_interface_boundaries_no_grad(
                    dcoll, gas_model_fluid,
                    dd_vol_fluid, dd_vol_holder,
                    fluid_state, holder_state.dv.thermal_conductivity,
                    holder_state.dv.temperature,
                    fluid_all_bnds_no_grad, holder_boundaries,
                    interface_noslip=True,
                    interface_radiation=True)

            sample_all_bnds_no_grad, holder_all_bnds_no_grad = \
                add_thermal_interface_boundaries_no_grad(
                    dcoll, gas_model_sample,
                    dd_vol_sample, dd_vol_holder,
                    sample_state, holder_state.dv.thermal_conductivity,
                    holder_state.dv.temperature,
                    sample_all_bnds_no_grad, holder_all_bnds_no_grad,
                    interface_noslip=True,
                    interface_radiation=False)

        # ~~~~~~~~~~~~~~

        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model_fluid, fluid_all_bnds_no_grad,
            quadrature_tag, dd=dd_vol_fluid, comm_tag=_FluidOpStatesTag,
            limiter_func=_limit_fluid_cv)

        sample_operator_states_quad = make_operator_fluid_states(
            dcoll, sample_state, gas_model_sample, sample_all_bnds_no_grad,
            quadrature_tag, dd=dd_vol_sample, comm_tag=_WallOpStatesTag,
            limiter_func=_limit_sample_cv)

        # ~~~~~~~~~~~~~~

        # fluid grad CV
        fluid_grad_cv = grad_cv_operator(
            dcoll, gas_model_fluid, fluid_all_bnds_no_grad, fluid_state,
            quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradCVTag)

        # fluid grad T
        fluid_grad_temperature = grad_t_operator(
            dcoll, gas_model_fluid, fluid_all_bnds_no_grad, fluid_state,
            quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradTempTag)

        if True:

            # sample grad CV
            sample_grad_cv = grad_cv_operator(
                dcoll, gas_model_sample, sample_all_bnds_no_grad,
                sample_state,
                quadrature_tag=quadrature_tag, dd=dd_vol_sample,
                operator_states_quad=sample_operator_states_quad,
                comm_tag=_SampleGradCVTag)

            # sample grad T
            sample_grad_temperature = grad_t_operator(
                dcoll, gas_model_sample, sample_all_bnds_no_grad,
                sample_state,
                quadrature_tag=quadrature_tag, dd=dd_vol_sample,
                operator_states_quad=sample_operator_states_quad,
                comm_tag=_SampleGradTempTag)

            # holder grad T
            holder_grad_temperature = wall_grad_t_operator(
                dcoll, holder_state.dv.thermal_conductivity,
                holder_all_bnds_no_grad, holder_state.dv.temperature,
                quadrature_tag=quadrature_tag, dd=dd_vol_holder,
                comm_tag=_HolderGradTempTag)

        # ~~~~~~~~~~~~~~~~~

        if True:

            fluid_all_boundaries, sample_all_boundaries = \
                add_multiphysics_interface_boundaries(
                    dcoll, dd_vol_fluid, dd_vol_sample,
                    gas_model_fluid, gas_model_sample,
                    fluid_state, sample_state,
                    fluid_grad_cv, sample_grad_cv,
                    fluid_grad_temperature, sample_grad_temperature,
                    fluid_boundaries, sample_boundaries,
                    limiter_func_fluid=_limit_fluid_cv,
                    limiter_func_wall=_limit_sample_cv,
                    interface_noslip=True,
                    # boundary_velocity=boundary_velocity,
                    interface_radiation=True,
                    wall_emissivity=emissivity, sigma=5.67e-8,
                    ambient_temperature=300.0,
                    wall_penalty_amount=wall_penalty_amount)

            fluid_all_boundaries, holder_all_boundaries = \
                add_thermal_interface_boundaries(
                    dcoll, gas_model_fluid, dd_vol_fluid, dd_vol_holder,
                    fluid_state, holder_state.dv.thermal_conductivity,
                    holder_state.dv.temperature,
                    fluid_grad_temperature, holder_grad_temperature,
                    fluid_all_boundaries, holder_boundaries,
                    interface_noslip=True,
                    interface_radiation=True,
                    wall_emissivity=emissivity, sigma=5.67e-8,
                    ambient_temperature=300.0,
                    wall_penalty_amount=wall_penalty_amount)


        if ignore_wall is False:
            sample_all_boundaries, holder_all_boundaries = \
                add_thermal_interface_boundaries(
                    dcoll, gas_model_sample, dd_vol_sample, dd_vol_holder,
                    sample_state, holder_state.dv.thermal_conductivity,
                    holder_state.dv.temperature,
                    sample_grad_temperature, holder_grad_temperature,
                    sample_all_boundaries, holder_all_boundaries,
                    interface_noslip=True,
                    interface_radiation=False,
                    wall_penalty_amount=wall_penalty_amount)

        #~~~~~~~~~~~~~
        if ignore_wall:
            sample_rhs = sample_zeros*0.0
            sample_sources = sample_zeros*0.0

        else:
            sample_rhs = wall_time_scale * ns_operator(
                dcoll, gas_model_sample, sample_state, sample_all_boundaries,
                quadrature_tag=quadrature_tag, dd=dd_vol_sample,
                operator_states_quad=sample_operator_states_quad,
                grad_cv=sample_grad_cv, grad_t=sample_grad_temperature,
                comm_tag=_SampleOperatorTag, inviscid_terms_on=False)

            sample_sources = wall_time_scale * (
                #eos.get_species_source_terms(sample_cv, sample_state.temperature)
                + sample_heterogeneous_source
                + axisym_source_sample(actx, dcoll, sample_state, 
                                       sample_grad_cv,
                                       sample_grad_temperature))

        #~~~~~~~~~~~~~
        if ignore_wall:
            holder_rhs = holder_zeros*0.0
            holder_sources = holder_zeros*0.0

        else:

            holder_energy_rhs = wall_time_scale * diffusion_operator(
                dcoll, holder_state.dv.thermal_conductivity,
                holder_all_boundaries, holder_state.dv.temperature,
                penalty_amount=wall_penalty_amount,
                quadrature_tag=quadrature_tag, dd=dd_vol_holder,
                grad_u=holder_grad_temperature, comm_tag=_HolderOperatorTag)

            holder_rhs = SolidWallConservedVars(mass=holder_zeros,
                                                energy=holder_energy_rhs)

            holder_sources = wall_time_scale * axisym_source_holder(
                actx, dcoll, holder_state, holder_grad_temperature)

        #~~~~~~~~~~~~~
        fluid_rhs = ns_operator(
            dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries,
            quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
            comm_tag=_FluidOperatorTag, inviscid_terms_on=True)

        fluid_sources = (
            chemical_source_term(fluid_cv, fluid_state.temperature)
            + sponge_func(cv=fluid_cv, cv_ref=ref_cv, sigma=sponge_sigma)
            + gravity_source_terms(fluid_cv)
            + axisym_source_fluid(actx, dcoll, fluid_state,
                                  fluid_grad_cv, fluid_grad_temperature))

        #~~~~~~~~~~~~~

#        from pytato.analysis import get_max_node_depth
#        print(f"{get_max_node_depth(fluid_rhs.mass[0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.energy[0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.momentum[0][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.momentum[1][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[0][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[1][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[2][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[3][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[4][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[5][0])=}")
#        print(f"{get_max_node_depth(fluid_rhs.species_mass[6][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.mass[0])=}")
#        print(f"{get_max_node_depth(fluid_sources.energy[0])=}")
#        print(f"{get_max_node_depth(fluid_sources.momentum[0][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.momentum[1][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[0][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[1][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[2][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[3][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[4][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[5][0])=}")
#        print(f"{get_max_node_depth(fluid_sources.species_mass[6][0])=}")
#        print('')
#        print(f"{get_max_node_depth(sample_mass_rhs[0])=}")
#        print(f"{get_max_node_depth(sample_rhs.mass[0])=}")
#        print(f"{get_max_node_depth(sample_rhs.energy[0])=}")
#        print(f"{get_max_node_depth(sample_rhs.momentum[0][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.momentum[1][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.species_mass[0][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.species_mass[1][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.species_mass[2][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.species_mass[3][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.species_mass[4][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.species_mass[5][0])=}")
#        print(f"{get_max_node_depth(sample_rhs.species_mass[6][0])=}")
#        print(f"{get_max_node_depth(sample_sources.mass[0])=}")
#        print(f"{get_max_node_depth(sample_sources.energy[0])=}")
#        print(f"{get_max_node_depth(sample_sources.momentum[0][0])=}")
#        print(f"{get_max_node_depth(sample_sources.momentum[1][0])=}")
#        print(f"{get_max_node_depth(sample_sources.species_mass[0][0])=}")
#        print(f"{get_max_node_depth(sample_sources.species_mass[1][0])=}")
#        print(f"{get_max_node_depth(sample_sources.species_mass[2][0])=}")
#        print(f"{get_max_node_depth(sample_sources.species_mass[3][0])=}")
#        print(f"{get_max_node_depth(sample_sources.species_mass[4][0])=}")
#        print(f"{get_max_node_depth(sample_sources.species_mass[5][0])=}")
#        print(f"{get_max_node_depth(sample_sources.species_mass[6][0])=}")
#        print('')
#        print(f"{get_max_node_depth(holder_rhs.mass[0])=}")
#        print(f"{get_max_node_depth(holder_rhs.energy[0])=}")
#        print(f"{get_max_node_depth(holder_sources.mass[0])=}")
#        print(f"{get_max_node_depth(holder_sources.energy[0])=}")

        return make_obj_array([
            fluid_rhs + fluid_sources,
            fluid_zeros,
            sample_rhs + sample_sources,
            sample_zeros,
            sample_mass_rhs,
            holder_rhs + holder_sources,
#            interface_zeros
            ])

    unfiltered_rhs_compiled = actx.compile(unfiltered_rhs)

    def my_rhs(t, state):

        if use_rhs_filter is True:
            rhs_state = unfiltered_rhs_compiled(t, state)
            filtered_sample_rhs = filter_sample_rhs_compiled(rhs_state[2])
            filtered_sample_mass = filter_sample_rhs_compiled(rhs_state[4])

            return make_obj_array([
                rhs_state[0], fluid_zeros,
                filtered_sample_rhs, sample_zeros, filtered_sample_mass,
#                rhs_state[5], interface_zeros])
                rhs_state[5]])

        else:
            return unfiltered_rhs(t, state)

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

        if ignore_wall is True:
            min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
        else:
            min_dt = np.min(actx.to_numpy(dt[2])) if local_dt else dt
            min_dt = min_dt*wall_time_scale

        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([
        fluid_state.cv, fluid_state.temperature,
        sample_state.cv, sample_state.temperature, sample_state.wv.material_densities,
#        holder_state.cv, interface_zeros])
        holder_state.cv])

    if local_dt == True:
        dt_fluid = _my_get_timestep_fluid(
            fluid_state,
            current_t + fluid_zeros,
            maximum_fluid_dt + fluid_zeros)
        dt_sample = _my_get_timestep_sample(sample_state)
        dt_holder = _my_get_timestep_holder(holder_state)

        dt_fluid = force_evaluation(actx, dt_fluid)
        dt_sample = force_evaluation(actx, dt_sample)
        dt_holder = force_evaluation(actx, dt_holder)

        dt = make_obj_array([
#            dt_fluid, fluid_zeros, dt_sample, sample_zeros, dt_sample, dt_holder, interface_zeros])
            dt_fluid, fluid_zeros, dt_sample, sample_zeros, dt_sample, dt_holder])

        t_fluid = force_evaluation(actx, current_t + fluid_zeros)
        t_sample = force_evaluation(actx, current_t + sample_zeros)
        t_holder = force_evaluation(actx, current_t + holder_zeros)

        t = make_obj_array([
#            t_fluid, t_fluid, t_sample, t_sample, t_sample, t_holder, interface_zeros])
            t_fluid, t_fluid, t_sample, t_sample, t_sample, t_holder])
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
        advance_state(
            rhs=my_rhs, timestepper=timestepper,
            pre_step_callback=my_pre_step,
            post_step_callback=my_post_step,
            istep=current_step, dt=dt, t=t, t_final=t_final,
            max_steps=niter, local_dt=local_dt,
            force_eval=force_eval, state=stepper_state,
            compile_rhs=False if use_rhs_filter is True else True)

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
    parser = argparse.ArgumentParser(description="MIRGE-Com Driver")
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

    args = parser.parse_args()

    lazy = args.lazy

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

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    main(actx_class, use_logmgr=args.log, casename=casename, 
         restart_filename=restart_file)





#        def get_node_count(ary):
#            if not isinstance(ary, DOFArray):
#                from arraycontext import map_reduce_array_container
#                return map_reduce_array_container(sum, get_node_count, ary)

#            from pytato.analysis import get_num_nodes
#            return get_num_nodes(ary[0])


#        print(f"{get_node_count(fluid_cv)=}")
#        print(f"{get_node_count(fluid_state)=}")
##        print(f"{get_node_count(fluid_operator_states_quad)=}")
##        print(f"{get_node_count(fluid_all_bnds_no_grad)=}")
##        print(f"{get_node_count(fluid_all_boundaries)=}")        
##        print(f"{get_node_count(fluid_grad_cv)=}")
##        print(f"{get_node_count(fluid_grad_temperature)=}")

#        print(f"{get_node_count(fluid_rhs)=}")
#        print(f"{get_node_count(fluid_sources)=}")
#        print(f"{get_node_count(sample_rhs)=}")
#        print(f"{get_node_count(sample_sources)=}")
#        print(f"{get_node_count(sample_mass_rhs)=}")
#        print(f"{get_node_count(holder_rhs)=}")
#        print(f"{get_node_count(holder_sources)=}")

#        sys.exit()
