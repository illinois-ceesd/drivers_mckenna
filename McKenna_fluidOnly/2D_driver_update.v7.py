"""Tue Sep  5 14:41:17 PDT 2023"""

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
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa

from functools import partial
from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array
from arraycontext import thaw, freeze
from meshmode.dof_array import DOFArray
#from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag, VolumeDomainTag
from grudge.dof_desc import DOFDesc, as_dofdesc, DISCR_TAG_BASE, VolumeDomainTag
from grudge.dof_desc import DD_VOLUME_ALL
from grudge.trace_pair import TracePair

from mirgecom.navierstokes import (
    grad_t_operator,
    grad_cv_operator,
    ns_operator)
from mirgecom.utils import force_evaluation, project_from_base
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    IsothermalWallBoundary,
    PressureOutflowBoundary,
    #PrescribedFluidBoundary,
    #AdiabaticNoslipWallBoundary,
    MengaldoBoundaryCondition,
    LinearizedOutflowBoundary,
    AdiabaticSlipBoundary)
from mirgecom.fluid import make_conserved
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    make_operator_fluid_states)
import cantera
from mirgecom.fluid import (
    make_conserved, 
    velocity_gradient,
    species_mass_fraction_gradient)
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage)

#########################################################################

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

class _FluidOperatorTag:
    pass

class _FluidOpStatesTag:
    pass

class _FluidGradCVTag:
    pass

class _FluidGradTempTag:
    pass

class Burner2D_Reactive:  # noqa

    def __init__(self, *, sigma, sigma_flame, pressure,
                 temperature, speedup_factor, mass_rate_burner, mass_rate_shroud,
                 species_unburn, species_burned, species_shroud, species_atm):

        self._sigma = sigma
        self._sigma_flame = sigma_flame
        self._pres = pressure
        self._speedup_factor = speedup_factor
        self._yu = species_unburn
        self._yb = species_burned
        self._ys = species_shroud
        self._ya = species_atm
        self._temp = temperature

        self._mass_rate_burner = mass_rate_burner
        self._mass_rate_shroud = mass_rate_shroud

    def __call__(self, x_vec, eos, flow_rate, state_minus=None,
                 prescribe_species=False, time=None):

        actx = x_vec[0].array_context

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

        int_diam = 2.38*25.4/2000.0  # radius, actually
        ext_diam = 2.89*25.4/2000.0  # radius, actually

        # ~~~ shroud
        shroud_1 = 0.5*(1.0 + actx.np.tanh(1.0/(_sigma)*(x_vec[0] - int_diam)))
        shroud_2 = actx.np.where(actx.np.greater(x_vec[0], ext_diam),
                                 0.0, - actx.np.tanh(1.0/(_sigma)*(x_vec[0] - ext_diam)))
        shroud = shroud_1*shroud_2

        # ~~~ flame ignition
        core = 0.5*(1.0 - actx.np.tanh(1.0/_sigma*(x_vec[0] - int_diam)))

        # ~~~ atmosphere
        atmosphere = 1.0 - (shroud + core)

        # ~~~ after combustion products
        upper_atm = 0.5*(1.0 + actx.np.tanh(1.0/(2.0*self._sigma)*(x_vec[1] - upper_bnd)))

        # ~~~ flame ignition
        #flame = 0.5*(1.0 + actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - self._flaLoc)))
        flame = actx.np.tanh(1.0/(self._sigma_flame)*(x_vec[1] - 0.1))

        # ~~~ species
        if prescribe_species:
            yf = (flame*_yb + (1.0-flame)*_yu)*(1.0-upper_atm) + _ya*upper_atm
            ys = _ya*upper_atm + (1.0 - upper_atm)*_ys
            y = atmosphere*_ya + shroud*ys + core*yf
        else:
            y = state_minus.cv.species_mass_fractions

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
        smoothY = mom_burner*(1.0-upper_atm) + 0.0*upper_atm  # noqa
        mom_x = 0.0*x_vec[0]
        mom_y = core*smoothY + shroud*mom_shroud*(1.0-upper_atm)
        momentum = make_obj_array([mom_x, mom_y])
        velocity = momentum/mass

        # ~~~
        specmass = mass * y

        # ~~~
        internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=2, mass=mass, energy=energy,
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

    mesh_filename = "mesh_v2_100um_noSample"

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 10000
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    plot_gradients = True

    # default timestepping control
    integrator = "ssprk43"
    current_dt = 1.0e-6
    t_final = 2.0
    niter = 3000001
    local_dt = True
    constant_cfl = True
    current_cfl = 0.4
    
    # discretization and model control
    order = 2
    use_overintegration = False

    x0_sponge = 0.15
    sponge_amp = 400.0
    theta_factor = 0.02

    mechanism_file = "uiuc_7sp"
    equiv_ratio = 1.0
    speedup_factor = 10.0
    chem_rate = 1.0
    total_flow_rate = 25.0
    Twall = 300.0
    solve_the_flame = True

    restart_iterations = False

##########################################################################

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

    # {{{ Set up initial state using Cantera

    # Use Cantera for initialization
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.

    air = "O2:0.21,N2:0.79"
    fuel = "C2H4:1"
    cantera_soln.set_equivalence_ratio(phi=equiv_ratio,
                                       fuel=fuel, oxidizer=air)
    temp_unburned = 300.0
    pres_unburned = 101325.0
    cantera_soln.TP = temp_unburned, pres_unburned
    x_reference = cantera_soln.X
    y_reference = cantera_soln.Y
    rho_int = cantera_soln.density

    r_int = 2.38*25.4/2000.0
    r_ext = 2.89*25.4/2000.0

    try:
        total_flow_rate
    except NameError:
        total_flow_rate = None

    if total_flow_rate is not None:
        flow_rate = total_flow_rate*1.0
    else:
        idx_fuel = cantera_soln.species_index("C2H4")
        flow_rate = air_flow_rate/(sum(x_reference) - x_reference[idx_fuel])

    A_int = np.pi*r_int**2  # noqa
    lmin_to_m3s = 1.66667e-5
    u_int = flow_rate*lmin_to_m3s/A_int
    rhoU_int = rho_int*u_int  # noqa

    #mass_shroud = shroud_rate*1.0
    A_ext = np.pi*(r_ext**2 - r_int**2)  # noqa
    u_ext = u_int #mass_shroud*lmin_to_m3s/A_ext
    rho_ext = 101325.0/((cantera.gas_constant/cantera_soln.molecular_weights[-1])*300.)
    # mdot_ext = rho_ext*u_ext*A_ext
    rhoU_ext = rho_ext*u_ext  # noqa

    width = 0.1
    width_mm = width*1000
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
    print("ratio=", u_ext/u_int,"\n")

    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x_reference

    # Pull temperature, density, mass fractions, and pressure from Cantera
    # set the mass flow rate at the inlet
    sim = cantera.ImpingingJet(gas=cantera_soln, width=width)
    sim.inlet.mdot = rhoU_int
    sim.set_refine_criteria(ratio=2, slope=0.1, curve=0.1, prune=0.0)
    sim.set_initial_guess(products='equil')
    sim.solve(loglevel=0, refine_grid=True, auto=True)

    # ~~~ Reactants
    assert np.absolute(sim.density[0]*sim.velocity[0] - rhoU_int) < 1e-11
    rho_unburned = sim.density[0]
    y_unburned = sim.Y[:,0]
    
    # ~~~ Products
    index_burned = np.argmax(sim.T) 
    temp_burned = sim.T[index_burned]
    rho_burned = sim.density[index_burned]
    y_burned = sim.Y[:,index_burned]

    # ~~~ Atmosphere
    x = np.zeros(nspecies)
    x[cantera_soln.species_index("O2")] = 0.21
    x[cantera_soln.species_index("N2")] = 0.79
    cantera_soln.TPX = temp_unburned, pres_unburned, x

    y_atmosphere = np.zeros(nspecies)
    dummy, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    temp_atmosphere, rho_atmosphere, y_atmosphere = cantera_soln.TDY

    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_shroud = y_atmosphere*0.0
    y_shroud[cantera_soln.species_index("N2")] = 1.0

    cantera_soln.TPY = 300.0, 101325.0, y_shroud
    temp_shroud, rho_shroud = cantera_soln.TD

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = \
        get_pyrometheus_wrapper_class_from_cantera(
             cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 300.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    species_names = pyrometheus_mechanism.species_names

    # }}}    

    # {{{ Initialize transport model
    from mirgecom.transport import MixtureAveragedTransport
    physical_transport = MixtureAveragedTransport(pyrometheus_mechanism,
                                                  factor=speedup_factor)

    gas_model = GasModel(eos=eos, transport=physical_transport)

    # }}}

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned:")
    print(f"T = {temp_unburned}")
    print(f"D = {rho_unburned}")
    print(f"Y = {y_unburned}")
    print(f" ")
    print(f"Burned:")
    print(f"T = {temp_burned}")
    print(f"D = {rho_burned}")
    print(f"Y = {y_burned}")
    print(f" ")
    print(f"Atmosphere:")
    print(f"T = {temp_atmosphere}")
    print(f"D = {rho_atmosphere}")
    print(f"Y = {y_atmosphere}")
    print(f" ")
    print(f"Shroud:")
    print(f"T = {temp_shroud}")
    print(f"D = {rho_shroud}")
    print(f"Y = {y_shroud}")

#############################################################################

    def reaction_damping(dcoll, field, **kwargs):

        actx = field.array_context
        nodes = force_evaluation(actx, dcoll.nodes())
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

    from mirgecom.artificial_viscosity import smoothness_indicator
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

    from mirgecom.limiter import bound_preserving_limiter

    from grudge.discretization import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_MODAL
    from meshmode.transform_metadata import FirstAxisIsElementsTag

    def drop_order(dcoll: DiscretizationCollection, field, theta,
                   positivity_preserving=False, dd=DD_VOLUME_ALL):
        actx = field.array_context

        # Compute cell averages of the state
        def cancel_polynomials(grp):
            return actx.from_numpy(np.asarray([1 if sum(mode_id) == 0
                                               else 0 for mode_id in grp.mode_ids()]))

        # map from nodal to modal
        if dd is None:
            dd = DD_VOLUME_ALL

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
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, 1e-13), cell_avgs, 1e-13)    

        return theta*(field - cell_avgs) + cell_avgs


    def _limit_fluid_cv(cv, temperature_seed, gas_model, dd=None):

        temperature = gas_model.eos.temperature(
            cv=cv, temperature_seed=temperature_seed)
        pressure = gas_model.eos.pressure(
            cv=cv, temperature=temperature)

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
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)


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
        sigma=0.00020, sigma_flame=0.00020,
        temperature=temp_burned, pressure=101325.0,
        speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

    ref_state = Burner2D_Reactive(
        sigma=0.00020, sigma_flame=0.00001,
        temperature=temp_burned, pressure=101325.0,
        speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

##############################################################################

    restart_step = None
    if restart_file is None:      

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0
  
        import os
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

        comm.Barrier()

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh2_path, force_ambient_dim=dim,
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
        first_step = restart_step+0

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

    zeros = nodes[0]*0.0
    ones = nodes[0]*0.0 + 1.0

#####################################################################################

    smooth_region = force_evaluation(actx, smoothness_region(dcoll, nodes[0]))

    reaction_rates_damping = force_evaluation(actx, reaction_damping(dcoll, nodes[0]))

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed,
            limiter_func=_limit_fluid_cv
        )

    get_fluid_state = actx.compile(_get_fluid_state)

#####################################################################################

    from grudge.op import nodal_min_loc, nodal_max_loc
    from grudge.op import nodal_min
    from grudge.op import nodal_max
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
        
#################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = fluid_init(nodes, eos, flow_rate=flow_rate,
                                prescribe_species=True)
    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t))

        if restart_iterations:
            current_t = 0.0
            current_step = 0

#        if restart_order != order:
#            restart_discr = EagerDGDiscretization(
#                actx,
#                local_mesh,
#                order=restart_order,
#                mpi_communicator=comm)
#            from meshmode.discretization.connection import make_same_mesh_connection
#            connection = make_same_mesh_connection(
#                actx,
#                dcoll.discr_from_dd("vol"),
#                restart_discr.discr_from_dd("vol"))

#            current_cv = connection(restart_data["state"])
#            tseed = connection(restart_data["temperature_seed"])
#        else:
        current_cv = restart_data["state"]
        tseed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)


    tseed = force_evaluation(actx, 300.0 + nodes[0]*0.0)
    current_cv = force_evaluation(actx, current_cv)

    current_state = get_fluid_state(current_cv, tseed)

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

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)

#####################################################################################

    # initialize the sponge field
    sponge_x_thickness = 0.055
    sponge_y_thickness = 0.055

    xMaxLoc = x0_sponge + sponge_x_thickness
    yMinLoc = 0.04

    sponge_init = InitSponge(amplitude=sponge_amp,
        x_max=xMaxLoc, y_min=yMinLoc,
        x_thickness=sponge_x_thickness,
        y_thickness=sponge_y_thickness
    )

    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=nodes))
    ref_cv = force_evaluation(actx, ref_state(nodes, eos, flow_rate,
                                              state_minus=current_state,
                                              prescribe_species=True))

#####################################################################################

    # {{{

    inflow_nodes = force_evaluation(actx, dcoll.nodes(dd_vol.trace("inlet")))
    inflow_temperature = inflow_nodes[0]*0.0 + 300.0

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos,
            flow_rate=flow_rate, state_minus=state_minus,
            prescribe_species=True)
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
                                temperature_seed=300.0)

    from mirgecom.inviscid import inviscid_flux
    from mirgecom.viscous import viscous_flux
    from mirgecom.flux import num_flux_central, num_flux_lfr

    class MyPrescribedBoundary(MengaldoBoundaryCondition):
        r"""Prescribed my boundary function."""

        def __init__(self, wall_temperature, boundary_state_func):
            """Initialize the boundary condition object."""
            self._wall_temp = wall_temperature
            self._bnd_state_func = boundary_state_func

        def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            state_bc = self._bnd_state_func(
                dcoll, dd_bdry, gas_model, state_minus, **kwargs)

            mom_x = -state_minus.cv.momentum[0]
            mom_y = 2.0*state_bc.cv.momentum[1] - state_minus.cv.momentum[1]
            mom_plus = make_obj_array([mom_x, mom_y])

            int_energy_plus = state_bc.cv.mass*eos.get_internal_energy(
                temperature=state_minus.temperature,
                species_mass_fractions=state_bc.species_mass_fractions)
            kin_energy_plus = 0.5*np.dot(mom_plus, mom_plus)/state_bc.cv.mass

            cv_plus = make_conserved(dim=2,
                                     mass=state_bc.cv.mass,
                                     energy=int_energy_plus+kin_energy_plus,
                                     momentum=mom_plus,
                                     species_mass=state_bc.cv.species_mass)

            return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                    temperature_seed=state_minus.temperature)

        def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            return self._bnd_state_func(dcoll, dd_bdry, gas_model,
                                        state_minus, **kwargs)

        def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
            """Get temperature value used in grad(T)."""
            actx = state_minus.array_context
            wall_temp = project_from_base(dcoll, dd_bdry, self._wall_temp)
            return actx.np.zeros_like(state_minus.temperature) + wall_temp

        def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus,
                       grad_cv_minus, normal, **kwargs):
            """Return grad(CV) for boundary calculation of viscous flux."""
            return grad_cv_minus

        def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal,
                                **kwargs):
            """Return BC on grad(temperature)."""
            # Mengaldo Eqns (50-51)
            return grad_t_minus

        def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                               numerical_flux_func, **kwargs):

            state_plus = self.state_plus(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, **kwargs)

            state_pair = TracePair(dd_bdry, interior=state_minus,
                                   exterior=state_plus)

    # }}}

    linear_bnd = LinearizedOutflowBoundary(free_stream_density=rho_atmosphere,
        free_stream_pressure=101325.0, free_stream_velocity=np.zeros(shape=(dim,)),
        free_stream_species_mass_fractions=y_atmosphere)
    
    boundaries = {
        BoundaryDomainTag("inlet"): MyPrescribedBoundary(boundary_state_func=inlet_bnd_state_func,
                                                         wall_temperature=300.0),
        BoundaryDomainTag("symmetry"): AdiabaticSlipBoundary(),
        BoundaryDomainTag("outlet"): PressureOutflowBoundary(boundary_pressure=101325.0),
        BoundaryDomainTag("linear"): linear_bnd,
        BoundaryDomainTag("burner"): IsothermalWallBoundary(wall_temperature=300.0)
    }

####################################################################################

    visualizer = make_visualizer(dcoll)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(
        dim=dim, order=order, nelements=local_nelements,
        global_nelements=global_nelements, dt=current_dt, t_final=t_final,
        nstatus=nstatus, nviz=nviz, t_initial=current_t, cfl=current_cfl,
        constant_cfl=constant_cfl, initname=initname, eosname=eosname,
        casename=casename)
    if rank == 0:
        logger.info(init_message)

#########################################################################

    from grudge.dt_utils import characteristic_lengthscales, dt_geometric_factors
    def my_write_viz(step, t, state, dt=None, smoothness=None):

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
                      ("RR", chem_rate*reaction_rates_damping),
                        ]

        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], state.cv.species_mass_fractions[i])
                for i in range(nspecies))
                      
        print('Writing solution file...')
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_filename:
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

    from arraycontext import outer
    from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
    from grudge import op
    from meshmode.discretization.connection import FACE_RESTR_ALL
    from mirgecom.flux import num_flux_central

    def my_derivative_function(actx, dcoll, field, field_bounds, bnd_cond, comm_tag, dd_vol=DD_VOLUME_ALL):    

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
                (sum(interior_flux(u_tpair) for u_tpair in interior_trace_pairs(dcoll, field, volume_dd=dd_vol, comm_tag=comm_tag))
                + sum(boundary_flux(bdtag, bdry) for bdtag, bdry in field_bounds.items())
                )
            )
        )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.fluid import velocity_gradient, species_mass_fraction_gradient

    off_axis_x = 1e-7
    nodes_are_off_axis = actx.np.greater(nodes[0], off_axis_x)  # noqa
   
    def axisym_source_terms(actx, dcoll, state, grad_cv, grad_t):      
        cv = state.cv
        dv = state.dv
        
        mu = state.tv.viscosity
        beta = gas_model.transport.volume_viscosity(cv, dv, eos)
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

        d2udr2   = my_derivative_function(actx, dcoll,  dudr, boundaries, 'replicate', comm_tag=_MyGradTag_1)[0]
        d2vdr2   = my_derivative_function(actx, dcoll,  dvdr, boundaries, 'replicate', comm_tag=_MyGradTag_2)[0]
        d2udrdy  = my_derivative_function(actx, dcoll,  dudy, boundaries, 'replicate', comm_tag=_MyGradTag_3)[0]
                
        dmudr    = my_derivative_function(actx, dcoll,    mu, boundaries, 'replicate', comm_tag=_MyGradTag_4)[0]
        dbetadr  = my_derivative_function(actx, dcoll,  beta, boundaries, 'replicate', comm_tag=_MyGradTag_5)[0]
        dbetady  = my_derivative_function(actx, dcoll,  beta, boundaries, 'replicate', comm_tag=_MyGradTag_6)[1]
        
        qr = - kappa*grad_t[0]
        dqrdr = 0.0

        dyidr = grad_y[:,0]
#        dyi2dr2 = my_derivative_function(actx, dcoll,     dyidr, 'replicate')[:,0]   #XXX
        
        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              nodes_are_off_axis, u/nodes[0], dudr )

        dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

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
        
        # FIXME add species diffusion term
        source_rhoE_dom = -( (cv.energy+dv.pressure)*u + qr ) \
                          + u*tau_rr + v*tau_ry \
                          + u**2*dbetadr + beta*2.0*u*dudr \
                          + u*v*dbetady + u*beta*dvdy + v*beta*dudy

        source_spec_dom = - cv.species_mass*u + cv.mass*d_ij*dyidr
        """
        """

        source_mass_sng = - drhoudr
        source_rhoU_sng = 0.0
        source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady
        # FIXME add species diffusion term
        source_rhoE_sng = -( (cv.energy+dv.pressure)*dudr + dqrdr ) \
                                + tau_rr*dudr + v*dtaurydr \
                                + 2.0*beta*dudr**2 \
                                + beta*dudr*dvdy \
                                + v*dudr*dbetady \
                                + v*beta*d2udrdy
        source_spec_sng = - cv.species_mass*dudr #+ d_ij*dyi2dr2
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


    # ~~~~~~~
    def gravity_source_terms(cv):
        """Gravity."""
        gravity = - 9.80665 * speedup_factor**2
        delta_rho = cv.mass - rho_atmosphere
        return make_conserved(
            dim=2,
            mass=zeros,
            energy=delta_rho*cv.velocity[1]*gravity,
            momentum=make_obj_array([zeros, delta_rho*gravity]),
            species_mass=cv.species_mass*0.0)


    def chemical_source_term(cv, temperature):
        return chem_rate*speedup_factor*reaction_rates_damping*(
            eos.get_species_source_terms(cv, temperature))

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed = state
        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)

        smoothness = smooth_region + sponge_sigma/sponge_amp #*smoothness_indicator(dcoll, cv.mass)
        cv = drop_order_cv(cv, smoothness, theta_factor)

        fluid_state = get_fluid_state(cv, tseed)

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

            ngarbage = 10
            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
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
                health_errors = global_reduce(my_health_check(
                    fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=fluid_state.cv,
                                 tseed=fluid_state.temperature)

            if do_viz:               
                my_write_viz(step=step, t=t, state=fluid_state, dt=dt,
                     smoothness=smoothness)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state, dt=dt, smoothness=smoothness)
            raise

        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt

    def my_rhs(t, state):
        cv, tseed = state

        smoothness = smooth_region + sponge_sigma/sponge_amp

        cv = _drop_order_cv(cv, smoothness, theta_factor)

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=tseed, limiter_func=_limit_fluid_cv)

        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries,
            quadrature_tag, dd=dd_vol, comm_tag=_FluidOpStatesTag,
            limiter_func=_limit_fluid_cv)

        # fluid grad CV
        grad_cv = grad_cv_operator(
            dcoll, gas_model, boundaries, fluid_state,
            dd=dd_vol, operator_states_quad=fluid_operator_states_quad,
            quadrature_tag=quadrature_tag, comm_tag=_FluidGradCVTag)

        # fluid grad T
        grad_t = grad_t_operator(
            dcoll, gas_model, boundaries, fluid_state,
            dd=dd_vol, operator_states_quad=fluid_operator_states_quad,
            quadrature_tag=quadrature_tag, comm_tag=_FluidGradTempTag)

        ns_rhs = ns_operator(
            dcoll, state=fluid_state, time=t, boundaries=boundaries,
            gas_model=gas_model, grad_cv=grad_cv, grad_t=grad_t,
            operator_states_quad=fluid_operator_states_quad,
            quadrature_tag=quadrature_tag, comm_tag=_FluidOperatorTag)

        sources = (
            chemical_source_term(cv, fluid_state.temperature)
            + gravity_source_terms(cv)
            + axisym_source_terms(actx, dcoll, fluid_state, grad_cv, grad_t)
            + sponge_func(cv=cv, cv_ref=ref_cv, sigma=sponge_sigma))
        
        return make_obj_array([ns_rhs + sources, fluid_state.temperature*0.0])

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

        min_dt = np.min(actx.to_numpy(dt)) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)         
            logmgr.tick_after()

        return state, dt

##############################################################################

    if local_dt == True:
        dt = force_evaluation(actx, actx.np.minimum(
             current_dt,
             get_sim_timestep(dcoll, current_state, current_t,
                     current_dt, current_cfl, gas_model,
                     constant_cfl=constant_cfl, local_dt=local_dt))
        )
        t = force_evaluation(actx, current_t + zeros)
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(
            rhs=my_rhs, timestepper=timestepper,
            pre_step_callback=my_pre_step, post_step_callback=my_post_step,
            state=make_obj_array([current_state.cv, tseed]),
            dt=dt, t_final=t_final, t=t, max_steps=niter, local_dt=local_dt,
            istep=current_step)

    current_cv, tseed = stepper_state

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_state = make_fluid_state(current_cv, gas_model, tseed)

    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     tseed=tseed)

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
