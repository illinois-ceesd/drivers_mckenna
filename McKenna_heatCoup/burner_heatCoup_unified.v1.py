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

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.utils import force_evaluation as force_eval
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
    LinearizedOutflow2DBoundary
)
from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient, make_conserved
)
from mirgecom.transport import MixtureAveragedTransport
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
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
from mirgecom.diffusion import (
    diffusion_operator,
    grad_operator as wall_grad_t_operator,
    NeumannDiffusionBoundary,
    grad_facial_flux_weighted, grad_facial_flux_central,
    diffusion_facial_flux_harmonic, diffusion_facial_flux_central
)
from mirgecom.wall_model import (
    SolidWallModel, SolidWallState, SolidWallConservedVars
)


#########################################################################

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


class Burner2D_Reactive:  # noqa

    def __init__(self, *, sigma, sigma_flame, flame_loc, pressure,
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
        self._flaLoc = flame_loc

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


class Pyrolysis:
    r"""Evaluate the source terms for the pyrolysis decomposition.

    The source terms follow as Arrhenius-like equation given by

    .. math::

        \dot{\omega}_i^p = \mathcal{A}_{i} T^{n_{i}}
        \exp\left(- \frac{E_{i}}{RT} \right)
        \left( \frac{\epsilon_i \rho_i -
            \epsilon^c_i \rho^c_i}{\epsilon^0_i \rho^0_i} \right)^{m_i}

    For TACOT, 2 different reactions, which are assumed to only happen after
    a minimum temperature, are considered based on the resin constituents.
    The third reaction is the fiber oxidation, which is not handled here for now.

    .. automethod:: get_source_terms
    """

    def __init__(self, virgin_mass, char_mass, fiber_mass, pre_exponential):
        """Temperature in which each reaction starts."""
        self._Tcrit = np.array([333.3, 555.6])
        self._virgin_mass = virgin_mass
        self._char_mass = char_mass
        self._fiber_mass = fiber_mass
        self._pre_exp = pre_exponential

        print("virgin mass", self._virgin_mass)
        print("virgin mass (2)", self._virgin_mass*0.75, self._virgin_mass*0.5, self._virgin_mass*0.25)
        print("char mass", self._char_mass)
        print("pre exp", self._pre_exp)

    def get_source_terms(self, temperature, chi):
        r"""Return the source terms of pyrolysis decomposition for TACOT.

        Parameters
        ----------
        temperature: :class:`~meshmode.dof_array.DOFArray`
            The temperature of the bulk material.

        chi: numpy.ndarray
            Either the solid mass $\rho_i$ of all fractions of the resin or
            the progress ratio $\chi$ of the decomposition. The actual
            parameter depends on the modeling itself.

        Returns
        -------
        source: numpy.ndarray
            The source terms for the pyrolysis
        """
        actx = temperature.array_context

        # The density parameters are hard-coded for TACOT. They depend on the
        # virgin and char volume fraction.
        return make_obj_array([
            # reaction 1
            actx.np.where(actx.np.less(temperature, self._Tcrit[0]),
                0.0, (
#                    -(30.*((chi[0] - 0.00)/30.)**3)*12000.
                    -(self._virgin_mass*0.25*((chi[0] - 0.00)/self._virgin_mass*0.25)**3)*self._pre_exp[0]
                    * actx.np.exp(-8556.000/temperature))),
            # reaction 2
            actx.np.where(actx.np.less(temperature, self._Tcrit[1]),
                0.0, (
#                    -(90.*((chi[1] - 60.0)/90.)**3)*4.48e9
                    -(self._virgin_mass*0.75*((chi[1] - self._virgin_mass*0.5)/self._virgin_mass*0.75)**3)*self._pre_exp[1]
                    * actx.np.exp(-20444.44/temperature))),
            # fiber oxidation: include in the RHS but dont do anything with it.
            actx.np.zeros_like(temperature)])


from mirgecom.wall_model import PorousWallEOS
class TacotEOS(PorousWallEOS):
    """Evaluate the properties of the solid state containing resin and fibers.

    A linear weighting between the virgin and chared states is applied to
    yield the material properties. Polynomials were generated offline to avoid
    interpolation and they are not valid for temperatures above 3200K.

    .. automethod:: void_fraction
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: volume_fraction
    .. automethod:: permeability
    .. automethod:: emissivity
    .. automethod:: tortuosity
    .. automethod:: decomposition_progress
    """

    def __init__(self, virgin_volume_fraction, char_volume_fraction,
                       fiber_volume_fraction, kappa0_virgin, kappa0_char,
                       h0_virgin, h0_char, char_mass, virgin_mass,
                       virgin_emissivity, char_emissivity):
        """Bulk density considering the porosity and intrinsic density.

        The fiber and all resin components must be considered.
        """
        self._char_mass = char_mass
        self._virgin_mass = virgin_mass
        self._fiber_volume_fraction = fiber_volume_fraction
        self._virgin_volume_fraction = virgin_volume_fraction
        self._char_volume_fraction = char_volume_fraction
        self._kappa0_virgin = kappa0_virgin
        self._kappa0_char = kappa0_char
        self._h0_virgin = h0_virgin
        self._h0_char = h0_char
        self._emissivity_virgin = virgin_emissivity
        self._emissivity_char = char_emissivity

        print("char mass:", self._char_mass)
        print("virgin mass:", self._virgin_mass)
        print("fiber frac:", self._fiber_volume_fraction)
        print("virgin frac:", self._virgin_volume_fraction)
        print("char frac:", self._char_volume_fraction)
        print("kappa_0 virgin:", self._kappa0_virgin)
        print("kappa_0 char:", self._kappa0_char)
        print("h0 virgin:", self._h0_virgin)
        print("h0 char:", self._h0_char)
        print("emissivity virgin:", self._emissivity_virgin)
        print("emissivity char:", self._emissivity_char)

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)

    def enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Solid enthalpy as a function of pyrolysis progress."""
        virgin = (
            - 1.360688853105e-11*temperature**5 + 1.521029626150e-07*temperature**4
            - 6.733769958659e-04*temperature**3 + 1.497082282729e+00*temperature**2
            + 3.009865156984e+02*temperature + self._h0_virgin)

        char = (
            - 1.279887694729e-11*temperature**5 + 1.491175465285e-07*temperature**4
            - 6.994595296860e-04*temperature**3 + 1.691564018109e+00*temperature**2
            - 3.441837408320e+01*temperature + self._h0_char)

        return virgin*tau + char*(1.0 - tau)

    def heat_capacity(self, temperature: DOFArray,
                      tau: DOFArray) -> DOFArray:
        r"""Solid heat capacity $C_{p_s}$ as a function of pyrolysis progress."""
        actx = temperature.array_context

        virgin = (
            - 6.803444000e-11*temperature**4 + 6.0841184e-07*temperature**3
            - 2.020131033e-03*temperature**2 + 2.9941644e+00*temperature**1
            + 3.0098651e+02)

        char = (
            - 6.39943845e-11*temperature**4 + 5.96470188e-07*temperature**3
            - 2.09837859e-03*temperature**2 + 3.38312804e+00*temperature**1
            - 3.44183741e+01)

        return virgin*tau + char*(1.0 - tau)

    def thermal_conductivity(self, temperature: DOFArray,
                             tau: DOFArray) -> DOFArray:
        """Solid thermal conductivity as a function of pyrolysis progress."""
        virgin = (
            + 2.31290019732353e-17*temperature**5 - 2.167785032562e-13*temperature**4
            + 8.24498395180905e-10*temperature**3 - 1.221612456223e-06*temperature**2
            + 8.46459266618945e-04*temperature + self._kappa0_virgin)

        char = (
            - 7.378279908877e-18*temperature**5 + 4.709353498411e-14*temperature**4
            + 1.530236899258e-11*temperature**3 - 2.305611352452e-07*temperature**2
            + 3.668624886569e-04*temperature + self._kappa0_char)

        return virgin*tau + char*(1.0 - tau)

    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        fiber = self._fiber_volume_fraction
        virgin = self._virgin_volume_fraction
        char = self._char_volume_fraction
        return virgin*tau + char*(1.0 - tau) + fiber

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the composite material."""
        virgin = 1.6e-11
        char = 2.0e-11
        return virgin*tau + char*(1.0 - tau)

    def emissivity(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Emissivity for energy radiation."""
        virgin = self._emissivity_virgin
        char = self._emissivity_char
        return virgin*tau + char*(1.0 - tau)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity."""
        virgin = 1.2
        char = 1.1
        return virgin*tau + char*(1.0 - tau)

    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the pyrolysis is locally complete and only charred
        material exists:

        .. math::
            \tau = \frac{\rho_0}{\rho_0 - \rho_c}
                    \left( 1 - \frac{\rho_c}{\rho(t)} \right)
        """
        char_mass = self._char_mass
        virgin_mass = self._virgin_mass
        return virgin_mass/(virgin_mass - char_mass)*(1.0 - char_mass/mass)


class No_Oxidation_Model():  # noqa N801
    def get_source_terms(self, temperature, **kwargs):
        return temperature*0.0


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
         restart_file=None, user_input_file=False):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    # my_material = "copper"
    # my_material = "fiber"
    # my_material = "composite"

    # width = 0.005
    # width = 0.010
    # width = 0.015
    # width = 0.020
    # width = 0.025

    ignore_wall = False

    # ~~~~~~~~~~~~~~~~~~

    # default i/o frequencies
    nviz = 25000
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "ssprk43"
    t_final = 2.0
    niter = 4000001
    local_dt = True
    constant_cfl = True
    current_cfl = 0.2

    # discretization and model control
    order = 2

    chem_rate = 1.0
    speedup_factor = 7.5

    equiv_ratio = 1.0
    total_flow_rate = 17.0
    # air_flow_rate = 18.8
    shroud_rate = 11.85
    prescribe_species = True

    width_mm = str('%02i' % (width*1000)) + "mm"
    if my_material == "copper":
        current_dt = 1.0e-7
        wall_time_scale = 100.0  # wall speed-up
        mechanism_file = "uiuc_7sp"
        solid_domains = ["solid"]

        if use_tpe:
            mesh_filename = f"mesh_v1_{width_mm}_020um_heatProbe_quads"
        else:
            mesh_filename = f"mesh_v1_{width_mm}_020um_heatProbe"

    else:
        current_dt = 1.0e-6
        wall_time_scale = 10.0*speedup_factor  # wall speed-up
        mechanism_file = "uiuc_8sp_phenol"
        solid_domains = ["wall_sample", "wall_alumina", "wall_graphite"] # XXX adiabatic

        if use_tpe:
            mesh_filename = f"mesh_13m_{width_mm}_015um_2domains_quads"
        else:
            mesh_filename = f"mesh_12m_{width_mm}_015um_2domains"

    temp_wall = 300.0
    wall_penalty_amount = 1.0
    use_radiation = True

    restart_iterations = False
    #restart_iterations = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # set up driver parameters
    from mirgecom.simutil import configurate
    from mirgecom.io import read_and_distribute_yaml_data
    input_data = read_and_distribute_yaml_data(comm, user_input_file)

    # heat conductivity at T=0K for polynomial fitting
    kappa0_virgin = configurate("kappa0_virgin", input_data, 2.38711269e-01)
    kappa0_char = configurate("kappa0_char", input_data, 3.12089881e-01)

    # reference enthalps at T=0K for polynomial fitting
    h0_virgin = configurate("h0_virgin", input_data, -1.0627680e+06)
    h0_char = configurate("h0_char", input_data, -1.23543810e+05)

    # wall emissivity for radiation
    virgin_wall_emissivity = configurate("virgin_wall_emissivity", input_data, 0.80)
    char_wall_emissivity = configurate("char_wall_emissivity", input_data, 0.90)

    # volume occupied by the solid components in the bulk volume
    virgin_volume_fraction = configurate("virgin_volume_fraction", input_data, 0.10)
    fiber_volume_fraction = configurate("fiber_volume_fraction", input_data, 0.10)
    char_volume_fraction = virgin_volume_fraction/2.0

    # Arrhenius parameters for pyrolysis
    pre_exponential = np.zeros(2,)
    pre_exponential[0] = configurate("pre_exponential_0", input_data, 12000.0)
    pre_exponential[1] = configurate("pre_exponential_1", input_data, 4.480e9)

    # https://www.azom.com/article.aspx?ArticleID=2850
    # https://matweb.com/search/DataSheet.aspx?MatGUID=9aebe83845c04c1db5126fada6f76f7e&ckck=1
    wall_copper_rho = 8920.0
    wall_copper_cp = 385.0
    wall_copper_kappa = 391.1

    # Average from https://www.azom.com/properties.aspx?ArticleID=52 for alumina
    wall_alumina_rho = 3500.0
    wall_alumina_cp = 700.0
    wall_alumina_kappa = 25.00

    # Average from https://www.azom.com/article.aspx?ArticleID=1630 for graphite
    # FIXME There is a table with the temperature-dependent data for graphite
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

    transport = "Mixture"

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
            volume_to_tags = {"fluid": ["fluid"], "solid": solid_domains}
            # XXX adiabatic
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
          volume_to_local_mesh_data["fluid"][0].nelements +
          volume_to_local_mesh_data["solid"][0].nelements)
    # XXX adiabatic

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order,
        tensor_product_elements=use_tpe)

    quadrature_tag = DISCR_TAG_BASE
#    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
#    if use_overintegration:
#        quadrature_tag = DISCR_TAG_QUAD
#    else:
#        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("solid"), DISCR_TAG_BASE)
    # XXX adiabatic

    if my_material != "copper":
        from mirgecom.utils import mask_from_elements
        wall_vol_discr = dcoll.discr_from_dd(dd_vol_solid)
        wall_tag_to_elements = volume_to_local_mesh_data["solid"][1]
        wall_sample_mask = mask_from_elements(
            dcoll, dd_vol_solid, actx, wall_tag_to_elements["wall_sample"])
        wall_alumina_mask = mask_from_elements(
            dcoll, dd_vol_solid, actx, wall_tag_to_elements["wall_alumina"])
        wall_graphite_mask = mask_from_elements(
            dcoll, dd_vol_solid, actx, wall_tag_to_elements["wall_graphite"])
    # XXX adiabatic

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))
    # XXX adiabatic

    fluid_zeros = force_eval(actx, fluid_nodes[0]*0.0)
    solid_zeros = force_eval(actx, solid_nodes[0]*0.0)
    # XXX adiabatic

    # ~~~~~~~~~~
    from grudge.dt_utils import characteristic_lengthscales
    char_length_fluid = force_eval(
        actx, characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid))
    char_length_solid = force_eval(
        actx, characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid))
    # XXX adiabatic

    # ~~~~~~~~~~
    if my_material == "copper":
        from grudge.discretization import filter_part_boundaries
        solid_dd_list = filter_part_boundaries(dcoll, volume_dd=dd_vol_solid,
                                               neighbor_volume_dd=dd_vol_fluid)

        interface_nodes = op.project(dcoll, dd_vol_solid,
                                     solid_dd_list[0], solid_nodes)
        interface_zeros = actx.np.zeros_like(interface_nodes[0])

    else:
        from mirgecom.multiphysics.phenolics_coupled_fluid_wall import (
            get_porous_domain_interface)
        interface_sample, interface_nodes, solid_dd_list = \
            get_porous_domain_interface(actx, dcoll, dd_vol_fluid,
                                        dd_vol_solid, wall_sample_mask)

        interface_zeros = actx.np.zeros_like(interface_nodes[0])

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

        print("volume = ", volume, integral_volume - volume)
        print("surface = ", area, integral_surface - area)

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

    mass_shroud = shroud_rate*1.0
    A_int = np.pi*r_int**2  # noqa
    A_ext = np.pi*(r_ext**2 - r_int**2)  # noqa
    lmin_to_m3s = 1.66667e-5
    u_int = flow_rate*lmin_to_m3s/A_int
    u_ext = mass_shroud*lmin_to_m3s/A_ext
    rhoU_int = rho_int*u_int  # noqa

    rho_ext = 101325.0/((8314.46/cantera_soln.molecular_weights[-1])*300.)
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
    print("ratio=", u_ext/u_int,"\n")

    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x_reference

    if prescribe_species:
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

    else:
        # ~~~ Reactants
        y_unburned = y_reference*1.0
        _, rho_unburned, y_unburned = cantera_soln.TDY

        # ~~~ Products

        cantera_soln.TPY = 1800.0, pres_unburned, y_unburned
        cantera_soln.equilibrate("TP")
        temp_burned, rho_burned, y_burned = cantera_soln.TDY

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

    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
                                cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 1234.56789
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    transport_model = MixtureAveragedTransport(pyrometheus_mechanism,
                                               factor=speedup_factor)

    gas_model = GasModel(eos=eos, transport=transport_model)

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

    if my_material == "copper":

        use_wv_tseed = False

        from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
            add_interface_boundaries, add_interface_boundaries_no_grad)

        emissivity = 0.03*speedup_factor

        def _solid_enthalpy_func(temperature, **kwargs):
            return wall_copper_cp*temperature

        def _solid_heat_capacity_func(temperature, **kwargs):
            return wall_copper_cp + solid_zeros

        def _solid_thermal_cond_func(temperature, **kwargs):
            return wall_copper_kappa + solid_zeros

        solid_wall_model = SolidWallModel(
            enthalpy_func=_solid_enthalpy_func,
            heat_capacity_func=_solid_heat_capacity_func,
            thermal_conductivity_func=_solid_thermal_cond_func)

    else:

        use_wv_tseed = True

        if my_material == "fiber":

            from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
                add_interface_boundaries, add_interface_boundaries_no_grad)

            wall_sample_density = 0.12*1400.0 + solid_zeros

            import mirgecom.materials.carbon_fiber as material_sample
            material = material_sample.FiberEOS(char_mass=0.0, virgin_mass=168.0)
            #decomposition = material_sample.Y3_Oxidation_Model(wall_material=material)
            decomposition = No_Oxidation_Model()

        if my_material == "composite":

            from mirgecom.multiphysics.phenolics_coupled_fluid_wall import (
                add_interface_boundaries, add_interface_boundaries_no_grad)

            wall_sample_density = np.empty((3,), dtype=object)

            wall_sample_density[0] = virgin_volume_fraction*1200.0*0.25 + solid_zeros
            wall_sample_density[1] = virgin_volume_fraction*1200.0*0.75 + solid_zeros
            wall_sample_density[2] = fiber_volume_fraction*1600.0 + solid_zeros

            char_mass = fiber_volume_fraction*1600.0 + virgin_volume_fraction*1200.0*0.5
            virgin_mass = fiber_volume_fraction*1600.0 + virgin_volume_fraction*1200.0*1.0
            material = TacotEOS(virgin_volume_fraction=virgin_volume_fraction,
                                char_volume_fraction=char_volume_fraction,
                                fiber_volume_fraction=fiber_volume_fraction,
                                kappa0_virgin=kappa0_virgin, kappa0_char=kappa0_char,
                                h0_virgin=h0_virgin, h0_char=h0_char,
                                char_mass=char_mass, virgin_mass=virgin_mass,
                                char_emissivity=char_wall_emissivity,
                                virgin_emissivity=virgin_wall_emissivity)
            decomposition = Pyrolysis(virgin_mass=virgin_volume_fraction*1200.0*1.0,
                                      char_mass=virgin_volume_fraction*1200.0*0.5,
                                      fiber_mass=fiber_volume_fraction*1600.0,
                                      pre_exponential=pre_exponential)

        if isinstance(wall_sample_density, DOFArray) is False:
            wall_alumina_density = actx.np.zeros_like(wall_sample_density)
            wall_alumina_density[-1] = wall_alumina_rho

            wall_graphite_density = actx.np.zeros_like(wall_sample_density)
            wall_graphite_density[-1] = wall_graphite_rho

        wall_densities = (
            wall_sample_density * wall_sample_mask
            + wall_alumina_rho * wall_alumina_mask
            + wall_graphite_rho * wall_graphite_mask)
    # XXX adiabatic

        def _get_solid_enthalpy(temperature, tau):
            wall_sample_h = material.enthalpy(temperature, tau)
            wall_alumina_h = wall_alumina_cp * temperature
            wall_graphite_h = wall_graphite_cp * temperature
            return (wall_sample_h * wall_sample_mask
                    + wall_alumina_h * wall_alumina_mask
                    + wall_graphite_h * wall_graphite_mask)
    # XXX adiabatic

        def _get_solid_heat_capacity(temperature, tau):
            wall_sample_cp = material.heat_capacity(temperature, tau)
            return (wall_sample_cp * wall_sample_mask
                    + wall_alumina_cp * wall_alumina_mask
                    + wall_graphite_cp * wall_graphite_mask)
    # XXX adiabatic

        def _get_solid_thermal_conductivity(temperature, tau):
            wall_sample_kappa = material.thermal_conductivity(temperature, tau)
            return (wall_sample_kappa * wall_sample_mask
                    + wall_alumina_kappa * wall_alumina_mask
                    + wall_graphite_kappa * wall_graphite_mask)
    # XXX adiabatic

        def _get_solid_decomposition_progress(mass):
            wall_sample_tau = material.decomposition_progress(mass)
            return (wall_sample_tau * wall_sample_mask
                    + 1.0 * wall_alumina_mask
                    + 1.0 * wall_graphite_mask)
    # XXX adiabatic

        def _get_emissivity(temperature, tau):
            wall_sample_emissivity = material.emissivity(temperature, tau)
            return speedup_factor*(
                wall_sample_emissivity * wall_sample_mask
                 + 0.00 * wall_alumina_mask
                 + 0.85 * wall_graphite_mask)
    # XXX adiabatic

        solid_wall_model = SolidWallModel(
            enthalpy_func=_get_solid_enthalpy,
            heat_capacity_func=_get_solid_heat_capacity,
            thermal_conductivity_func=_get_solid_thermal_conductivity,
            decomposition_func=_get_solid_decomposition_progress)
    # XXX adiabatic

    # }}}

#############################################################################

    def reaction_damping(dcoll, nodes, **kwargs):

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def smoothness_region(dcoll, nodes):
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                            momentum=mass_lim*cv.velocity,
                            species_mass=mass_lim*spec_lim)

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

    fluid_init = Burner2D_Reactive(sigma=0.00020,
        sigma_flame=0.000075, temperature=temp_burned, pressure=101325.0,
        flame_loc=0.10025, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

    ref_state = Burner2D_Reactive(sigma=0.00020,
        sigma_flame=0.00001, temperature=temp_burned, pressure=101325.0,
        flame_loc=0.1050, speedup_factor=speedup_factor,
        mass_rate_burner=rhoU_int, mass_rate_shroud=rhoU_ext,
        species_shroud=y_shroud, species_atm=y_atmosphere,
        species_unburn=y_unburned, species_burned=y_burned)

###############################################################################

    smooth_region = force_eval(actx, smoothness_region(dcoll, fluid_nodes))

    reaction_rates_damping = force_eval(
            actx, reaction_damping(dcoll, fluid_nodes))

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(
            cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid,
        )

    get_fluid_state = actx.compile(_get_fluid_state)

    def _get_porous_solid_state(wv, wv_tseed):
        wdv = solid_wall_model.dependent_vars(wv=wv, tseed=wv_tseed)
        return SolidWallState(cv=wv, dv=wdv)
    # XXX adiabatic

    def _get_copper_solid_state(cv):
        wdv = solid_wall_model.dependent_vars(cv)
        return SolidWallState(cv=cv, dv=wdv)
    # XXX adiabatic

    if my_material == "copper":
        get_solid_state = actx.compile(_get_copper_solid_state)
    else:
        get_solid_state = actx.compile(_get_porous_solid_state)
        

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        fluid_cv = fluid_init(fluid_nodes, eos, flow_rate=flow_rate,
                              prescribe_species=prescribe_species)
        tseed = force_eval(actx, 300.0 + fluid_zeros)
        wv_tseed = force_eval(actx, temp_wall + solid_zeros)

        if my_material == "copper":
            from mirgecom.materials.initializer import SolidWallInitializer
            solid_init = SolidWallInitializer(temperature=300.0,
                                              material_densities=wall_copper_rho)
            solid_cv = solid_init(solid_nodes, solid_wall_model)
        else:
    # XXX adiabatic
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

            solid_cv = SolidWallConservedVars(mass=wall_densities,
                                              energy=wall_energy)

        last_stored_time = -1.0
        my_file = open("temperature_file.dat", "w")
        my_file.close()

        my_file = open("massloss_file.dat", "w")
        my_file.close()

    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if np.isscalar(current_t) is False:
            if ignore_wall:
                current_t = np.min(actx.to_numpy(current_t[0]))
            else:
                current_t = np.min(actx.to_numpy(current_t[2]))

        if restart_iterations:
            current_t = 0.0
            current_step = 0

        data = np.genfromtxt("temperature_file.dat", delimiter=",")
        if data.shape == 2:
            last_stored_time = data[-1, 0]
        else:
            last_stored_time = -1.0  # sometimes the file only has 1 line...

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
            solid_cv = wall_connection(restart_data["wv"])
            if use_wv_tseed:
                wv_tseed = wall_connection(restart_data["wall_temperature_seed"])
    # XXX adiabatic
        else:
            fluid_cv = restart_data["cv"]
            tseed = restart_data["temperature_seed"]
            solid_cv = restart_data["wv"]
            if use_wv_tseed:
                wv_tseed = restart_data["wall_temperature_seed"]
    # XXX adiabatic

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    if my_material != "copper":

        tau = solid_wall_model.decomposition_progress(wall_sample_density)
        wall_mass = solid_wall_model.solid_density(wall_sample_density)

        initial_mass = integral(dcoll, dd_vol_solid,
                                wall_mass*wall_sample_mask*dV)

#########################################################################

    tseed = force_eval(actx, tseed)
    fluid_cv = force_eval(actx, fluid_cv)
    fluid_state = get_fluid_state(fluid_cv, tseed)

    solid_cv = force_eval(actx, solid_cv)
    if use_wv_tseed:
        wv_tseed = force_eval(actx, wv_tseed)
        solid_state = get_solid_state(solid_cv, wv_tseed)
    # XXX adiabatic
    else:
        solid_state = get_solid_state(solid_cv)

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
    ref_cv = force_eval(actx, ref_state(fluid_nodes, eos, flow_rate,
                                        state_minus=fluid_state,
                                        prescribe_species=prescribe_species))

##############################################################################

    inflow_nodes = force_eval(actx, dcoll.nodes(dd_vol_fluid.trace("inlet")))
    inflow_temperature = inflow_nodes[0]*0.0 + 300.0

    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return inflow_temperature

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos,
            flow_rate=flow_rate, state_minus=state_minus,
            prescribe_species=prescribe_species)
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
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

            cv = make_conserved(dim=2, mass=state_plus.cv.mass,
                                energy=energy_plus, momentum=mom_plus,
                                species_mass=state_plus.cv.species_mass)

            return make_fluid_state(cv=cv, gas_model=gas_model, temperature_seed=300.0)

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
                f_minus_normal=inviscid_flux(state_pair.int)@normal,
                f_plus_normal=inviscid_flux(state_pair.ext)@normal,
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

                # FIXME use "2 dudn^P - dudn^-" ???

                return grad_cv_minus.replace(
                    energy=grad_cv_minus*0.0,  # unused
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

    linear_bnd = LinearizedOutflow2DBoundary(
        dim=dim,
        free_stream_density=rho_atmosphere, free_stream_pressure=101325.0,
        free_stream_velocity=np.zeros(shape=(dim,)),
        free_stream_species_mass_fractions=y_atmosphere)

    fluid_boundaries = {
        dd_vol_fluid.trace("inlet").domain_tag:
            MyPrescribedBoundary(bnd_state_func=inlet_bnd_state_func,
                                 temperature_func=bnd_temperature_func),
        dd_vol_fluid.trace("symmetry").domain_tag:
            AdiabaticSlipBoundary(),
        dd_vol_fluid.trace("burner").domain_tag:
            AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("linear").domain_tag:
            linear_bnd,
        dd_vol_fluid.trace("outlet").domain_tag:
            PressureOutflowBoundary(boundary_pressure=101325.0),
    }

    wall_symmetry = NeumannDiffusionBoundary(0.0)
    solid_boundaries = {
        dd_vol_solid.trace("wall_sym").domain_tag: wall_symmetry
    }

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)
    # XXX adiabatic

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

        rho = fluid_state.cv.mass
        cp = eos.heat_capacity_cp(fluid_state.cv, fluid_state.temperature)
        fluid_viz_fields = [
            ("CV_rho", fluid_state.cv.mass),
            ("CV_rhoU", fluid_state.cv.momentum),
            ("CV_rhoE", fluid_state.cv.energy),
            ("DV_P", fluid_state.pressure),
            ("DV_T", fluid_state.temperature),
            ("DV_U", fluid_state.velocity[0]),
            ("DV_V", fluid_state.velocity[1]),
            ("sponge", sponge_sigma),
            ("smoothness", 1.0 - theta_factor*smoothness),
            # ("dt", dt[0] if local_dt else None),
            # ("mu", fluid_state.tv.viscosity),
            # ("alpha", fluid_state.tv.thermal_conductivity/(rho*cp)),
            # ("kappa", fluid_state.tv.thermal_conductivity),
            # ("Dij", fluid_state.tv.species_diffusivity[0]),
            # ("RR", chem_rate*reaction_rates_damping),
        ]

        # species mass fractions
        fluid_viz_fields.extend((
            "Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        # if grad_cv_fluid is not None:
        #     fluid_viz_fields.extend((
        #         ("fluid_grad_cv_rho", grad_cv_fluid.mass),
        #         ("fluid_grad_cv_rhoU", grad_cv_fluid.momentum[0]),
        #         ("fluid_grad_cv_rhoV", grad_cv_fluid.momentum[1]),
        #         ("fluid_grad_cv_rhoE", grad_cv_fluid.energy),
        #     ))

        # if grad_t_fluid is not None:
        #     fluid_viz_fields.append(("fluid_grad_t", grad_t_fluid))

        wv = solid_state.cv
        wdv = solid_state.dv
        # wall_alpha = solid_wall_model.thermal_diffusivity(
        #     mass=wv.mass, temperature=wdv.temperature,
        #     thermal_conductivity=wdv.thermal_conductivity)
        solid_viz_fields = [
            ("wv_energy", wv.energy),
            ("cfl", solid_zeros),  # FIXME
            ("wall_kappa", wdv.thermal_conductivity),
            ("wall_progress", wdv.tau),
            ("wall_temperature", wdv.temperature),
            ("wall_grad_t", grad_t_solid),
        ]

        if use_wv_tseed:
            solid_mass_rhs = decomposition.get_source_terms(
                temperature=wdv.temperature, chi=wv.mass)
        if wv.mass.shape[0] > 1:
            solid_viz_fields.extend(("wv_mass_" + str(i), wv.mass[i])
                                     for i in range(wv.mass.shape[0]))
            solid_viz_fields.extend(("mass_rhs" + str(i), solid_mass_rhs[i])
                                     for i in range(wv.mass.shape[0]))
        else:
            solid_viz_fields.append(("wv_mass", wv.mass))

        print("Writing solution file...")
        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t,
            overwrite=True, comm=comm)
        write_visfile(
            dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True, comm=comm)
    # XXX adiabatic

    def my_write_restart(step, t, state):
        if rank == 0:
            print("Writing restart file...")

        if my_material == "copper":
            cv, tseed, wv = state
        else:
            cv, tseed, wv, wv_tseed, _ = state

        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_file:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "nspecies": nspecies,
                "wv": wv,
                "wall_temperature_seed": wv_tseed if use_wv_tseed else None,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)
    # XXX adiabatic

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
    fluid_nodes_are_off_axis = actx.np.greater(fluid_nodes[0], off_axis_x)
    solid_nodes_are_off_axis = actx.np.greater(solid_nodes[0], off_axis_x)

    def axisym_source_fluid(actx, dcoll, state, grad_cv, grad_t):
        cv = state.cv
        dv = state.dv

        mu = state.tv.viscosity
        beta = transport_model.volume_viscosity(cv, dv, gas_model.eos)
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
                                          dd_vol_fluid, "replicate", _MyGradTag_1)[0] #XXX
        d2vdr2   = my_derivative_function(actx, dcoll, dvdr, fluid_boundaries,
                                          dd_vol_fluid, "replicate", _MyGradTag_2)[0] #XXX
        d2udrdy  = my_derivative_function(actx, dcoll, dudy, fluid_boundaries,
                                          dd_vol_fluid, "replicate", _MyGradTag_3)[0] #XXX
                
        dmudr    = my_derivative_function(actx, dcoll,   mu, fluid_boundaries,
                                          dd_vol_fluid, "replicate", _MyGradTag_4)[0]
        dbetadr  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, "replicate", _MyGradTag_5)[0]
        dbetady  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, "replicate", _MyGradTag_6)[1]
        
        qr = - (kappa*grad_t)[0] #FIXME add species enthalpy term
        dqrdr = 0.0 #- (dkappadr*grad_t[0] + kappa*d2Tdr2) #XXX
        
        dyidr = grad_y[:,0]
        #dyi2dr2 = my_derivative_function(actx, dcoll, dyidr, "replicate")[:,0]   #XXX
        
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
        return chem_rate*speedup_factor*reaction_rates_damping*(
            eos.get_species_source_terms(cv, temperature))

    # ~~~~~~
    if my_material != "copper":

        def blowing_momentum(source):

            # volume integral of the source terms
            integral_volume_source = \
                integral(dcoll, dd_vol_solid, source*wall_sample_mask*dV)

            # surface integral of the density
            integral_surface = \
                integral(dcoll, solid_dd_list[0], dS)

            momentum = \
                -1.0*integral_volume_source/integral_surface*interface_sample

            return force_eval(actx, momentum)


##############################################################################

    from grudge.op import nodal_min, nodal_max

    def vol_min(dd_vol, x):
        return actx.to_numpy(nodal_min(dcoll, dd_vol, x, initial=+np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(nodal_max(dcoll, dd_vol, x, initial=-np.inf))[()]

    def my_get_wall_timestep(solid_state):
        return current_dt + solid_zeros
#        wall_diffusivity = solid_wall_model.thermal_diffusivity(solid_state)
#        return char_length_solid**2/(wall_diffusivity)

    def _my_get_timestep_wall(solid_state, t, dt):
        return current_cfl*my_get_wall_timestep(solid_state)

    def _my_get_timestep_fluid(fluid_state, t, dt):
        return get_sim_timestep(
            dcoll, fluid_state, t, dt, current_cfl,
            constant_cfl=constant_cfl, local_dt=local_dt, fluid_dd=dd_vol_fluid)

    my_get_timestep_wall = actx.compile(_my_get_timestep_wall)
    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)

    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        if my_material == "copper":
            cv, tseed, wv = state
        else:
            cv, tseed, wv, wv_tseed, _ = state

        cv = force_eval(actx, cv)
        tseed = force_eval(actx, tseed)

        # include both outflow and sponge in the damping region
        smoothness = force_eval(actx, smooth_region + sponge_sigma/sponge_amp)

        # damp the outflow
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed)
        cv = fluid_state.cv

        # wall variables
        wv = force_eval(actx, wv)
        if my_material == "copper":
            solid_state = get_solid_state(wv)            
        else:
            wv_tseed = force_eval(actx, wv_tseed)
            solid_state = get_solid_state(wv, wv_tseed)
        wv = solid_state.cv
        wdv = solid_state.dv

        # wall approximated blowing
        if my_material == "copper":
            boundary_momentum = interface_zeros

        if my_material == "fiber":
            boundary_momentum = interface_zeros

        if my_material == "composite":
            solid_mass_rhs = decomposition.get_source_terms(
                temperature=solid_state.dv.temperature, chi=solid_state.cv.mass)

            boundary_momentum = \
                speedup_factor * blowing_momentum(-1.0*sum(solid_mass_rhs))

        t = force_eval(actx, t)
        dt_fluid = force_eval(actx, actx.np.minimum(
            current_dt, my_get_timestep_fluid(fluid_state, t[0], dt[0])))
#        dt_solid = force_eval(actx, actx.np.minimum(
#            1.0e-8, my_get_timestep_wall(solid_state, t[2], dt[2])))
        dt_solid = force_eval(actx, current_dt + solid_zeros)

        if my_material == "copper":
            dt = make_obj_array([dt_fluid, fluid_zeros, dt_solid])
        else:
            dt = make_obj_array([dt_fluid, fluid_zeros, dt_solid, solid_zeros,
                                 interface_zeros])

        try:
            if my_material == "copper":
                state = make_obj_array([
                    fluid_state.cv, fluid_state.dv.temperature,
                    solid_state.cv])          
            else:
                state = make_obj_array([
                    fluid_state.cv, fluid_state.dv.temperature,
                    solid_state.cv, solid_state.dv.temperature,
                    boundary_momentum])

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            t_elapsed = time.time() - t_start
            if t_shutdown - t_elapsed < 60.0:
                my_write_restart(step, t, state)
                sys.exit()

            ngarbage = 10
            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            if step % 1000 == 0:
                dd_centerline = dd_vol_solid.trace("wall_sym")
                temperature_centerline = op.project(
                    dcoll, dd_vol_solid, dd_centerline, solid_state.dv.temperature)
                min_temp_center = vol_min(dd_centerline, temperature_centerline)
                max_temp_center = vol_max(dd_centerline, temperature_centerline)
                max_temp = vol_max(dd_vol_solid, solid_state.dv.temperature)

                wall_time = np.max(actx.to_numpy(t[2]))
                if wall_time > last_stored_time:

                    # temperature
                    my_file = open("temperature_file.dat", "a")
                    my_file.write(f"{wall_time:.8f}, {min_temp_center:.8f}, {max_temp_center:.8f}, {max_temp:.8f} \n")
                    my_file.close()

                    if rank == 0:
                        logger.info(f"Temperature (center) = {max_temp_center:.8f} at t = {wall_time:.8f}")
                        logger.info(f"Temperature (edge) = {max_temp:.8f} at t = {wall_time:.8f}")

                    # mass loss
                    if my_material != "copper":

                        wall_mass = solid_wall_model.solid_density(solid_state.cv.mass)
                        sample_mass = integral(dcoll, dd_vol_solid,
                             wall_mass * wall_sample_mask * dV)
                        mass_loss = initial_mass - sample_mass

                        if rank == 0:
                            logger.info(f"Mass loss = {mass_loss} at t = {wall_time:.8f}")

                        my_file = open("massloss_file.dat", "a")
                        my_file.write(f"{wall_time:.8f}, {mass_loss} \n")
                        my_file.close()

            if do_health:
                ## FIXME warning in lazy compilation
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

            if do_restart:
                my_write_restart(step, t, state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                solid_state=solid_state, smoothness=smoothness)
            raise

        return state, dt

    def _get_rhs(t, state):

        if my_material == "copper":
            fluid_state, solid_state = state
        else:
            fluid_state, solid_state, boundary_momentum = state

        cv = fluid_state.cv
        wv = solid_state.cv
        wdv = solid_state.dv

        #~~~~~~~~~~~~~
        if my_material == "copper":
            fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
                add_interface_boundaries_no_grad(
                    dcoll, gas_model,
                    dd_vol_fluid, dd_vol_solid,
                    fluid_state, wdv.thermal_conductivity, wdv.temperature,
                    fluid_boundaries, solid_boundaries,
                    interface_noslip=True, interface_radiation=use_radiation)

        if my_material == "fiber":
            fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
                add_interface_boundaries_no_grad(
                    dcoll, gas_model,
                    dd_vol_fluid, dd_vol_solid,
                    fluid_state, wdv.thermal_conductivity, wdv.temperature,
                    fluid_boundaries, solid_boundaries)

        if my_material == "composite":
            fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
                add_interface_boundaries_no_grad(
                    dcoll, gas_model,
                    dd_vol_fluid, dd_vol_solid,
                    fluid_state, wdv.thermal_conductivity, wdv.temperature,
                    boundary_momentum, interface_sample,
                    fluid_boundaries, solid_boundaries)

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
            numerical_flux_func=grad_facial_flux_weighted,
            comm_tag=_SolidGradTempTag)

        if my_material == "copper":
            fluid_all_boundaries, solid_all_boundaries = \
                add_interface_boundaries(
                    dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
                    fluid_state, wdv.thermal_conductivity,
                    wdv.temperature,
                    fluid_grad_temperature, solid_grad_temperature,
                    fluid_boundaries, solid_boundaries,
                    interface_noslip=True, interface_radiation=use_radiation,
                    wall_emissivity=emissivity, sigma=5.67e-8,
                    ambient_temperature=300.0,
                    wall_penalty_amount=wall_penalty_amount)

        if my_material == "fiber":
            wall_emissivity = _get_emissivity(temperature=wdv.temperature,
                                              tau=wdv.tau)

            fluid_all_boundaries, solid_all_boundaries = \
                add_interface_boundaries(
                    dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
                    fluid_state, wdv.thermal_conductivity,
                    wdv.temperature,
                    fluid_grad_temperature, solid_grad_temperature,
                    fluid_boundaries, solid_boundaries,
                    wall_emissivity=wall_emissivity, sigma=5.67e-8,
                    ambient_temperature=300.0,
                    wall_penalty_amount=wall_penalty_amount)

        if my_material == "composite":
            wall_emissivity = _get_emissivity(temperature=wdv.temperature,
                                              tau=wdv.tau)

            fluid_all_boundaries, solid_all_boundaries = \
                add_interface_boundaries(
                    dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
                    fluid_state, wdv.thermal_conductivity,
                    wdv.temperature,
                    boundary_momentum, interface_sample,
                    fluid_grad_temperature, solid_grad_temperature,
                    fluid_boundaries, solid_boundaries,
                    wall_emissivity=wall_emissivity, sigma=5.67e-8,
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
                                  fluid_grad_temperature))

        #~~~~~~~~~~~~~
        if ignore_wall:
            solid_rhs = SolidWallConservedVars(mass=wv.mass*0.0,
                                               energy=solid_zeros)
            solid_sources = 0.0

            if my_material == "copper":
                return make_obj_array([fluid_rhs + fluid_sources, fluid_zeros,
                                       solid_rhs + solid_sources])
            else:
                return make_obj_array([fluid_rhs + fluid_sources, fluid_zeros,
                                       solid_rhs + solid_sources, solid_zeros,
                                       interface_zeros])

        else:
            if my_material == "copper":
                solid_energy_rhs = diffusion_operator(
                    dcoll, wdv.thermal_conductivity, solid_all_boundaries,
                    wdv.temperature,
                    penalty_amount=wall_penalty_amount,
                    quadrature_tag=quadrature_tag,
                    dd=dd_vol_solid,
                    grad_u=solid_grad_temperature,
                    comm_tag=_SolidOperatorTag)

                solid_sources = wall_time_scale * axisym_source_solid(
                        actx, dcoll, solid_state, solid_grad_temperature)

                solid_rhs = wall_time_scale * (
                    SolidWallConservedVars(mass=solid_zeros,
                                           energy=solid_energy_rhs))

                return make_obj_array([fluid_rhs + fluid_sources, fluid_zeros,
                                       solid_rhs + solid_sources])
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
                    comm_tag=_SolidOperatorTag,
                    diffusion_numerical_flux_func=diffusion_facial_flux_harmonic,
                )

                solid_sources = wall_time_scale * axisym_source_solid(
                    actx, dcoll, solid_state, solid_grad_temperature)

                solid_rhs = wall_time_scale * SolidWallConservedVars(
                    mass=solid_mass_rhs*wall_sample_mask,
                    energy=solid_energy_rhs)

                return make_obj_array([fluid_rhs + fluid_sources, fluid_zeros,
                                       solid_rhs + solid_sources, solid_zeros,
                                       interface_zeros])

        #~~~~~~~~~~~~~

    get_rhs_compiled = actx.compile(_get_rhs)

    def my_rhs(t, state):
        if my_material == "copper":
            cv, tseed, wv = state
        else:
            cv, tseed, wv, wv_tseed, boundary_momentum = state

        cv = force_eval(actx, cv)
        tseed = force_eval(actx, tseed)

        t = force_eval(actx, t)
        smoothness = force_eval(actx, smooth_region + sponge_sigma/sponge_amp)

        # apply outflow damping
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed)
        fluid_state = force_eval(actx, fluid_state)
        cv = fluid_state.cv

        # construct wall state
        wv = force_eval(actx, wv)
        if my_material == "copper":
            solid_state = get_solid_state(wv)
        else:
            wv_tseed = force_eval(actx, wv_tseed)
            solid_state = get_solid_state(wv, wv_tseed)

        solid_state = force_eval(actx, solid_state)
        wv = solid_state.cv
        wdv = solid_state.dv

        if my_material == "copper":
            actual_state = make_obj_array([fluid_state, solid_state])
        else:
            boundary_momentum = force_eval(actx, boundary_momentum)

            actual_state = make_obj_array([fluid_state, solid_state,
                                           boundary_momentum])

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
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections")
                gc.freeze()

        # min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
        min_dt = np.min(actx.to_numpy(dt[2])) if local_dt else dt
        min_dt = min_dt*wall_time_scale
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    dt_fluid = force_eval(
        actx, actx.np.minimum(
            current_dt, my_get_timestep_fluid(
                fluid_state,
                force_eval(actx, current_t + fluid_zeros),
                force_eval(actx, current_dt + fluid_zeros))))
    dt_solid = force_eval(actx, current_dt + solid_zeros)

    t_fluid = force_eval(actx, current_t + fluid_zeros)
    t_solid = force_eval(actx, current_t + solid_zeros)

    if my_material == "copper":
        stepper_state = make_obj_array([fluid_state.cv, fluid_state.dv.temperature,
                                        solid_state.cv])

        dt = make_obj_array([dt_fluid, fluid_zeros, dt_solid])
        t = make_obj_array([t_fluid, t_fluid, t_solid])
    else:
        stepper_state = make_obj_array([fluid_state.cv, fluid_state.dv.temperature,
                                        solid_state.cv, solid_state.dv.temperature,
                                        interface_zeros])

        dt = make_obj_array([dt_fluid, fluid_zeros, dt_solid, solid_zeros, interface_zeros])
        t = make_obj_array([t_fluid, t_fluid, t_solid, t_solid, interface_zeros])

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
 
    # final_cv, tseed, final_wv, wv_tseed = stepper_state
    # final_state = get_fluid_state(final_cv, tseed)
    # final_wdv = solid_wall_model.dependent_vars(final_wv, wv_tseed)

    # # Dump the final data
    # if rank == 0:
    #     logger.info("Checkpointing final state ...")

    # my_write_restart(step=final_step, t=final_t, state=stepper_state)

    # my_write_viz(step=final_step, t=final_t, dt=current_dt,
    #              cv=final_state.cv, dv=current_state.dv,
    #              wv=final_wv, wdv=final_wdv)

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

    main(actx_class, use_logmgr=args.log, casename=casename, use_tpe=args.tpe,
         restart_file=restart_file, user_input_file=input_file)
