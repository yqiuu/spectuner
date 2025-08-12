The spectral line model
=======================
Spectuner implements the one-dimensional LTE spectral line model. For a single
velocity component, the spectrum of one vibration state is given by

.. math::

    \begin{aligned}
        J_\nu &= \eta_\nu(\theta, \nu) \left(S_\nu  - J^{bg}_\nu
        \right) \left(1 - e^{-\tau_\nu}\right), \label{eqn:slm_first} \\
        \eta(\theta, \nu) &= \frac{\theta^2}{\theta^2 + \theta^2_\text{T}(\nu)} \\
        S_\nu &= \frac{h\nu}{k} \frac{1}{e^\frac{h\nu}{kT} - 1} \\
        \tau_\nu &= \sum_t \tau^t_\nu, \\
        \tau^t_\nu &= \frac{c^2}{8\pi \nu^2}
        N_\text{tot} \frac{ A^t_{ul} g^t_u}{Q(T_\text{ex})} e^{-\frac{E^t_l}{k T_\text{ex}}} (1 - e^{-\frac{h\nu^t}{k T_\text{ex}}}) \phi^t_\nu,  \\
        \phi^t_\nu &= \frac{1}{\sqrt{2\pi}\sigma^t} \exp \left[ -\frac{1}{2} \left( \frac{\nu - \delta \nu^t}{\sigma^t}\right)^2  \right], \\
        \delta \nu^t &= \left( 1 - \frac{v_\text{offset}}{c} \right) \nu^t, \\
        \sigma^t &= \frac{1}{2 \sqrt{2 \ln 2}} \frac{\Delta v}{c} \delta \nu_t,
    \end{aligned}

where :math:`J^{bg}_\nu` is the background intensity,
:math:`c` is the speed of light and :math:`k` is the Boltzmann constant. For
single dish telescopes, the beam size is calculated by

.. math::
    \theta_\text{T} = 1.22 \frac{c}{\nu D} \frac{180}{\pi} \; \text{deg},

where :math:`D` is the diameter of the telescope. For interferometers, the
beam size is given by

.. math::
    \theta_\text{T} = \sqrt{\theta_\text{maj}\theta_\text{min}},

where :math:`\theta_\text{maj,min}` is the major (minor) axis of the synthesis
beam.

The model includes five fitting parameters:

- :math:`\theta`: Source size.
- :math:`T_\text{ex}`: Excitation temperature.
- :math:`N_\text{tot}`: Column density.
- :math:`\Delta v`: Velocity width.
- :math:`v_\text{offset}`: Velocity offset.

The following properties should be loaded from a spectroscopic database:

- :math:`\nu^t`: Transition frequency.
- :math:`A^t_\text{ul}`: Einstein A cofficient.
- :math:`g^t_u`: Upper state degeneracy.
- :math:`E^t_\text{l}`: Energy of the lower state.
- :math:`Q(T_\text{ex})`: Partition function.

Furthermore, the code takes into the instrumental resolution effect according
to Möller et al. (2017). The following integral is applied to computing the
output model spectrum:

.. math::
    J'(\nu) = \frac{1}{\Delta \nu_\text{c}} \int^{\nu + \Delta \nu_\text{c}/2}_{\nu - \Delta \nu_\text{c}/2} J(\nu') \, d\nu',

where :math:`\Delta \nu_\text{c}` is the channel width.


References
----------

* Möller, T., Endres, C., & Schilke, P. (2017), eXtended CASA Line Analysis
  Software Suite (XCLASS), Astronomy and Astrophysics, 598, A7.