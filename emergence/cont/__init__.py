"""Continuous substrates with no supplied finite ontology.

    continuous world  →  ???  →  discrete behavioural objects  →  possible algebra

The substrate is a controlled ODE ``dx/dt = f(x) + u(t)`` on R^n.  State,
time, controls and observations are continuous; numerical integration is an
approximation of the continuous system, and its timestep is varied as a
control.  The extractor receives only states, continuous controls and the
resulting future observations, and asks blindly whether some states are
behaviourally substitutable.  Names such as "well", "basin", "sign" or
"attractor" are evaluation language only and never enter extraction.
"""
