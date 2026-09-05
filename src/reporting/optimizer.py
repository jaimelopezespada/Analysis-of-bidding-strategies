"""Shared Pyomo/HiGHS MILP builder for order-type strategies.

Only SCO genuinely needs a per-scenario solver: given a fixed candidate theta
(P^V, TF, mav_frac), the thesis's own SCO equations leave real decision
freedom in whether each period t is *committed* (a genuine per-period binary
w_t^s, independent of whether t is in-the-money or out-of-the-money relative
to P^V) and how much to dispatch once committed (q_t^s in [MAV_t, Qbar_t]),
jointly with whether the day-level Minimum Income Condition is met — and,
once technical_min is activated, whether a feasible dispatch path even
exists. That coupling cannot be resolved in closed form in general, so it is
solved as a single MILP spanning all scenarios at once.

Two clearing models are available (RunConfig.sco_model):
- "aware" (default): clearing maximises the DECLARED surplus
  sum (lambda - P^V) * q — exactly the information EUPHEMIA has (P^V, TF,
  MAV). Private costs (C, C^SU) and CVaR only enter afterwards, on the fixed
  dispatch, in the strategy layer — mirroring SBO/Simple. That objective is
  separable per hour, so the production path solves it in CLOSED FORM
  (orders/sco.py::_clear_sco_aware, ~4 orders of magnitude faster);
  build_sco_model_aware below encodes the same model as a MILP and is kept
  only as a cross-check for tests.
- "naive" (legacy benchmark): the MILP maximises the REAL risk-adjusted
  profit (matching eq. 19: theta fixed, all S scenarios, and the CVaR
  eta/z_s variables in one program), i.e. the market clears with perfect
  knowledge of the generator's private costs — an upper bound the real
  market cannot replicate. Only this model genuinely needs the solver
  (OTM dispatch bounded by the MIC and CVaR couple the decisions).

Note: an out-of-the-money period (P^V > lambda_t^s) is *not* structurally
excluded from dispatch. It can still be committed and cleared for at least
MAV_t, exactly like an in-the-money period, as long as the aggregate
Minimum Income Condition (mic_rule, computed over all t using P^V) still
holds once that period's negative contribution is included. The only gate
on OTM dispatch is that aggregate condition — never the period's own
moneyness.

Only technical_min (Q_min) is modelled as a real constraint here, not
ramp/gradient. The thesis's own "nueva tipología" (SBO, SCO, EXBO, LSBO) has
no bid parameter through which an agent can express a ramp limit — gradient
("condición de gradiente de carga") was a feature of the pre-2025 OMIE
*ofertas complejas clásicas* (see "Antigua tipología"), not of the harmonised
SDAC products this thesis studies. EUPHEMIA itself does not enforce
inter-period ramp when clearing these order types, so modelling it here would
constrain the optimizer more than the real market does. ``ramp_limit``
remains a TechnologyConfig field purely for the technology's qualitative
techno-economic characterisation (Tabla firma de restricciones), with no
effect on dispatch.

Simple orders are, by the thesis's own eq. 1, a pure per-period price-clears-
or-not rule with no day-level acceptance binary — there is no lever within
that product to respect an intertemporal Q_min constraint (this is exactly
the motivating example in the thesis's introduction: a simple order is blind
to whole-day economics). Technical constraints are therefore not applicable
to "simple" and are not modelled here.

SBO's block is fill-or-kill on a *declared* profile (Q_bar_t is data, not a
decision once the block is defined): either the whole declared profile clears
or none of it. There is no per-scenario intra-day freedom to solve for, so
Q_min reduces to a static validity check on the declared availability profile
itself (see validate_block_profile) rather than a MILP.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from bidding.config import RiskObjective, TechnologyConfig


def solve_model(model: pyo.ConcreteModel, solver: str = "appsi_highs") -> None:
    opt = pyo.SolverFactory(solver)
    result = opt.solve(model)
    status = result.solver.termination_condition
    if status not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        raise RuntimeError(f"Solver did not find an optimal/feasible solution: {status}")


def _build_sco_base(
    tech: TechnologyConfig,
    lambda_matrix: np.ndarray,
    avail_matrix: np.ndarray,
    theta: dict,
) -> pyo.ConcreteModel:
    """Variables and constraints shared by both SCO clearing models: the
    physical dispatch band (w_le_u, q_le_w, q_ge_mav, optional q_ge_qmin) and
    the Minimum Income Condition. Only the objective differs between the
    "naive" and "aware" builders below."""
    S, T = lambda_matrix.shape
    pv = float(theta["price_variable"])
    tf = float(theta["fixed_term"])
    mav_frac = float(theta["mav_fraction"])
    enforce_tech = tech.enforce_technical_constraints
    qmin = tech.technical_min
    # ramp_limit is intentionally not enforced — see module docstring.

    mav = mav_frac * avail_matrix                  # (S, T)

    m = pyo.ConcreteModel()
    m.S = pyo.RangeSet(0, S - 1)
    m.T = pyo.RangeSet(0, T - 1)

    def q_bounds(m, s, t):
        return (0.0, float(avail_matrix[s, t]))

    m.q = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals, bounds=q_bounds)
    m.u = pyo.Var(m.S, domain=pyo.Binary)  # day-level SCO accept

    # Per-period commitment w[s,t] is a genuine decision for every period,
    # in-the-money or out-of-the-money alike — moneyness plays no role here.
    # A period only clears if committed, and once committed it must clear at
    # least MAV_t. Whether committing an OTM period is worthwhile is decided
    # by the solver through mic_rule and the objective, not by a structural
    # bound.
    m.w = pyo.Var(m.S, m.T, domain=pyo.Binary)  # period committed

    def w_le_u_rule(m, s, t):
        return m.w[s, t] <= m.u[s]
    m.w_le_u = pyo.Constraint(m.S, m.T, rule=w_le_u_rule)

    def q_le_w_rule(m, s, t):
        return m.q[s, t] <= float(avail_matrix[s, t]) * m.w[s, t]
    m.q_le_w = pyo.Constraint(m.S, m.T, rule=q_le_w_rule)

    def q_ge_mav_rule(m, s, t):
        return m.q[s, t] >= float(mav[s, t]) * m.w[s, t]
    m.q_ge_mav = pyo.Constraint(m.S, m.T, rule=q_ge_mav_rule)

    if enforce_tech:
        # Physical technical minimum — separate, opt-in constraint on top of
        # the (always-on) MAV floor above. Also independent of moneyness.
        def q_ge_qmin_rule(m, s, t):
            return m.q[s, t] >= qmin * m.w[s, t]
        m.q_ge_qmin = pyo.Constraint(m.S, m.T, rule=q_ge_qmin_rule)

    # Minimum Income Condition (eq. 4): trivially satisfied (0>=0) when u^s=0
    # since q is forced to 0 in that case by the constraints above.
    def mic_rule(m, s):
        revenue_surplus = sum((lambda_matrix[s, t] - pv) * m.q[s, t] for t in m.T)
        return revenue_surplus >= tf * m.u[s]
    m.mic = pyo.Constraint(m.S, rule=mic_rule)

    return m


def build_sco_model_naive(
    tech: TechnologyConfig,
    lambda_matrix: np.ndarray,
    avail_matrix: np.ndarray,
    probs: np.ndarray,
    cvar_alpha: float,
    objective: RiskObjective,
    theta: dict,
) -> pyo.ConcreteModel:
    """
    Legacy "naive" clearing model: one MILP for a fixed theta spanning all
    scenarios plus the CVaR eta/z_s variables (eq. 2-4, 10, 12-19), whose
    objective is the REAL risk-adjusted profit (variable cost C and startup
    cost C^SU included).

    This gives the solver perfect knowledge of the generator's private costs
    when deciding acceptance (u) and dispatch (q, w) — something EUPHEMIA can
    never have, since it only sees the declared parameters (P^V, TF, MAV).
    The result is therefore an upper bound: the model self-protects against
    loss-making scenarios in a way the real market cannot replicate, which is
    exactly why TF=0 comes out systematically optimal under this model (the
    protection TF should provide is already supplied, unrealistically, by the
    cost-informed u/w decisions). Kept as a benchmark; see
    build_sco_model_aware for the market-consistent model.
    """
    S, T = lambda_matrix.shape
    cost = np.asarray(tech.cost_array(T), dtype=float)

    m = _build_sco_base(tech, lambda_matrix, avail_matrix, theta)

    # Per-scenario profit (eq. 10)
    def profit_expr(m, s):
        return sum((lambda_matrix[s, t] - cost[t]) * m.q[s, t] for t in m.T) - tech.startup_cost * m.u[s]
    m.profit = pyo.Expression(m.S, rule=profit_expr)

    # CVaR (Rockafellar-Uryasev, eq. 16-18)
    m.eta = pyo.Var(domain=pyo.Reals)
    m.z = pyo.Var(m.S, domain=pyo.NonNegativeReals)

    def cvar_z_rule(m, s):
        return m.z[s] >= m.eta - m.profit[s]
    m.cvar_z = pyo.Constraint(m.S, rule=cvar_z_rule)

    expected_profit_expr = sum(float(probs[s]) * m.profit[s] for s in m.S)
    cvar_expr = m.eta - (1.0 / (1.0 - cvar_alpha)) * sum(float(probs[s]) * m.z[s] for s in m.S)

    beta = objective.beta
    m.obj = pyo.Objective(
        expr=(1.0 - beta) * expected_profit_expr + beta * cvar_expr,
        sense=pyo.maximize,
    )

    return m


def build_sco_model_aware(
    tech: TechnologyConfig,
    lambda_matrix: np.ndarray,
    avail_matrix: np.ndarray,
    probs: np.ndarray,
    theta: dict,
) -> pyo.ConcreteModel:
    """
    Market-consistent ("aware") day-level SCO clearing model.

    The revised aware implementation treats the SCO as a single daily order.
    If the day is accepted by the MIC, every period in the declared daily
    profile is dispatched at full availability. The only acceptance gate is the
    day-level declared surplus against TF; the order is not screened period-by-
    period by moneyness.

    This MILP is kept as a cross-check against the closed-form oracle in
    tests (test_milp_consistency). Since the profile is fixed once the day is
    accepted, the transfer from the day-level binary u to the full daily q
    profile is encoded explicitly below.
    """
    pv = float(theta["price_variable"])

    m = _build_sco_base(tech, lambda_matrix, avail_matrix, theta)

    def q_full_profile_rule(m, s, t):
        return m.q[s, t] == float(avail_matrix[s, t]) * m.u[s]
    m.q_full_profile = pyo.Constraint(m.S, m.T, rule=q_full_profile_rule)

    def w_day_rule(m, s, t):
        return m.w[s, t] == m.u[s]
    m.w_day = pyo.Constraint(m.S, m.T, rule=w_day_rule)

    m.obj = pyo.Objective(
        expr=sum(
            float(probs[s]) * (lambda_matrix[s, t] - pv) * m.q[s, t]
            for s in m.S for t in m.T
        ),
        sense=pyo.maximize,
    )

    return m


def build_sco_model(
    tech: TechnologyConfig,
    lambda_matrix: np.ndarray,
    avail_matrix: np.ndarray,
    probs: np.ndarray,
    cvar_alpha: float,
    objective: RiskObjective,
    theta: dict,
    sco_model: str = "aware",
) -> pyo.ConcreteModel:
    """Dispatch to the selected SCO clearing model.

    ``cvar_alpha``/``objective`` are only consumed by the "naive" branch
    (risk enters its clearing objective); under "aware" risk is applied
    afterwards, when the strategy layer selects the best theta.
    """
    if sco_model == "naive":
        return build_sco_model_naive(
            tech, lambda_matrix, avail_matrix, probs, cvar_alpha, objective, theta
        )
    if sco_model == "aware":
        return build_sco_model_aware(tech, lambda_matrix, avail_matrix, probs, theta)
    raise ValueError(f"Unknown sco_model {sco_model!r} — expected 'naive' or 'aware'.")


def extract_sco_results(model: pyo.ConcreteModel, S: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Pull the solved (q, u) arrays out of a solved SCO model."""
    q = np.zeros((S, T))
    u = np.zeros(S)
    for s in range(S):
        u[s] = pyo.value(model.u[s])
        for t in range(T):
            q[s, t] = pyo.value(model.q[s, t])
    return q, u


def validate_block_profile(tech: TechnologyConfig, avail_matrix: np.ndarray) -> None:
    """
    SBO's block profile Q_bar_t is declared data, not a per-scenario decision:
    the whole profile clears or none of it (fill-or-kill). So Q_min is a
    static validity check on the declared profile, not a MILP — raise a clear
    error if the technology's own availability profile cannot physically
    satisfy its own technical_min whenever it is producing. ramp_limit is not
    checked here — see module docstring.
    """
    if not tech.enforce_technical_constraints:
        return
    qmin = tech.technical_min
    on = avail_matrix > 0
    if qmin > 0 and np.any(on & (avail_matrix < qmin)):
        raise ValueError(
            f"{tech.name}: SBO's declared availability profile has period(s) with "
            f"0 < Q_bar_t < technical_min ({qmin}) — an all-or-nothing block cannot "
            "be dispatched below its own technical minimum. Fix the availability "
            "profile or technical_min."
        )
