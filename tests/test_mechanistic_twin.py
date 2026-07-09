"""Unit tests for the mechanistic core (isolated component)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mechanistic_twin import (MechanisticRenalModel, egfr_of_N, N_of_egfr,
                              DIALYSIS_eGFR, N_DIALYSIS, G_MAX)

def test_N_egfr_are_inverses():
    for e in [20, 50, 80, 110]:
        n = N_of_egfr(e)
        assert abs(egfr_of_N(n) - e) < 1e-6, "the eGFR<->N mapping must be consistent"

def test_N_decreases_monotonically():
    m = MechanisticRenalModel(a1c=9, sbp=150, uacr=300, u=0)
    t, N, egfr, t_dial = m.simulate(N0=0.6, years=20)
    assert all(N[i+1] <= N[i] + 1e-9 for i in range(len(N)-1)), \
        "N must be non-increasing (nephron irreversibility)"

def test_treatment_delays_dialysis():
    """The core mechanism of the model: treatment must delay dialysis."""
    untreated = MechanisticRenalModel(a1c=9, sbp=150, uacr=300, u=0.0)
    treated = MechanisticRenalModel(a1c=9, sbp=150, uacr=300, u=1.0)
    _, _, _, t_untreated = untreated.simulate(N0=0.62, years=25)
    _, _, _, t_treated = treated.simulate(N0=0.62, years=25)
    assert t_treated > t_untreated, "the treated patient must reach dialysis later"

def test_risk_profile_vs_control():
    """Face validity: a risk profile progresses faster than a controlled profile."""
    risk = MechanisticRenalModel(a1c=9.0, sbp=150, uacr=300, u=0.0)
    control = MechanisticRenalModel(a1c=6.8, sbp=125, uacr=30, u=0.0)
    _, _, _, t_risk = risk.simulate(N0=0.62, years=25)
    _, _, _, t_control = control.simulate(N0=0.62, years=25)
    assert t_risk < t_control

def test_hazard_grows_as_N_falls():
    """Verifies the hyperfiltration mechanism: hazard(N) must be decreasing in N."""
    m = MechanisticRenalModel(a1c=9, sbp=150, uacr=300)
    h_low  = m.hazard(0.10)
    h_high  = m.hazard(0.80)
    assert h_low > h_high, "hazard per nephron must grow as N gets small"
