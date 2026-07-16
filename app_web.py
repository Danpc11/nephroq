"""
================================================================================
NEPHROQ WEB INTERFACE  ·  Type 2 Diabetes -> CKD   (Streamlit)
================================================================================
Interactive, BILINGUAL (English / Spanish) web app to explore the model with
clinicians/collaborators. Language toggle is at the top of the sidebar.
All visible strings and the example patients live in src/i18n.py (single
source of truth, shared with the Colab notebook).

Run locally:            streamlit run app_web.py
Deploys for free from GitHub on Streamlit Community Cloud or HF Spaces
(see docs/WEB_DEPLOYMENT.md).
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from egfr_measurement import egfr_cr, egfr_cr_cys
import model_core as core
from model_core import DIALYSIS_eGFR, gfr_category
import personalize as pz
from i18n import t as _t, PRESETS, LANGUAGES, preset_by_id

# ------------------------------------------------------------------------------
# CALIBRATION: three tiers, from highest to lowest priority.
#   1) st.secrets["calibration"]     -> future real clinical cohort (private, NDA)
#   2) calibration/mimic_calibration.json -> generated locally with calibrate_mimic.py
#   3) hardcoded public values -> fallback if none of the above exist
# ------------------------------------------------------------------------------
Q_POP_PUBLIC, KHF_POP_PUBLIC = 1.52, 0.0141
W_POP_PUBLIC = np.array([0.0144, 0.0180, 0.0108])   # weights [A1c, UACR, blood pressure]
MIMIC_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration", "mimic_calibration.json")

def load_calibration():
    # tier 1: private secret (future, real clinical cohort)
    try:
        cal = st.secrets["calibration"]
        q = float(cal["q"]); khf = float(cal["k_hf"])
        w = np.array([float(cal["w_a1c"]), float(cal["w_uacr"]), float(cal["w_sbp"])])
        return q, khf, w, "private (clinical cohort)", "private", "pass", [], None
    except Exception:
        pass
    # tier 2: local calibration with MIMIC-IV (calibrate_mimic.py)
    try:
        import json
        with open(MIMIC_JSON_PATH) as f:
            cal = json.load(f)
        q = float(cal["q"]); khf = float(cal["k_hf"])
        w = np.array([float(cal["w_a1c"]), float(cal["w_uacr"]), float(cal["w_sbp"])])
        quality_status = cal.get("quality_status", "unknown")
        quality_reasons = cal.get("quality_reasons", [])
        bootstrap_params = cal.get("bootstrap_params") or None
        src = f"MIMIC-IV {cal.get('mimic_version','')} (n={cal.get('n_patients','?')} patients)"
        return q, khf, w, src, "mimic", quality_status, quality_reasons, bootstrap_params
    except Exception:
        pass
    # tier 3: public fallback (synthetic + Al-Shamsi validation)
    return Q_POP_PUBLIC, KHF_POP_PUBLIC, W_POP_PUBLIC, None, "public", "pass", [], None

(Q_POP, KHF_POP, W_POP, CALIBRATION_SOURCE, CALIB_TIER,
 CALIBRATION_QUALITY, CALIBRATION_QUALITY_REASONS, BOOTSTRAP_PARAMS) = load_calibration()

st.set_page_config(page_title="NephroQ · Diabetes → CKD", page_icon="🩺", layout="wide")

# ---- language (read from session_state so the whole page reacts on rerun) ----
st.session_state.setdefault("_lang_label", "English")
LANG = LANGUAGES[st.session_state["_lang_label"]]
def _(key, **kw):
    return _t(LANG, key, **kw)

st.title(_("title"))
st.caption(_("disclaimer"))
st.caption(_("research_use"))

src_display = _("src_trial") if CALIB_TIER == "public" else CALIBRATION_SOURCE
st.caption(_("active_calibration", src=src_display))
if CALIB_TIER == "public":
    st.warning(_("demo_mode"))
elif CALIBRATION_QUALITY != "pass":
    # SAFE-BY-DEFAULT POLICY.
    # Previously the app displayed a warning but still PROJECTED with the flagged
    # parameters -- so in a clinician demo the curves on screen came from a
    # calibration the pipeline itself had judged untrustworthy. A warning above a
    # plot does not undo the plot. Now the flagged calibration is NOT used unless
    # the user explicitly opts in (research mode); otherwise we fall back to the
    # public calibration, which at least is honest about being a demo.
    _reasons = ", ".join(str(r).replace("_", " ") for r in (CALIBRATION_QUALITY_REASONS or [])) or "—"
    st.error(_("quality_warning", reasons=_reasons))
    _use_flagged = st.checkbox(_("quality_optin"), value=False)
    if _use_flagged:
        st.error(_("using_flagged"))
    else:
        Q_POP, KHF_POP, W_POP = Q_POP_PUBLIC, KHF_POP_PUBLIC, W_POP_PUBLIC
        BOOTSTRAP_PARAMS = None
        CALIB_TIER = "public"
        st.info(_("fell_back_public"))
        st.warning(_("demo_mode"))

# ------------------------------------------------------------------------------
# Seed widget state once, then apply any pending preset BEFORE the widgets are
# instantiated (Streamlit requires session_state set before the reading widget).
# ------------------------------------------------------------------------------
_DEFAULTS = dict(age=58, sex="F", creatinine=1.3, hba1c=8.1, uacr=145.0, sbp=142)
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
if "_pending_preset" in st.session_state:
    _pid = st.session_state.pop("_pending_preset")
    _preset = preset_by_id(_pid)
    if _preset:
        for _k, _v in _preset["markers"].items():
            st.session_state[_k] = _v
        st.session_state["_active_preset"] = _pid

with st.sidebar:
    st.radio(_("language"), list(LANGUAGES.keys()), key="_lang_label", horizontal=True)
    st.divider()
    st.header(_("examples_header"))
    st.caption(_("examples_caption"))
    for _p in PRESETS:
        if st.button(_p["label"][LANG], use_container_width=True, key=f"btn_{_p['id']}"):
            st.session_state["_pending_preset"] = _p["id"]
            st.rerun()
    if st.button(_("reset"), use_container_width=True):
        for _k, _v in _DEFAULTS.items():
            st.session_state[_k] = _v
        st.session_state.pop("_active_preset", None)
        st.rerun()
    st.divider()

    st.header(_("markers_header"))
    age = st.number_input(_("age"), 18, 100, key="age")
    sex = st.radio(_("sex"), ["F", "M"], horizontal=True, key="sex")
    st.divider()
    st.subheader(_("blood"))
    creatinine = st.number_input(_("creatinine"), 0.3, 10.0, step=0.1, key="creatinine")
    use_cystatin = st.checkbox(_("have_cystatin"))
    cystatin = st.number_input(_("cystatin"), 0.3, 8.0, 1.3, step=0.1) if use_cystatin else None
    hba1c = st.number_input(_("hba1c"), 4.0, 15.0, step=0.1, key="hba1c")
    st.subheader(_("urine"))
    uacr = st.number_input(_("uacr"), 0.0, 3000.0, step=5.0, key="uacr")
    st.subheader(_("in_clinic"))
    sbp = st.number_input(_("sbp"), 80, 220, key="sbp")
    st.divider()
    treated = st.checkbox(_("treated"), value=False)
    st.divider()

    # --- measurement history -> per-patient personalization ---
    st.header(_("history_header"))
    st.caption(_("history_caption"))
    if "history" not in st.session_state:
        # Start EMPTY. Pre-filling fictitious creatinines silently switched on
        # personalization for a patient whose history the clinician never entered,
        # producing a "personalized" projection built on invented data. A "load
        # example" button below offers demo values explicitly, on request.
        st.session_state["history"] = pd.DataFrame(
            {"years_ago": pd.Series([], dtype=float),
             "creatinine": pd.Series([], dtype=float)})
    hist = st.data_editor(
        st.session_state["history"], num_rows="dynamic", use_container_width=True,
        column_config={
            "years_ago": st.column_config.NumberColumn(_("history_years_ago"),
                                                       min_value=0.0, max_value=25.0, step=0.5),
            "creatinine": st.column_config.NumberColumn(_("history_creat"),
                                                        min_value=0.3, max_value=15.0, step=0.05),
        }, key="history_editor")

# ---- baseline eGFR ----
if cystatin:
    egfr0 = egfr_cr_cys(creatinine, cystatin, age, sex)
    method = _("method_cr_cys")
else:
    egfr0 = egfr_cr(creatinine, age, sex)
    method = _("method_cr")

@st.cache_resource(show_spinner=False)
def _get_personalizer():
    """Trained once per app process and cached. The estimator is derived purely
    from simulations of the mechanistic model, so it is regenerated on demand
    rather than shipped as a binary in the repository."""
    return pz.get_estimator()


# ---- PER-PATIENT PERSONALIZATION (amortized inference) ------------------------
# The measurement history (if any) is converted to an eGFR series with the
# patient's own age/sex, then a neural estimator -- trained purely on simulations
# from the mechanistic model -- infers this patient's injury rate and collapse
# exponent. The forward projection is still the ODE: the network only solves the
# inverse problem.
PERSONAL = dict(personalized=False, params=dict(core.TRIAL_CALIBRATION_V2))
try:
    h = hist.dropna()
    h = h[(h["years_ago"] >= 0) & (h["creatinine"] > 0)].sort_values("years_ago", ascending=False)
    if len(h) >= pz.MIN_VISITS - 1:      # + today's measurement
        # today's measurement closes the series
        t_hist = np.append((h["years_ago"].max() - h["years_ago"]).to_numpy(float),
                           float(h["years_ago"].max()))
        # Historical creatinines must be converted to eGFR using the patient's
        # age AT THE TIME of each measurement, not their current age. A creatinine
        # from 3 years ago belongs to a patient who was 3 years younger, and
        # CKD-EPI is age-dependent; using current age biases the reconstructed
        # history (it inflates the apparent early decline). today's sample uses
        # today's age.
        age_at = (age - h["years_ago"]).to_numpy(float)
        e_hist = np.append(
            np.array([egfr_cr(c, a, sex) for c, a in zip(h["creatinine"], age_at)]),
            egfr0)
        with st.spinner(_("training_estimator")):
            _est = _get_personalizer()
        PERSONAL = pz.personalize(t_hist, e_hist, hba1c, uacr, sbp, estimator=_est)
except Exception:
    pass                    # never let personalization break the app
P_PARAMS = PERSONAL["params"]

# The ACTIVE TIER's population parameters, as a dict. This is the base that
# personalization (s_i) is applied on top of, and the correct precedence root:
# public -> trial anchors; otherwise the local MIMIC/private calibration.
P_PARAMS_POP = dict(core.TRIAL_CALIBRATION_V2)
if CALIB_TIER != "public":
    P_PARAMS_POP.update(q=Q_POP, k_hf=KHF_POP,
                        w_a1c=W_POP[0], w_uacr=W_POP[1], w_sbp=W_POP[2])

_active = st.session_state.get("_active_preset")
_ap = preset_by_id(_active) if _active else None
if _ap:
    st.info(_("example_loaded", label=_ap["label"][LANG], note=_ap["note"][LANG]))

if PERSONAL["personalized"]:
    st.success(_("personalized_on", scale=PERSONAL["scale"], scale_sd=PERSONAL["scale_sd"],
                 q=PERSONAL["q"], q_sd=PERSONAL["q_sd"]))
    st.caption(_("personalized_caveat"))
else:
    st.info(_("personalized_off"))

col1, col2, col3 = st.columns(3)
col1.metric(_("baseline_egfr"), f"{egfr0:.1f} mL/min/1.73m²", help=_("baseline_help", method=method))
col2.metric(_("kdigo"), gfr_category(egfr0))

# ---- projection ----
def project(a1c, sbp, uacr, egfr0, treated, q=None, khf=None, w=None, years=15):
    """
    Project a trajectory with the model (v2: saturating hyperfiltration +
    endogenous albuminuria). Returns (t, eGFR, time_to_threshold, UACR).

    There is ONE model. The calibration tiers only change its parameters:
    the default parameters are anchored to published trials, and a local
    MIMIC calibration (if present) overrides q / k_hf / the covariate weights.
    """
    # Parameter precedence, applied in the correct order:
    #   1. Start from the ACTIVE calibration tier (public trials, or a local
    #      MIMIC/private calibration if present). This sets the POPULATION
    #      parameters: q, k_hf, and the covariate weights.
    #   2. Apply this patient's individual susceptibility s_i ON TOP, as a
    #      multiplier of the hazard scale. Personalization rescales the population
    #      model; it does not replace it, and a MIMIC/private tier must NOT
    #      overwrite it (that was the bug -- the tier used to clobber the
    #      personalized parameters after the fact).
    # q stays at the POPULATION value throughout: the repo's own experiments show
    # q is not identifiable per patient, so only s_i (scale) is individual.
    if q is not None:
        # explicit override (used by the scenario controls / tests)
        p = dict(P_PARAMS_POP)
        p.update(q=q, k_hf=khf, w_a1c=w[0], w_uacr=w[1], w_sbp=w[2])
    else:
        p = dict(P_PARAMS_POP)          # active tier's population parameters
        if PERSONAL["personalized"]:
            s_i = PERSONAL["scale"]     # individual susceptibility, applied on top
            p["k_hf"]   = P_PARAMS_POP["k_hf"]   * s_i
            p["w_a1c"]  = P_PARAMS_POP["w_a1c"]  * s_i
            p["w_uacr"] = P_PARAMS_POP["w_uacr"] * s_i
            p["w_sbp"]  = P_PARAMS_POP["w_sbp"]  * s_i
            # q is deliberately left at the population value
    t, egfr, uacr_t, t_thr = core.simulate_trajectory_v2(
        egfr0=egfr0, a1c=a1c, uacr0=uacr, sbp=sbp,
        u=1.0 if treated else 0.0, p=p, years=years)
    return t, egfr, t_thr, uacr_t


t_a, e_a, td_a, ua_a = project(hba1c, sbp, uacr, egfr0, treated)
t_b, e_b, td_b, ua_b = project(hba1c, sbp, uacr, egfr0, not treated)
label_a = _("label_current_tx") if treated else _("label_no_tx")
label_b = _("label_reno_added") if not treated else _("label_tx_stopped")
horizon = int(t_a[-1])

# ---- bootstrap PARAMETER-uncertainty band (parameter uncertainty ONLY) --------
e_a_lo = e_a_hi = td_a_lo = td_a_hi = None
p_reach_threshold = None

# DEGENERATE-BOOTSTRAP GUARD.
# If the calibration's optimizer terminated prematurely (see the README (Limitations)
# "optimizer scaling"), every bootstrap replicate returns essentially the SAME
# parameters. Their spread would then be ~0 and the band would collapse onto the
# central line -- rendering as a *falsely precise* projection, which is worse
# than showing no band at all.
#
# The check is done on the TRAJECTORIES, not on q/k_hf alone: the metabolic
# weights (w_a1c/w_uacr/w_sbp) can carry real variability even when q and k_hf
# look frozen, and it is the spread of the projected curve -- what the user
# actually sees -- that decides whether a band is meaningful.
_boot_degenerate = False
_boot_traj = None
if BOOTSTRAP_PARAMS and len(BOOTSTRAP_PARAMS) >= 2:
    _traj = []
    for bp in BOOTSTRAP_PARAMS:
        _bw = np.array([bp["w_a1c"], bp["w_uacr"], bp["w_sbp"]])
        _, _e_b, _, _ = project(hba1c, sbp, uacr, egfr0, treated,
                                q=bp["q"], khf=bp["k_hf"], w=_bw)
        _traj.append(_e_b)
    _boot_traj = np.array(_traj)
    # max over time of the across-resample std, in mL/min/1.73m2
    _spread = float(np.nanmax(np.std(_boot_traj, axis=0)))
    _boot_degenerate = _spread < 1e-3

if BOOTSTRAP_PARAMS and _boot_degenerate:
    BOOTSTRAP_PARAMS = None   # fall through to the point-estimate-only path below

if BOOTSTRAP_PARAMS:
    boot_e_a, boot_td_a = [], []
    for bp in BOOTSTRAP_PARAMS:
        bw = np.array([bp["w_a1c"], bp["w_uacr"], bp["w_sbp"]])
        _, e_boot, td_boot, _ = project(hba1c, sbp, uacr, egfr0, treated,
                                       q=bp["q"], khf=bp["k_hf"], w=bw)
        boot_e_a.append(e_boot); boot_td_a.append(td_boot)
    boot_e_a = np.array(boot_e_a)
    e_a_lo, e_a_hi = np.percentile(boot_e_a, [5, 95], axis=0)
    boot_td_a = np.array(boot_td_a)
    finite_td = boot_td_a[np.isfinite(boot_td_a)]
    p_reach_threshold = len(finite_td) / len(boot_td_a)
    if p_reach_threshold >= 0.5 and len(finite_td) >= 3:
        td_a_lo, td_a_hi = np.percentile(finite_td, [5, 95])

state = _("state_current") if treated else _("state_untreated")
time_str = _("years", v=td_a) if np.isfinite(td_a) else _("gt_years", v=horizon)
col3.metric(_("time_title", state=state), time_str, help=_("time_help"))

if BOOTSTRAP_PARAMS:
    st.caption(_("boot_reach", n=len(BOOTSTRAP_PARAMS), pct=100*p_reach_threshold, horizon=horizon))
    if td_a_lo is not None:
        st.caption(_("boot_interval", lo=td_a_lo, hi=td_a_hi))
    else:
        st.caption(_("boot_no_interval", horizon=horizon))
elif _boot_degenerate:
    st.warning(_("boot_degenerate"))
else:
    st.caption(_("boot_none"))

# constrained_layout (rather than tight_layout) keeps the axis labels, title and
# legend from being clipped when Streamlit scales the figure down on a narrow
# screen; use_container_width lets it track the column width instead of being
# pinned to a fixed pixel size. The legend labels are long in Spanish, so they
# go below the axes rather than overlapping the curves.
fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
if e_a_lo is not None:
    ax.fill_between(t_a, e_a_lo, e_a_hi, color="#E24B4A", alpha=0.15, label=_("band_label"))
ax.plot(t_a, e_a, lw=2.5, color="#E24B4A", label=label_a)
ax.plot(t_b, e_b, lw=2.5, color="#1D9E75", label=label_b)
ax.axhline(DIALYSIS_eGFR, color="k", lw=1, ls="--")
ax.text(0.3, DIALYSIS_eGFR + 2, _("plot_threshold"), fontsize=9)
ax.set_xlabel(_("plot_x")); ax.set_ylabel(_("plot_y"))
ax.set_title(_("plot_title"))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=9, frameon=False)
ax.grid(alpha=0.15)
st.pyplot(fig, use_container_width=True)
plt.close(fig)   # Streamlit reruns on every widget change; without this, figures accumulate.

if np.isfinite(td_a) and np.isfinite(td_b):
    st.info(_("diff_info", label=label_b, d=abs(td_b - td_a)))

# --- NEW IN v2: albuminuria is a model OUTPUT, so we can plot it ---------------
if ua_a is not None and ua_b is not None:
    ua_treated = ua_a if treated else ua_b
    ua_untreated = ua_b if treated else ua_a
    drop = 100.0 * (1.0 - ua_treated[0] / ua_untreated[0]) if ua_untreated[0] > 0 else 0.0
    fig2, ax2 = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    ax2.plot(t_a, ua_a, lw=2.2, color="#E24B4A", label=label_a)
    ax2.plot(t_b, ua_b, lw=2.2, color="#1D9E75", label=label_b)
    ax2.set_xlabel(_("plot_x")); ax2.set_ylabel(_("uacr_y"))
    ax2.set_title(_("uacr_plot_title"))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=9, frameon=False)
    ax2.grid(alpha=0.15)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.caption(_("uacr_note", drop=drop))

if not use_cystatin:
    st.warning(_("cystatin_warning"))

with st.expander(_("expander_title")):
    # The description must match the ACTIVE calibration: the public tier really is a
    # hierarchical Bayesian fit on synthetic data, but a MIMIC calibration produced by
    # calibrate_mimic.py is robust nonlinear least squares + patient-level bootstrap.
    st.markdown(_("expander_body_mimic" if CALIB_TIER == "mimic" else "expander_body_v2"))

st.divider()
st.caption(_("footer"))
