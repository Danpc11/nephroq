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

NOT a diagnostic tool. Research prototype (TRL4).
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from egfr_measurement import egfr_cr, egfr_cr_cys
from mechanistic_twin import MechanisticRenalModel, N_of_egfr, DIALYSIS_eGFR, gfr_category
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

src_display = _("src_public") if CALIB_TIER == "public" else CALIBRATION_SOURCE
st.caption(_("active_calibration", src=src_display))
if CALIB_TIER == "public":
    st.warning(_("demo_mode"))
elif CALIBRATION_QUALITY != "pass":
    st.error(_("quality_warning", reasons=CALIBRATION_QUALITY_REASONS))

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

# ---- baseline eGFR ----
if cystatin:
    egfr0 = egfr_cr_cys(creatinine, cystatin, age, sex)
    method = _("method_cr_cys")
else:
    egfr0 = egfr_cr(creatinine, age, sex)
    method = _("method_cr")

_active = st.session_state.get("_active_preset")
_ap = preset_by_id(_active) if _active else None
if _ap:
    st.info(_("example_loaded", label=_ap["label"][LANG], note=_ap["note"][LANG]))

col1, col2, col3 = st.columns(3)
col1.metric(_("baseline_egfr"), f"{egfr0:.1f} mL/min/1.73m²", help=_("baseline_help", method=method))
col2.metric(_("kdigo"), gfr_category(egfr0))

# ---- projection ----
def project(a1c, sbp, uacr, egfr0, treated, q=None, khf=None, w=None, years=15):
    q = Q_POP if q is None else q
    khf = KHF_POP if khf is None else khf
    w = W_POP if w is None else w
    m = MechanisticRenalModel(a1c=a1c, sbp=sbp, uacr=uacr, u=1.0 if treated else 0.0,
                              k_hf=khf, q=q, w_a1c=w[0], w_uacr=w[1], w_sbp=w[2])
    t, N, egfr, t_dial = m.simulate(N_of_egfr(egfr0), years=years)
    return t, egfr, t_dial

t_a, e_a, td_a = project(hba1c, sbp, uacr, egfr0, treated)
t_b, e_b, td_b = project(hba1c, sbp, uacr, egfr0, not treated)
label_a = _("label_current_tx") if treated else _("label_no_tx")
label_b = _("label_reno_added") if not treated else _("label_tx_stopped")
horizon = int(t_a[-1])

# ---- bootstrap PARAMETER-uncertainty band (parameter uncertainty ONLY) --------
e_a_lo = e_a_hi = td_a_lo = td_a_hi = None
p_reach_threshold = None
if BOOTSTRAP_PARAMS:
    boot_e_a, boot_td_a = [], []
    for bp in BOOTSTRAP_PARAMS:
        bw = np.array([bp["w_a1c"], bp["w_uacr"], bp["w_sbp"]])
        _, e_boot, td_boot = project(hba1c, sbp, uacr, egfr0, treated,
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
else:
    st.caption(_("boot_none"))

fig, ax = plt.subplots(figsize=(9, 4.5))
if e_a_lo is not None:
    ax.fill_between(t_a, e_a_lo, e_a_hi, color="#E24B4A", alpha=0.15, label=_("band_label"))
ax.plot(t_a, e_a, lw=2.5, color="#E24B4A", label=label_a)
ax.plot(t_b, e_b, lw=2.5, color="#1D9E75", label=label_b)
ax.axhline(DIALYSIS_eGFR, color="k", lw=1, ls="--")
ax.text(0.3, DIALYSIS_eGFR + 2, _("plot_threshold"), fontsize=9)
ax.set_xlabel(_("plot_x")); ax.set_ylabel(_("plot_y"))
ax.set_title(_("plot_title")); ax.legend()
st.pyplot(fig)

if np.isfinite(td_a) and np.isfinite(td_b):
    st.info(_("diff_info", label=label_b, d=abs(td_b - td_a)))

if not use_cystatin:
    st.warning(_("cystatin_warning"))

with st.expander(_("expander_title")):
    st.markdown(_("expander_body"))

st.divider()
st.caption(_("footer"))
