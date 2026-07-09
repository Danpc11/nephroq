"""
================================================================================
NEPHROQ WEB INTERFACE  ·  Type 2 Diabetes -> CKD   (Streamlit)
================================================================================
Interactive web app to explore the model with clinicians/collaborators.
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
from mechanistic_twin import MechanisticRenalModel, N_of_egfr, DIALYSIS_eGFR

# ------------------------------------------------------------------------------
# CALIBRATION: three tiers, from highest to lowest priority.
#   1) st.secrets["calibration"]     -> future real clinical cohort (private, NDA)
#   2) calibration/mimic_calibration.json -> generated locally with calibrate_mimic.py
#      (research/demo of THIS repository; aggregate, not PHI; see docs/MIMIC_COMPLIANCE.md)
#   3) hardcoded public values -> fallback if none of the above exist
#      (e.g. someone clones the repo without running calibrate_mimic.py)
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
        return q, khf, w, "private (clinical cohort)"
    except Exception:
        pass
    # tier 2: local calibration with MIMIC-IV (calibrate_mimic.py)
    try:
        import json
        with open(MIMIC_JSON_PATH) as f:
            cal = json.load(f)
        q = float(cal["q"]); khf = float(cal["k_hf"])
        w = np.array([float(cal["w_a1c"]), float(cal["w_uacr"]), float(cal["w_sbp"])])
        return q, khf, w, f"MIMIC-IV {cal.get('mimic_version','')} (n={cal.get('n_patients','?')} patients)"
    except Exception:
        pass
    # tier 3: public fallback (synthetic + Al-Shamsi validation)
    return Q_POP_PUBLIC, KHF_POP_PUBLIC, W_POP_PUBLIC, "public (synthetic + Al-Shamsi 2018 validation)"

Q_POP, KHF_POP, W_POP, CALIBRATION_SOURCE = load_calibration()

st.set_page_config(page_title="NephroQ · Diabetes → CKD", page_icon="🩺", layout="wide")

st.title("🩺 NephroQ — renal risk digital twin in type 2 diabetes")
st.caption("Research prototype (TRL4) — NOT a diagnostic tool. "
          "Must not be used for clinical decisions without qualified medical supervision.")
st.caption(f"Active calibration: **{CALIBRATION_SOURCE}**")
if CALIBRATION_SOURCE.startswith("public"):
    st.warning("**Demonstration mode** — projections are generated from a synthetic "
              "research calibration and must not be interpreted as individualized "
              "clinical predictions.")

with st.sidebar:
    st.header("Patient markers")
    age = st.number_input("Age (years)", 18, 100, 58)
    sex = st.radio("Sex", ["F", "M"], horizontal=True)
    st.divider()
    st.subheader("Blood")
    creatinine = st.number_input("Serum creatinine (mg/dL)", 0.3, 10.0, 1.3, step=0.1)
    use_cystatin = st.checkbox("I have cystatin C (more precise)")
    cystatin = st.number_input("Cystatin C (mg/L)", 0.3, 8.0, 1.3, step=0.1) if use_cystatin else None
    hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 8.1, step=0.1)
    st.subheader("Urine")
    uacr = st.number_input("UACR — urine albumin/creatinine ratio (mg/g)", 0.0, 3000.0, 145.0, step=5.0)
    st.subheader("In clinic")
    sbp = st.number_input("Systolic blood pressure (mmHg)", 80, 220, 142)
    st.divider()
    treated = st.checkbox("Already receiving a renoprotective therapy (illustrative: SGLT2i/ACEi-ARB combined effect)",
                          value=False)

# ---- baseline eGFR ----
if cystatin:
    egfr0 = egfr_cr_cys(creatinine, cystatin, age, sex)
    method = "creatinine + cystatin (more precise)"
else:
    egfr0 = egfr_cr(creatinine, age, sex)
    method = "creatinine only"

def gfr_category(egfr):
    """KDIGO GFR category. G3 splits into G3a (45-59) and G3b (30-44) --
    a single eGFR value gives the GFR category, not a CKD diagnosis (that
    requires persistence >=3 months plus cause and albuminuria, per KDIGO)."""
    if egfr >= 90: return "G1"
    if egfr >= 60: return "G2"
    if egfr >= 45: return "G3a"
    if egfr >= 30: return "G3b"
    if egfr >= 15: return "G4"
    return "G5"

col1, col2, col3 = st.columns(3)
col1.metric("Baseline eGFR", f"{egfr0:.1f} mL/min/1.73m²", help=f"Calculated with: {method}")
col2.metric("KDIGO GFR category", gfr_category(egfr0))

# ---- projection ----
def project(a1c, sbp, uacr, egfr0, treated, years=15):
    m = MechanisticRenalModel(a1c=a1c, sbp=sbp, uacr=uacr, u=1.0 if treated else 0.0,
                              k_hf=KHF_POP, q=Q_POP,
                              w_a1c=W_POP[0], w_uacr=W_POP[1], w_sbp=W_POP[2])
    t, N, egfr, t_dial = m.simulate(N_of_egfr(egfr0), years=years)
    return t, egfr, t_dial

t_a, e_a, td_a = project(hba1c, sbp, uacr, egfr0, treated)
t_b, e_b, td_b = project(hba1c, sbp, uacr, egfr0, not treated)
label_a = "Current treatment" if treated else "No treatment (current scenario)"
label_b = "Illustrative renoprotective scenario added" if not treated else "If treatment is stopped"

col3.metric(f"Modeled time to eGFR<15 ({'current' if treated else 'untreated'})",
           f"{td_a:.1f} years" if np.isfinite(td_a) else ">15 years",
           help="This is a modeled kidney-function threshold (eGFR<15), not a "
                "prediction of when dialysis would actually start. Real dialysis "
                "initiation depends on symptoms, labs, and clinical judgment.")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_a, e_a, lw=2.5, color="#E24B4A", label=label_a)
ax.plot(t_b, e_b, lw=2.5, color="#1D9E75", label=label_b)
ax.axhline(DIALYSIS_eGFR, color="k", lw=1, ls="--")
ax.text(0.3, DIALYSIS_eGFR + 2, "modeled eGFR<15 threshold", fontsize=9)
ax.set_xlabel("years"); ax.set_ylabel("projected eGFR (mL/min/1.73m²)")
ax.set_title("Illustrative model projection of renal function"); ax.legend()
st.pyplot(fig)

if np.isfinite(td_a) and np.isfinite(td_b):
    diff = abs(td_b - td_a)
    st.info(f"**{label_b}** changes the modeled time to the eGFR<15 threshold by "
           f"approximately **{diff:.1f} years** relative to the current scenario, "
           f"under the assumptions of the current research model. This is not a "
           f"prediction of dialysis initiation.")

if not use_cystatin:
    st.warning("eGFR was calculated with creatinine only. Requesting cystatin C reduces "
              "the estimation error of the feedback exponent (q) by ~5×.")

with st.expander("What does this mean? (to share with the patient/physician)"):
    st.markdown("""
    This model simulates the progressive loss of functional nephrons using two
    mechanisms: **hyperfiltration** (as nephrons are lost, the remaining ones
    become overloaded and are damaged faster) and **compensation** (eGFR stays
    stable while there is reserve, and drops faster near the end).

    The model parameters were calibrated with hierarchical Bayesian inference
    on verified synthetic data and a first face-validity check against real
    published data. **It has not been validated on a prospective clinical
    cohort** — see `docs/MODEL_DOCUMENTATION.md` for the full project status
    and what remains for clinical publication.
    """)

st.divider()
st.caption("Source code and full documentation: "
          "[github.com/<your-username>/nephroq](https://github.com)")
