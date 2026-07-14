"""
================================================================================
MVP CALIBRATION WITH REAL DATA  ·  Digital twin Diabetes -> CKD
================================================================================
Minimum viable product: ingests patient data, calibrates the mechanistic
model, and produces a VALIDATION report presentable to clinicians.

Accepts a longitudinal CSV with columns (mappable):
    patient_id, time_years, egfr, hba1c, uacr, sbp
If no CSV is given, it generates a realistic synthetic cohort to run immediately.

Validations produced (the ones that convince a clinician):
  (1) Fit:         observed vs predicted (R^2, RMSE).
  (2) FORECAST:    fit the first part of the trajectory and predict the FUTURE,
                   compared against linear extrapolation (the key test).
  (3) Progressor discrimination (concordance of final eGFR).
Saves a figure + a .md report with the numbers.

Input file: TAB-SEPARATED (.tsv), one row per visit per patient:

    patient_id\ttime_years\tegfr\thba1c\tuacr\tsbp
    P001\t0.0\t58.2\t8.4\t180\t145
    P001\t0.6\t55.1\t8.1\t210\t142

Tabs are the default because clinical exports routinely contain commas inside
fields (free-text sites, "Apellido, Nombre"), which silently corrupt a CSV. A
comma-separated file still works -- the delimiter is sniffed -- but tabs are
what this expects.

Usage:   CKD_DATA=my_data.tsv  python mvp_calibration.py
         (CKD_CSV is still honoured for backward compatibility)
================================================================================
"""
import os, numpy as np
from scipy.optimize import minimize_scalar, least_squares
import matplotlib.pyplot as plt

import model_core as core
from model_core import N_of_egfr, egfr_of_N

rng = np.random.default_rng(2)
G_MAX = 120.0          # physiological eGFR ceiling (same as model_core)

# THE MODEL LIVES IN model_core. This file used to carry its own fixed-step Euler
# integrator and the OLD unbounded hazard, which drifted from the app by up to
# 13 mL/min at 10 years -- so a user who calibrated on their own data was fitting
# a DIFFERENT model from the one the app projects with. It now calls the same
# simulator as everything else. There is one model.

def _params(q, khf, w):
    p = dict(core.TRIAL_CALIBRATION_V2)
    p.update(q=float(q), k_hf=float(khf),
             w_a1c=float(w[0]), w_uacr=float(w[1]), w_sbp=float(w[2]))
    return p

def predict(q, khf, cov, w, tq, e0):
    """eGFR at times `tq`. `cov` = (HbA1c, UACR, SBP) at BASELINE."""
    a1c, uacr, sbp = cov
    tq = np.atleast_1d(np.asarray(tq, dtype=float))
    return core.predict_egfr_at_v2(e0, a1c, uacr, sbp, 0.0, _params(q, khf, w), tq)

def simulate(q, khf, cov, w, tmax, e0):
    """Kept for the synthetic-cohort generator. Returns (t, N)."""
    a1c, uacr, sbp = cov
    t, egfr, _, _ = core.simulate_trajectory_v2(e0, a1c, uacr, sbp, u=0.0,
                                                p=_params(q, khf, w),
                                                years=float(tmax), n=200)
    return t, N_of_egfr(egfr)

# --------------------------------------------------- data loading
def load_table(path):
    """
    Read a TAB-separated visit table (a comma-separated one also works: the
    delimiter is sniffed from the header).
    """
    import csv as _csv
    import pandas as pd

    with open(path, "r", newline="") as fh:
        head = fh.readline()
    if "\t" in head:
        sep = "\t"
    else:
        try:
            sep = _csv.Sniffer().sniff(head, delimiters="\t,;|").delimiter
        except Exception:
            sep = ","
    print(f"  delimiter detected: {'TAB' if sep == chr(9) else repr(sep)}")
    df = pd.read_csv(path, sep=sep)
    # flexible mapping of common names -> standard schema
    ren={c.lower():c for c in df.columns}
    def col(*names):
        for n in names:
            if n in ren: return ren[n]
        raise KeyError(f"missing column {names}")
    pid=col("patient_id","id","subject"); tt=col("time_years","time","t","years")
    eg=col("egfr","egfr_value"); ha=col("hba1c","a1c"); ua=col("uacr","acr","albuminuria"); bp=col("sbp","systolic","bp")
    # Development-set medians, used ONLY to fill a covariate a patient never had.
    # They are computed from BASELINE rows across the cohort -- never from a
    # patient's own later visits, which would leak the future into a baseline
    # forecast.
    firsts = df.sort_values(tt).groupby(df[pid]).first()
    defaults = {c: float(firsts[c].median()) for c in (ha, ua, bp)
                if firsts[c].notna().any()}

    pats, n_imputed = [], {ha: 0, ua: 0, bp: 0}
    for k, g in df.groupby(df[pid]):
        g = g.dropna(subset=[tt, eg]).sort_values(tt)
        if len(g) < 3:
            continue
        # BASELINE covariates: the value at (or before) the index visit.
        cov = []
        for c in (ha, ua, bp):
            v = g[c].iloc[0]
            if not np.isfinite(v):
                v = defaults.get(c, np.nan)
                n_imputed[c] += 1
            cov.append(float(v))
        pats.append(dict(cov=tuple(cov), e0=float(g[eg].iloc[0]),
                         t=g[tt].values.astype(float), e=g[eg].values.astype(float)))

    if pats:
        for c, n in n_imputed.items():
            if n:
                print(f"  {c}: baseline missing for {n}/{len(pats)} patients "
                      f"({100*n/len(pats):.0f}%) -> filled with the cohort baseline median")
    return pats

def make_synth(n=200):
    # Distributions CALIBRATED to the S1 Table (Al-Shamsi 2018, PLOS One, n=491).
    # Mixture of two groups: high risk (profile of those who developed CKD 3-5)
    # and low risk (profile of those who did not). p_high enriched to 0.35 because
    # the population to monitor is diabetic/at-risk, not the general population.
    #   HbA1c, SBP, baseline eGFR -> REAL means/SDs from the table.
    #   UACR -> IMPUTED by group (the S1 Table did NOT measure albuminuria).
    QT,KT,WT,TAU=1.52,0.0141,np.array([0.0144,0.018,0.0108]),0.45   # from the Bayesian fit
    P=[]
    for _ in range(n):
        if rng.random()<0.35:   # "developed CKD" profile (n=56 real)
            a1c=np.clip(rng.normal(8.30,2.57),5.5,14.0)
            sbp=np.clip(rng.normal(136.7,17.7),100,200)
            e0 =np.clip(rng.normal(79.6,12.5),40,110)
            uacr=float(np.clip(rng.lognormal(np.log(150),0.9),10,1500))  # imputed (high)
        else:                   # "no CKD" profile (n=435 real)
            a1c=np.clip(rng.normal(6.38,1.44),4.8,11.0)
            sbp=np.clip(rng.normal(130.7,15.3),95,190)
            e0 =np.clip(rng.normal(100.5,17.8),60,125)
            uacr=float(np.clip(rng.lognormal(np.log(25),0.8),5,400))     # imputed (low)
        cov=(a1c,uacr,sbp); eta=rng.normal(0,TAU)
        ts,Ns=simulate(QT,KT*np.exp(eta),cov,WT,9,e0); e=egfr_of_N(Ns)
        fu=rng.choice([2.0,4.0,6.0,8.0],p=[0.2,0.3,0.3,0.2]); tg=np.arange(0,fu+1e-3,0.25)
        eg=np.interp(tg,ts,e); k=eg>12; tg,eg=tg[k],eg[k]
        m=rng.random(len(tg))>0.2; tg,eg=tg[m],eg[m]
        if len(tg)<6: continue
        P.append(dict(cov=cov,e0=e0,t=tg,e=np.maximum(eg+rng.normal(0,3.0,len(eg)),1)))
    return P

DATA = os.environ.get("CKD_DATA", "") or os.environ.get("CKD_CSV", "")
if DATA and os.path.exists(DATA):
    print(f"Real data: {DATA}")
    cohort = load_table(DATA); SYNTH = False
else:
    print("No input file -> synthetic cohort (demo).  For real data: "
          "CKD_DATA=my_data.tsv  (tab-separated)")
    cohort = make_synth(); SYNTH = True
for p in cohort: p["w"]=np.array([0.0144,0.018,0.0108])
print(f"{len(cohort)} patients loaded.\n")

# --------------------------------------------------- 1) population calibration
def res_pop(p):
    q=0.8+1.7/(1+np.exp(-p[0])); khf=np.exp(p[1]); w=np.exp(p[2:5])
    out=[]
    for pat in cohort:
        pat["w"]=w; out.append((predict(q,khf,pat["cov"],w,pat["t"],pat["e0"])-pat["e"])/3.0)
    r=np.concatenate(out); return np.where(np.isfinite(r),r,50.)
sol=least_squares(res_pop,[0,np.log(0.012),*np.log([0.014,0.018,0.011])],method="trf",max_nfev=2500)
q_hat=0.8+1.7/(1+np.exp(-sol.x[0])); khf_hat=np.exp(sol.x[1]); w_hat=np.exp(sol.x[2:5])
for p in cohort: p["w"]=w_hat
print(f"Population parameters:  q={q_hat:.2f}   k_hf={khf_hat:.4f}   w={np.round(w_hat,4)}")

def fit_eta(pat,t,e,tau=0.4):
    def obj(eta):
        r=(predict(q_hat,khf_hat*np.exp(eta),pat["cov"],pat["w"],t,e[0])-e)/3.0
        return 0.5*np.sum(r**2)+eta**2/(2*tau**2)
    return minimize_scalar(obj,bounds=(-1.5,1.5),method="bounded").x

# --------------------------------------------------- 2) FORECAST VALIDATION
print("\n--- FORECAST VALIDATION (fit the past, predict the future) ---")
rm_model,rm_lin=[],[]; pred_end,obs_end,min_te=[],[],[]
for pat in cohort:
    t,e=pat["t"],pat["e"]
    if len(t)<8 or t[-1]-t[0]<1.5: continue
    cut=t[0]+0.6*(t[-1]-t[0]); tr=t<=cut; te=~tr
    if te.sum()<2 or tr.sum()<4: continue
    eta=fit_eta(pat,t[tr],e[tr])                       # fit using only the past
    e_pred=predict(q_hat,khf_hat*np.exp(eta),pat["cov"],pat["w"],t[te],e[0])
    rm_model.append(np.sqrt(np.mean((e_pred-e[te])**2)))
    a,b=np.polyfit(t[tr],e[tr],1); e_lin=np.clip(a*t[te]+b,0,G_MAX)  # linear baseline
    rm_lin.append(np.sqrt(np.mean((e_lin-e[te])**2)))
    pred_end.append(e_pred[-1]); obs_end.append(e[te][-1]); min_te.append(e[te].min())
rm_model,rm_lin,min_te=np.array(rm_model),np.array(rm_lin),np.array(min_te)
win=np.mean(rm_model<rm_lin)
print(f"Patients evaluated: {len(rm_model)}")
print(f"Forecast RMSE  MODEL  = {rm_model.mean():.2f} mL/min/1.73m²")
print(f"Forecast RMSE  LINEAR = {rm_lin.mean():.2f} mL/min/1.73m²")
print(f"The model wins in {100*win:.0f}% of patients (overall)")
# stratification: progressors (enter follow-up at low eGFR, where curvature matters)
prog=min_te<45
if prog.sum()>=5:
    wp=np.mean(rm_model[prog]<rm_lin[prog])
    print(f"  · PROGRESSORS (eGFR<45 during follow-up, n={prog.sum()}): "
          f"model {rm_model[prog].mean():.2f} vs linear {rm_lin[prog].mean():.2f}, wins {100*wp:.0f}%")
if (~prog).sum()>=5:
    print(f"  · STABLE (n={(~prog).sum()}): "
          f"model {rm_model[~prog].mean():.2f} vs linear {rm_lin[~prog].mean():.2f}")
corr=np.corrcoef(pred_end,obs_end)[0,1]
print(f"Correlation of predicted vs observed final eGFR: r={corr:.2f}")

# --------------------------------------------------- 3) figure + report
fig,axes=plt.subplots(1,3,figsize=(16,4.6))
# (A) in-sample observed vs predicted
obs_all,pred_all=[],[]
for pat in cohort:
    eta=fit_eta(pat,pat["t"],pat["e"])
    pr=predict(q_hat,khf_hat*np.exp(eta),pat["cov"],pat["w"],pat["t"],pat["e0"])
    obs_all+=list(pat["e"]); pred_all+=list(pr)
obs_all,pred_all=np.array(obs_all),np.array(pred_all)
r2=1-np.sum((obs_all-pred_all)**2)/np.sum((obs_all-obs_all.mean())**2)
axes[0].scatter(obs_all,pred_all,s=6,alpha=0.3,color="#7F77DD")
axes[0].plot([0,120],[0,120],"k--",lw=1); axes[0].set_xlim(0,100); axes[0].set_ylim(0,100)
axes[0].set_xlabel("observed eGFR"); axes[0].set_ylabel("predicted eGFR")
axes[0].set_title(f"(A) Fit  (R²={r2:.2f})")
# (B) forecast examples
shown=0
for pat in cohort:
    t,e=pat["t"],pat["e"]
    if len(t)<10 or t[-1]-t[0]<3 or shown>=5: continue
    cut=t[0]+0.6*(t[-1]-t[0]); tr=t<=cut
    eta=fit_eta(pat,t[tr],e[tr]); tt=np.linspace(0,t[-1],80)
    c=plt.cm.viridis(shown/5)
    axes[1].plot(tt,predict(q_hat,khf_hat*np.exp(eta),pat["cov"],pat["w"],tt,e[0]),color=c,lw=1.6)
    axes[1].scatter(t[tr],e[tr],color=c,s=16); axes[1].scatter(t[~tr],e[~tr],color=c,s=28,marker="x")
    axes[1].axvline(cut,color=c,lw=0.5,ls=":"); shown+=1
axes[1].axhline(15,color="k",lw=1); axes[1].set_xlabel("years"); axes[1].set_ylabel("eGFR")
axes[1].set_title("(B) Forecast: • past (fit)  × future (test)")
# (C) model vs linear RMSE
axes[2].scatter(rm_lin,rm_model,s=14,alpha=0.5,color="#1D9E75")
mx=max(rm_lin.max(),rm_model.max())*1.05
axes[2].plot([0,mx],[0,mx],"k--",lw=1); axes[2].set_xlim(0,mx); axes[2].set_ylim(0,mx)
axes[2].set_xlabel("linear extrapolation RMSE"); axes[2].set_ylabel("mechanistic model RMSE")
axes[2].set_title(f"(C) Forecast: model wins {100*win:.0f}%\n(below diagonal = model is better)")
plt.tight_layout(); plt.savefig("../results/mvp_validation.png",dpi=130)

# .md report
with open("../results/validation_report.md","w") as f:
    f.write(f"""# NephroQ — Validation Report

**Data source:** {"synthetic (demonstration)" if SYNTH else DATA}
**Patients:** {len(cohort)}

## Calibrated population parameters
- Feedback exponent **q = {q_hat:.2f}**
- Hyperfiltration k_hf = {khf_hat:.4f}
- Insult weights (A1c, UACR, SBP) = {np.round(w_hat,4).tolist()}

## Fit
- R² (observed vs predicted) = **{r2:.2f}**

## Forecast validation (fit the past -> predict the future)
- Patients evaluated: {len(rm_model)}
- Forecast RMSE, **mechanistic model = {rm_model.mean():.2f}** mL/min/1.73m²
- Forecast RMSE, linear extrapolation = {rm_lin.mean():.2f} mL/min/1.73m²
- The model beats the line in **{100*win:.0f}%** of patients
- Correlation of predicted vs observed final eGFR: **r = {corr:.2f}**

{"> NOTE: synthetic data. Real validation requires patient data." if SYNTH else "> Validated on real data."}
""")
print("\nSaved: mvp_validation.png  and  validation_report.md")
