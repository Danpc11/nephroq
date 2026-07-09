"""
================================================================================
HIERARCHICAL MODEL (nonlinear mixed effects)  ·  Renal progression
================================================================================
Structure:
  - Shared POPULATION parameters: q, k_hf_pop, insult weights w.
  - RANDOM effect per patient: k_hf_i = k_hf_pop * exp(eta_i),  eta_i ~ N(0, tau^2)
    (individual susceptibility: genetic/unmeasured factors).

Key idea -> PARTIAL POOLING:
  each patient is estimated with a PRIOR centered on the population. With
  many visits, the patient's own data dominates; with FEW visits, the
  estimate "shrinks" toward the population. This is the correct remedy when
  per-patient data is scarce.

Fit: Empirical Bayes via EM
  E-step: MAP of eta_i  (residuals + eta_i^2/tau^2 penalty)
  M-step: update k_hf_pop and tau^2
Compared: NO pooling (each patient alone)  vs  PARTIAL pooling (hierarchical).

Data: uses real CSV if present (see load_real_data); otherwise a synthetic
      cohort mimicking the open UAE dataset (eGFR every 3 months, variable n).
================================================================================
"""
import os, numpy as np
from scipy.optimize import minimize_scalar, least_squares
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
G_MAX, ALPHA, N_FLOOR, K0 = 120.0, 0.80, 0.05, 0.0030
NOISE = 3.5

def N_of_egfr(e): return np.power(np.clip(e,1e-6,None)/G_MAX,1/ALPHA)
def egfr_of_N(N): return G_MAX*np.power(np.clip(N,1e-9,None),ALPHA)
def insult(cov,w):
    a1c,uacr,sbp=cov
    return w[0]*max(a1c-6.5,0)+w[1]*np.log1p(uacr/30)+w[2]*max(sbp-130,0)/10

def simulate(q,k_hf,cov,w,t_max,egfr0,dt=0.05):
    I=insult(cov,w); N=N_of_egfr(egfr0); n=int(t_max/dt)+1
    ts=np.linspace(0,t_max,n); Ns=np.empty(n); Ns[0]=N
    for k in range(1,n):
        Nc=min(max(Ns[k-1],N_FLOOR),1.0)
        h=min(K0+k_hf*(1/Nc)**q+I,50.0)
        Nn=Ns[k-1]-dt*Ns[k-1]*h
        Ns[k]=min(max(Nn if np.isfinite(Nn) else N_FLOOR,N_FLOOR),1.0)
    return ts,Ns

def predict(q,k_hf,pat):
    ts,Ns=simulate(q,k_hf,pat["cov"],pat["w"],float(np.max(pat["t"]))+0.1,pat["egfr0"])
    return np.clip(egfr_of_N(np.interp(pat["t"],ts,Ns)),0,G_MAX)

# ---------------------------------------------------------------- real data
def load_real_data(path):
    """
    Loads a CSV with columns:  patient_id, time_years, egfr, hba1c, uacr, sbp
    (map the EAU/CRIC dataset names to these).  Returns a list of patients.
    """
    import pandas as pd
    df=pd.read_csv(path)
    pats=[]
    for pid,g in df.groupby("patient_id"):
        g=g.sort_values("time_years")
        cov=(g["hba1c"].median(), g["uacr"].median(), g["sbp"].median())
        pats.append(dict(cov=cov, egfr0=float(g["egfr"].iloc[0]),
                         t=g["time_years"].values.astype(float),
                         e=g["egfr"].values.astype(float)))
    return pats

# ---------------------------------------------- synthetic cohort, UAE-like
Q_TRUE, KHF_POP_TRUE, W_TRUE, TAU_TRUE = 1.6, 0.012, np.array([0.0144,0.018,0.0108]), 0.45
def make_cohort(n_pat=300):
    pats=[]
    for _ in range(n_pat):
        eta=rng.normal(0,TAU_TRUE)
        cov=(rng.uniform(6.5,10), float(np.clip(rng.lognormal(np.log(80),1),5,1500)), rng.uniform(120,165))
        egfr0=rng.uniform(45,85)
        khf=KHF_POP_TRUE*np.exp(eta)
        ts,Ns=simulate(Q_TRUE,khf,cov,W_TRUE,8.0,egfr0); e=egfr_of_N(Ns)
        # variable follow-up: visits every 3 months, random length -> dense and sparse patients
        fu=rng.choice([0.75,1.5,3.0,5.0,7.5], p=[0.18,0.22,0.25,0.20,0.15])
        tg=np.arange(0,fu+1e-3,0.25)
        eg=np.interp(tg,ts,e); keep=eg>15; tg,eg=tg[keep],eg[keep]
        if len(tg)<3: continue
        m=rng.random(len(tg))>0.2; tg,eg=tg[m],eg[m]
        if len(tg)<3: continue
        pats.append(dict(cov=cov, egfr0=egfr0, w=W_TRUE.copy(), t=tg,
                         e=np.maximum(eg+rng.normal(0,NOISE,len(eg)),1.0), eta_true=eta))
    return pats

DATA_CSV=os.environ.get("CKD_CSV","")
if DATA_CSV and os.path.exists(DATA_CSV):
    print(f"Using real data: {DATA_CSV}")
    cohort=load_real_data(DATA_CSV)
    for p in cohort: p["w"]=W_TRUE.copy(); p["eta_true"]=np.nan
else:
    print("No real data present -> synthetic UAE-like cohort.")
    cohort=make_cohort(300)
n_visits=np.array([len(p["t"]) for p in cohort])
print(f"{len(cohort)} patients | visits: min {n_visits.min()}, median {int(np.median(n_visits))}, max {n_visits.max()}\n")

# ---------------------------------------------- 1) initial population fit (complete pooling)
def res_pop(p):
    q=0.8+1.7/(1+np.exp(-p[0])); khf=np.exp(p[1]); w=np.exp(p[2:5])
    out=[]
    for pat in cohort:
        pat["w"]=w
        out.append((predict(q,khf,pat)-pat["e"])/NOISE)
    return np.concatenate(out)
sol=least_squares(res_pop,[0,np.log(0.01),*np.log([0.014,0.018,0.011])],method="trf",max_nfev=3000)
q_hat=0.8+1.7/(1+np.exp(-sol.x[0])); khf_hat=np.exp(sol.x[1]); w_hat=np.exp(sol.x[2:5])
for p in cohort: p["w"]=w_hat
print(f"Population:  q={q_hat:.2f} (true {Q_TRUE})   k_hf_pop={khf_hat:.4f} (true {KHF_POP_TRUE})")

# ---------------------------------------------- 2) hierarchical EM for eta_i and tau
def map_eta(pat,khf_pop,tau,penalize=True):
    def obj(eta):
        r=(predict(q_hat,khf_pop*np.exp(eta),pat)-pat["e"])/NOISE
        pen=eta**2/(2*tau**2) if penalize else 0.0
        return 0.5*np.sum(r**2)+pen
    s=minimize_scalar(obj,bounds=(-1.5,1.5),method="bounded")
    return s.x

tau=0.4
for it in range(6):
    etas=np.array([map_eta(p,khf_hat,tau,penalize=True) for p in cohort])
    tau=max(np.sqrt(np.mean(etas**2)),0.05)
    # refine k_hf_pop with the current etas (1D)
    def res_khf(lk):
        khf=np.exp(lk[0])
        return np.concatenate([(predict(q_hat,khf*np.exp(etas[i]),cohort[i])-cohort[i]["e"])/NOISE
                               for i in range(len(cohort))])
    khf_hat=np.exp(least_squares(res_khf,[np.log(khf_hat)],max_nfev=200).x[0])
print(f"Hierarchical:  tau={tau:.2f} (true {TAU_TRUE})   refined k_hf_pop={khf_hat:.4f}\n")

# partial pooling (hierarchical) and no pooling (each patient alone)
eta_partial=np.array([map_eta(p,khf_hat,tau,penalize=True)  for p in cohort])
eta_nopool =np.array([map_eta(p,khf_hat,1e3,penalize=False) for p in cohort])
eta_true=np.array([p["eta_true"] for p in cohort])

if np.isfinite(eta_true).all():
    err_p=np.abs(eta_partial-eta_true); err_n=np.abs(eta_nopool-eta_true)
    print("Error in eta_i by number of visits:")
    print(f"{'visits':>10}{'no pooling':>14}{'partial pooling':>18}")
    for lo,hi in [(3,5),(6,10),(11,20),(21,40)]:
        m=(n_visits>=lo)&(n_visits<=hi)
        if m.sum():
            print(f"{f'{lo}-{hi}':>10}{np.mean(err_n[m]):>14.3f}{np.mean(err_p[m]):>18.3f}  (n={m.sum()})")

# ---------------------------------------------- 3) figure
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]
ax.scatter(n_visits+rng.uniform(-0.3,0.3,len(cohort)),eta_nopool,s=14,alpha=0.4,color="#E24B4A",label="no pooling")
ax.scatter(n_visits+rng.uniform(-0.3,0.3,len(cohort)),eta_partial,s=14,alpha=0.6,color="#1D9E75",label="partial pooling")
ax.axhline(0,color="0.6",lw=0.8,ls="--")
ax.set_xlabel("number of visits per patient"); ax.set_ylabel("estimated η (susceptibility)")
ax.set_title("(A) Shrinkage: with little data, η shrinks toward the population"); ax.legend(fontsize=9)

ax=axes[1]
if np.isfinite(eta_true).all():
    ax.scatter(eta_true,eta_nopool,s=14,alpha=0.4,color="#E24B4A",label="no pooling")
    ax.scatter(eta_true,eta_partial,s=14,alpha=0.6,color="#1D9E75",label="partial pooling")
    lims=[-1.2,1.2]; ax.plot(lims,lims,"k--",lw=1); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("true η"); ax.set_ylabel("estimated η")
    ax.set_title("(B) Partial pooling recovers η better"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("../results/hierarchical_model_demo.png",dpi=130)
print("\nFigure saved.")
