"""
================================================================================
AMORTIZED AI ESTIMATOR  ·  "optimize the weights when data is scarce"
================================================================================
Problem: per-patient fitting (least-squares) NEEDS many visits.
           With 3-4 points it is unstable or impossible (5 parameters, 3 data points).

Solution (amortized inference / AI):
  1) Simulate THOUSANDS of patients with parameters drawn from a prior.
  2) Summarize each trajectory into physical features (slope, curvature, ...).
  3) Train an ENSEMBLE of networks (MLP) mapping  features -> parameters.
  4) Instant inference on new patients; with FEW data points the network
     leans on the learned population structure (regularizes toward the prior).

We demonstrate: with few visits, amortized AI OUTPERFORMS and is more stable
than per-patient fitting. Identifiability: we fix k0 (degenerate with k_hf)
and infer theta = [q, k_hf, w_a1c, w_uacr, w_bp].
================================================================================
"""
import numpy as np
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
G_MAX, ALPHA, DIAL, N_FLOOR = 120.0, 0.80, 15.0, 0.05
K0_FIX = 0.0030                      # k0 fixed (degenerate with k_hf) -> infer the rest

def egfr_of_N(N): return G_MAX*np.power(np.clip(N,1e-9,None),ALPHA)
def N_of_egfr(e): return np.power(np.clip(e,1e-6,None)/G_MAX,1/ALPHA)

def insult(cov,w):
    a1c,uacr,sbp=cov
    return w[0]*max(a1c-6.5,0)+w[1]*np.log1p(uacr/30)+w[2]*max(sbp-130,0)/10

def simulate(theta,cov,t_max,egfr0,dt=0.04):
    q,k_hf=theta[0],theta[1]; I=insult(cov,theta[2:5]); N=N_of_egfr(egfr0)
    n=int(t_max/dt)+1; ts=np.linspace(0,t_max,n); Ns=np.empty(n); Ns[0]=N
    for k in range(1,n):
        Nc=min(max(Ns[k-1],N_FLOOR),1.0)
        h=min(K0_FIX+k_hf*(1/Nc)**q+I,50.0)
        Nn=Ns[k-1]-dt*Ns[k-1]*h
        Ns[k]=min(max(Nn if np.isfinite(Nn) else N_FLOOR,N_FLOOR),1.0)
    return ts,Ns

def predict_egfr(theta,cov,tq,egfr0):
    ts,Ns=simulate(theta,cov,float(np.max(tq))+0.1,egfr0)
    return np.clip(egfr_of_N(np.interp(tq,ts,Ns)),0,G_MAX)

# ---------- parameter prior ----------
def sample_theta():
    return np.array([rng.uniform(0.8,2.5),                 # q
                     np.exp(rng.uniform(np.log(0.004),np.log(0.03))),  # k_hf
                     rng.uniform(0.005,0.025),             # w_a1c
                     rng.uniform(0.005,0.030),             # w_uacr
                     rng.uniform(0.003,0.020)])            # w_bp
NOISE=3.5

def sample_patient(n_points=None):
    th=sample_theta()
    cov=(rng.uniform(6.5,10),float(np.clip(rng.lognormal(np.log(80),1.0),5,1500)),rng.uniform(120,165))
    egfr0=rng.uniform(45,85)
    ts,Ns=simulate(th,cov,8.0,egfr0); e=egfr_of_N(Ns)
    if n_points is None: n_points=rng.integers(3,28)
    # random visit times within the observable period (eGFR>15)
    tmax_obs=np.interp(DIAL,e[::-1],ts[::-1]) if e[-1]<DIAL else 8.0
    tobs=np.sort(rng.uniform(0,max(tmax_obs,1.0),n_points))
    eobs=np.maximum(np.interp(tobs,ts,e)+rng.normal(0,NOISE,n_points),1.0)
    return th,cov,egfr0,tobs,eobs

# ---------- physical features of a trajectory ----------
def featurize(cov,egfr0,tobs,eobs):
    n=len(tobs); tspan=tobs[-1]-tobs[0]+1e-3
    A=np.vstack([np.ones(n),tobs,tobs**2]).T
    coef,_,_,_=np.linalg.lstsq(A,eobs,rcond=None)   # robust quadratic fit
    slope,curv=coef[1],coef[2]
    a1c,uacr,sbp=cov
    return [egfr0,eobs[-1],eobs.mean(),slope,curv,n,tspan,
            a1c,np.log1p(uacr/30),(sbp-130)/10]

# ============== 1) generate training data ==============
print("Simulating training cohort...")
NTRAIN=9000
X,Y=[],[]
for _ in range(NTRAIN):
    th,cov,egfr0,tobs,eobs=sample_patient()
    X.append(featurize(cov,egfr0,tobs,eobs)); Y.append(np.log(th))
X,Y=np.array(X),np.array(Y)
scaler=StandardScaler().fit(X); Xs=scaler.transform(X)

# ============== 2) train ENSEMBLE of MLPs (gives uncertainty) ==============
print("Training network ensemble (amortized AI)...")
ensemble=[]
for s in range(5):
    net=MLPRegressor(hidden_layer_sizes=(128,128),activation="relu",
                     alpha=1e-4,max_iter=400,random_state=s)
    net.fit(Xs,Y); ensemble.append(net)

def ai_predict(cov,egfr0,tobs,eobs):
    x=scaler.transform([featurize(cov,egfr0,tobs,eobs)])
    preds=np.array([np.exp(net.predict(x)[0]) for net in ensemble])
    return preds.mean(0),preds.std(0)      # mean and spread (uncertainty)

# ============== 3) overall evaluation (held-out) ==============
names=["q","k_hf","w_a1c","w_uacr","w_bp"]
TH_true=[]; TH_ai=[]
test=[sample_patient() for _ in range(800)]
for th,cov,egfr0,tobs,eobs in test:
    m,_=ai_predict(cov,egfr0,tobs,eobs); TH_true.append(th); TH_ai.append(m)
TH_true,TH_ai=np.array(TH_true),np.array(TH_ai)
print("\nMean relative error (amortized AI, held-out):")
for i,nm in enumerate(names):
    rel=np.mean(np.abs(TH_ai[:,i]-TH_true[:,i])/TH_true[:,i])
    print(f"   {nm:<7}: {100*rel:5.1f}%")

# ============== 4) AI vs per-patient fit, by number of visits ==============
print("\nAmortized AI  vs  per-patient fit (q), by number of visits:")
W_POP=np.array([0.015,0.0175,0.0115])   # weights fixed at population mean
def lsq_per_patient(cov,egfr0,tobs,eobs):
    # FAIR baseline: only fits q and k_hf (2 params) -> runs with few points
    def res(p):
        th=np.array([0.8+1.7/(1+np.exp(-p[0])), np.exp(p[1]), *W_POP])
        return (predict_egfr(th,cov,tobs,egfr0)-eobs)/NOISE
    try:
        s=least_squares(res,np.zeros(2),method="trf",max_nfev=400)
        return 0.8+1.7/(1+np.exp(-s.x[0]))
    except Exception: return np.nan

print(f"{'n_visits':>10}{'err_q AI':>12}{'err_q LSQ':>12}")
npts=[3,4,6,10,20]; ai_err=[]; ls_err=[]
for npt in npts:
    ea,el=[],[]
    for _ in range(200):
        th,cov,egfr0,tobs,eobs=sample_patient(n_points=npt)
        m,_=ai_predict(cov,egfr0,tobs,eobs); ea.append(abs(m[0]-th[0]))
        ql=lsq_per_patient(cov,egfr0,tobs,eobs)
        if np.isfinite(ql): el.append(abs(ql-th[0]))
    ai_err.append(np.mean(ea)); ls_err.append(np.mean(el) if el else np.nan)
    print(f"{npt:>10}{ai_err[-1]:>12.3f}{ls_err[-1]:>12.3f}")

# ============== 5) figure ==============
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]
ax.scatter(TH_true[:,0],TH_ai[:,0],s=10,alpha=0.4,color="#7F77DD")
lims=[0.7,2.6]; ax.plot(lims,lims,"k--",lw=1)
ax.set_xlabel("true q"); ax.set_ylabel("estimated q (AI)")
ax.set_title("(A) Amortized AI recovers the exponent q"); ax.set_xlim(lims); ax.set_ylim(lims)
ax=axes[1]
ax.plot(npts,ai_err,"o-",color="#1D9E75",lw=2,label="amortized AI")
ax.plot(npts,ls_err,"s-",color="#E24B4A",lw=2,label="per-patient fit (2 params)")
ax.set_xlabel("number of visits per patient"); ax.set_ylabel("absolute error in q")
ax.set_title("(B) With scarce data, AI wins"); ax.legend(); ax.set_ylim(0,None)
plt.tight_layout(); plt.savefig("../results/amortized_ai_demo.png",dpi=130)
print("\nFigure saved.")
