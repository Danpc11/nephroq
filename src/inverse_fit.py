"""
================================================================================
INVERSE PROBLEM v3  ·  Numerically hardened + verification at ground truth.
theta = [q, k0, k_hf, w_a1c, w_uacr, w_bp]   (alpha fixed, N_ref=1, k_met absorbed)
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

rng = np.random.default_rng(11)
G_MAX, DIALYSIS_eGFR, ALPHA_FIX = 120.0, 15.0, 0.80
N_FLOOR = 0.05

def egfr_of_N(N):  return G_MAX*np.power(np.clip(N, 1e-9, None), ALPHA_FIX)
def N_of_egfr(e):  return np.power(np.clip(e, 1e-6, None)/G_MAX, 1.0/ALPHA_FIX)

def insult(cov, w):
    a1c, uacr, sbp = cov
    return w[0]*max(a1c-6.5,0)+w[1]*np.log1p(uacr/30)+w[2]*max(sbp-130,0)/10

def hazard(N, theta, I):
    q, k0, k_hf = theta[0], theta[1], theta[2]
    N = min(max(N, N_FLOOR), 1.0)
    h = k0 + k_hf*(1.0/N)**q + I
    return min(h, 50.0)                      # cap for stability

def simulate_N(theta, cov, t_max, egfr0, dt=0.02):
    N0 = N_of_egfr(egfr0); I = insult(cov, theta[3:6])
    n = int(np.ceil(t_max/dt))+1
    ts = np.linspace(0, t_max, n); Ns = np.empty(n); Ns[0] = N0
    for k in range(1, n):
        N = Ns[k-1]
        f1 = -N*hazard(N, theta, I)
        f2 = -(N+0.5*dt*f1)*hazard(N+0.5*dt*f1, theta, I)
        f3 = -(N+0.5*dt*f2)*hazard(N+0.5*dt*f2, theta, I)
        f4 = -(N+dt*f3)*hazard(N+dt*f3, theta, I)
        Nn = N + dt/6*(f1+2*f2+2*f3+f4)
        Ns[k] = Nn if np.isfinite(Nn) else N_FLOOR
        Ns[k] = min(max(Ns[k], N_FLOOR), 1.0)
    return ts, Ns

def predict_egfr(theta, cov, tq, egfr0):
    ts, Ns = simulate_N(theta, cov, float(np.max(tq))+0.1, egfr0)
    return np.clip(egfr_of_N(np.interp(tq, ts, Ns)), 0, G_MAX)

# bounded transformations (sigmoid) -> safe ranges, no blow-up
LO = np.array([0.5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
HI = np.array([3.0, 0.05, 0.05, 0.05, 0.05, 0.05])
def unpack(p): return LO + (HI-LO)/(1+np.exp(-p))
def pack(th):
    z = (th-LO)/(HI-LO); z = np.clip(z, 1e-4, 1-1e-4)
    return np.log(z/(1-z))

THETA_TRUE = np.array([1.60, 0.0030, 0.0120, 0.0144, 0.0180, 0.0108])
NOISE_SD = 3.5

def make_patient():
    cov = (rng.uniform(6.5,10), float(np.clip(rng.lognormal(np.log(80),1.0),5,1500)), rng.uniform(120,165))
    egfr0 = rng.uniform(45,85)
    ts, Ns = simulate_N(THETA_TRUE, cov, 8.0, egfr0)
    e = np.interp(np.arange(0,8.01,0.25), ts, egfr_of_N(Ns))
    tg = np.arange(0,8.01,0.25); keep = e > DIALYSIS_eGFR; tg, e = tg[keep], e[keep]
    m = rng.random(len(tg)) > 0.30; t_obs, e_clean = tg[m], e[m]
    return dict(cov=cov, egfr0=egfr0, t=t_obs, e=np.maximum(e_clean+rng.normal(0,NOISE_SD,len(e_clean)),1.0))

patients = [p for p in (make_patient() for _ in range(70)) if len(p["t"])>=6]
train, test = patients[:40], patients[40:60]
print(f"Patients: {len(train)} training, {len(test)} held-out")

def residuals(p, cohort):
    th = unpack(p)
    r = np.concatenate([(predict_egfr(th, q["cov"], q["t"], q["egfr0"]) - q["e"])/NOISE_SD for q in cohort])
    return np.where(np.isfinite(r), r, 100.0)

# --- VERIFICATION AT GROUND TRUTH ---
n_obs = sum(len(p["t"]) for p in train)
r_true = residuals(pack(THETA_TRUE), train)
print(f"\n[sanity check] chi2/n at theta_true = {np.sum(r_true**2)/n_obs:.2f}  (should be ≈ 1)")

# --- fit with multi-start ---
best = None
for s in range(6):
    p0 = pack(np.clip(THETA_TRUE*rng.uniform(0.4,1.8,6), LO*1.01, HI*0.99)) if s>0 \
         else pack(np.array([1.0,0.005,0.008,0.02,0.02,0.02]))
    sol = least_squares(residuals, p0, args=(train,), method="trf", max_nfev=4000)
    if best is None or sol.cost < best.cost: best = sol
sol = best
theta_hat = unpack(sol.x)
print(f"Fit: chi2/n = {2*sol.cost/n_obs:.2f}\n")

J = sol.jac; cov_p = np.linalg.pinv(J.T@J + 1e-9*np.eye(6))
eps=1e-5; Jt=np.zeros((6,6))
for j in range(6):
    dp=sol.x.copy(); dp[j]+=eps; Jt[:,j]=(unpack(dp)-theta_hat)/eps
se = np.sqrt(np.clip(np.diag(Jt@cov_p@Jt.T),0,None))

names=["q","k0","k_hf","w_a1c","w_uacr","w_bp"]
print(f"{'param':<8}{'true':>12}{'estimated':>12}{'±1σ':>12}"); print("-"*44)
for i,nm in enumerate(names):
    print(f"{nm:<8}{THETA_TRUE[i]:>12.4f}{theta_hat[i]:>12.4f}{se[i]:>12.4f}")
print(f"\n>> q = {theta_hat[0]:.2f} ± {se[0]:.2f}  (true {THETA_TRUE[0]:.2f})")

rmse=[np.sqrt(np.mean((predict_egfr(theta_hat,p["cov"],p["t"],p["egfr0"])-p["e"])**2)) for p in test]
print(f"\nHeld-out: mean RMSE = {np.mean(rmse):.2f} mL/min/1.73m²  (noise={NOISE_SD})")

fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]; xs=np.arange(6)
ax.bar(xs-0.2,np.ones(6),0.4,label="true",color="#1D9E75")
ax.bar(xs+0.2,theta_hat/THETA_TRUE,0.4,label="estimated",color="#BA7517",yerr=se/THETA_TRUE,capsize=3)
ax.axhline(1,color="0.6",lw=0.8,ls="--"); ax.set_xticks(xs); ax.set_xticklabels(names,rotation=30,ha="right")
ax.set_ylabel("estimated / true"); ax.set_ylim(0,1.8); ax.set_title("(A) Parameter recovery"); ax.legend(fontsize=9)
ax=axes[1]
for k,pat in enumerate(test[:6]):
    tt=np.linspace(0,max(pat["t"])+0.5,100); c=plt.cm.viridis(k/6)
    ax.plot(tt,predict_egfr(theta_hat,pat["cov"],tt,pat["egfr0"]),color=c,lw=1.8)
    ax.scatter(pat["t"],pat["e"],color=c,s=14,alpha=0.7)
ax.axhline(DIALYSIS_eGFR,color="k",lw=1); ax.set_xlabel("years"); ax.set_ylabel("eGFR")
ax.set_title("(B) Held-out prediction (line) vs data (points)")
plt.tight_layout(); plt.savefig("../results/inverse_fit_demo.png",dpi=130)
print("\nFigure saved.")
