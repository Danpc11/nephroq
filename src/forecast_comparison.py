"""Demonstrates the practical improvement: forecasting with a BIASED q (EM=1.06) vs
the BAYESIAN q (1.52). Generates a test cohort (truth q=1.6), fits each patient's
past, and predicts the future."""
import numpy as np
from scipy.optimize import minimize_scalar
rng=np.random.default_rng(99)
G_MAX,ALPHA,N_FLOOR,K0,NOISE=120.0,0.80,0.05,0.0030,3.0
def N_of_egfr(e): return np.power(np.clip(e,1e-6,None)/G_MAX,1/ALPHA)
def egfr_of_N(N): return G_MAX*np.power(np.clip(N,1e-9,None),ALPHA)
QT,KT,WT,TAUT=1.60,0.012,np.array([0.0144,0.018,0.0108]),0.45
def insult(cov,w):
    a,u,s=cov; return w[0]*max(a-6.5,0)+w[1]*np.log1p(u/30)+w[2]*max(s-130,0)/10
def sim(q,khf,I,tmax,e0,dt=0.05):
    N=N_of_egfr(e0); n=int(tmax/dt)+1; ts=np.linspace(0,tmax,n); Ns=np.empty(n); Ns[0]=N
    for k in range(1,n):
        Nc=min(max(Ns[k-1],N_FLOOR),1.0); h=min(K0+khf*(1/Nc)**q+I,50.0)
        Ns[k]=min(max(Ns[k-1]-dt*Ns[k-1]*h,N_FLOOR),1.0)
    return ts,Ns
def pred(q,khf,cov,w,eta,tq,e0):
    I=np.exp(eta)*insult(cov,w); ts,Ns=sim(q,khf,I,float(np.max(tq))+0.1,e0)
    return np.clip(egfr_of_N(np.interp(tq,ts,Ns)),0,G_MAX)
# test cohort
P=[]
for _ in range(200):
    eta=rng.normal(0,TAUT); cov=(rng.uniform(6.5,10),float(np.clip(rng.lognormal(np.log(80),1),5,1500)),rng.uniform(120,165))
    e0=rng.uniform(45,85); ts,Ns=sim(QT,KT,np.exp(eta)*insult(cov,WT),9,e0); e=egfr_of_N(Ns)
    fu=rng.choice([4.,6.,8.]); tg=np.arange(0,fu+1e-3,0.25); eg=np.interp(tg,ts,e); k=eg>12; tg,eg=tg[k],eg[k]
    m=rng.random(len(tg))>0.2; tg,eg=tg[m],eg[m]
    if len(tg)<8: continue
    P.append(dict(cov=cov,e0=e0,t=tg,e=np.maximum(eg+rng.normal(0,NOISE,len(eg)),1)))
def fit_eta(q,khf,cov,w,t,e):
    return minimize_scalar(lambda et:np.sum((pred(q,khf,cov,w,et,t,e[0])-e)**2)+et**2/(2*0.45**2),
                           bounds=(-1.5,1.5),method="bounded").x
def evaluate(q,khf,w):
    rm=[]; mn=[]
    for p in P:
        t,e=p["t"],p["e"]; cut=t[0]+0.6*(t[-1]-t[0]); tr=t<=cut; te=~tr
        if te.sum()<2 or tr.sum()<4: continue
        et=fit_eta(q,khf,p["cov"],w,t[tr],e[tr]); ep=pred(q,khf,p["cov"],w,et,t[te],e[0])
        rm.append(np.sqrt(np.mean((ep-e[te])**2))); mn.append(e[te].min())
    return np.array(rm),np.array(mn)
post=np.load("../results/post_params.npy"); qB,kB,wB=post[0],post[1],post[2:5]
print("Forecast (eGFR RMSE, mL/min/1.73m²):\n")
for name,q,khf,w in [("BIASED q (EM=1.06)",1.06,0.0229,np.array([0.0137,0.0187,0.0112])),
                     (f"BAYESIAN q ({qB:.2f})",qB,kB,wB)]:
    rm,mn=evaluate(q,khf,w); prog=mn<45
    print(f"{name:<24} overall={rm.mean():.2f}   progressors(eGFR<45)={rm[prog].mean():.2f} (n={prog.sum()})")
