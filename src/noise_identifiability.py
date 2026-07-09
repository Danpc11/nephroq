"""Demonstrates: lower eGFR measurement noise -> q better identified.
Compares the fit of q at three noise levels corresponding to the choice of
assay: creatinine alone, cystatin, and combined creatinine-cystatin."""
import numpy as np
from scipy.optimize import least_squares
rng=np.random.default_rng(11)
G_MAX,DIAL,ALPHA,N_FLOOR=120.0,15.0,0.80,0.05
def egfr_of_N(N): return G_MAX*np.power(np.clip(N,1e-9,None),ALPHA)
def N_of_egfr(e): return np.power(np.clip(e,1e-6,None)/G_MAX,1/ALPHA)
def insult(cov,w):
    a,u,s=cov; return w[0]*max(a-6.5,0)+w[1]*np.log1p(u/30)+w[2]*max(s-130,0)/10
def sim(th,cov,tmax,e0,dt=0.04):
    q,k0,khf=th[0],th[1],th[2]; I=insult(cov,th[3:6]); N=N_of_egfr(e0)
    n=int(tmax/dt)+1; ts=np.linspace(0,tmax,n); Ns=np.empty(n); Ns[0]=N
    for k in range(1,n):
        Nc=min(max(Ns[k-1],N_FLOOR),1.0); h=min(k0+khf*(1/Nc)**q+I,50.0)
        Nn=Ns[k-1]-dt*Ns[k-1]*h; Ns[k]=min(max(Nn if np.isfinite(Nn) else N_FLOOR,N_FLOOR),1.0)
    return ts,Ns
def pred(th,cov,tq,e0):
    ts,Ns=sim(th,cov,float(np.max(tq))+0.1,e0); return np.clip(egfr_of_N(np.interp(tq,ts,Ns)),0,G_MAX)
LO=np.array([0.5,1e-4,1e-4,1e-4,1e-4,1e-4]); HI=np.array([3,0.05,0.05,0.05,0.05,0.05])
def unpack(p): return LO+(HI-LO)/(1+np.exp(-p))
def pack(t): z=np.clip((t-LO)/(HI-LO),1e-4,1-1e-4); return np.log(z/(1-z))
TH=np.array([1.60,0.0030,0.0120,0.0144,0.0180,0.0108])
def cohort(noise,n=40):
    P=[]
    for _ in range(n):
        cov=(rng.uniform(6.5,10),float(np.clip(rng.lognormal(np.log(80),1),5,1500)),rng.uniform(120,165))
        e0=rng.uniform(45,85); ts,Ns=sim(TH,cov,8,e0); e=egfr_of_N(Ns)
        tg=np.arange(0,8.01,0.25); eg=np.interp(tg,ts,e); k=eg>DIAL; tg,eg=tg[k],eg[k]
        m=rng.random(len(tg))>0.3; tg,eg=tg[m],eg[m]
        if len(tg)<6: continue
        P.append(dict(cov=cov,e0=e0,t=tg,e=np.maximum(eg+rng.normal(0,noise,len(eg)),1)))
    return P
def fit(P,noise):
    def res(p):
        th=unpack(p); r=np.concatenate([(pred(th,q["cov"],q["t"],q["e0"])-q["e"])/noise for q in P])
        return np.where(np.isfinite(r),r,100.)
    best=None
    for s in range(4):
        p0=pack(np.clip(TH*rng.uniform(0.5,1.6,6),LO*1.01,HI*0.99)) if s else pack(np.array([1.,.005,.008,.02,.02,.02]))
        sol=least_squares(res,p0,method="trf",max_nfev=3000)
        if best is None or sol.cost<best.cost: best=sol
    th=unpack(best.x); J=best.jac; cp=np.linalg.pinv(J.T@J+1e-9*np.eye(6))
    eps=1e-5; Jt=np.zeros((6,6))
    for j in range(6):
        dp=best.x.copy(); dp[j]+=eps; Jt[:,j]=(unpack(dp)-th)/eps
    se=np.sqrt(np.clip(np.diag(Jt@cp@Jt.T),0,None)); return th[0],se[0]

print("eGFR assay             noise(σ)   q estimated     ±σ")
print("-"*55)
for name,noise in [("creatinine only",3.5),("cystatin C",2.6),("creat.+cystatin (best)",1.8)]:
    P=cohort(noise); q,se=fit(P,noise)
    print(f"{name:<22}{noise:>6.1f}{q:>12.2f}{se:>9.2f}   (true q=1.60)")
