"""
FULL BAYESIAN HIERARCHICAL MODEL v2  ·  identifiable q + corrected bias
- Random effect per patient on METABOLIC SUSCEPTIBILITY (not hyperfiltration):
  this keeps q and k_hf as shared, identifiable physical parameters.
  I_i = exp(eta_i)*insult,  eta_i ~ N(0,tau^2).
- Marginalization of eta via Gauss-Hermite quadrature.
- Adaptive (Haario) covariance Metropolis + precomputed interpolation.
"""
import numpy as np, matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
rng=np.random.default_rng(7)
G_MAX,ALPHA,N_FLOOR,K0,NOISE=120.0,0.80,0.05,0.0030,3.0
def N_of_egfr(e): return np.power(np.clip(e,1e-6,None)/G_MAX,1/ALPHA)
def egfr_of_N(N): return G_MAX*np.power(np.clip(N,1e-9,None),ALPHA)

QT,KT,WT,TAUT=1.60,0.012,np.array([0.0144,0.018,0.0108]),0.45
def insult1(cov,w):
    a,u,s=cov; return w[0]*max(a-6.5,0)+w[1]*np.log1p(u/30)+w[2]*max(s-130,0)/10
def sim1(q,khf,I,tmax,e0,dt=0.05):
    N=N_of_egfr(e0); n=int(tmax/dt)+1; ts=np.linspace(0,tmax,n); Ns=np.empty(n); Ns[0]=N
    for k in range(1,n):
        Nc=min(max(Ns[k-1],N_FLOOR),1.0); h=min(K0+khf*(1/Nc)**q+I,50.0)
        Ns[k]=min(max(Ns[k-1]-dt*Ns[k-1]*h,N_FLOOR),1.0)
    return ts,Ns
def make_cohort(n=80):
    P=[]
    for _ in range(n):
        eta=rng.normal(0,TAUT)
        cov=(rng.uniform(6.5,10),float(np.clip(rng.lognormal(np.log(80),1),5,1500)),rng.uniform(120,165))
        e0=rng.uniform(45,85); I=np.exp(eta)*insult1(cov,WT)   # random effect on susceptibility
        ts,Ns=sim1(QT,KT,I,9,e0); e=egfr_of_N(Ns)
        fu=rng.choice([3.,5.,7.,8.]); tg=np.arange(0,fu+1e-3,0.25); eg=np.interp(tg,ts,e)
        k=eg>12; tg,eg=tg[k],eg[k]; m=rng.random(len(tg))>0.2; tg,eg=tg[m],eg[m]
        if len(tg)<5: continue
        P.append(dict(cov=cov,e0=e0,t=tg,e=np.maximum(eg+rng.normal(0,NOISE,len(eg)),1),eta_true=eta))
    return P
cohort=make_cohort(80); nP=len(cohort)
print(f"{nP} patients (truth q={QT}, k_hf={KT}, tau={TAUT})\n")
covs=np.array([p["cov"] for p in cohort]); N0s=N_of_egfr(np.array([p["e0"] for p in cohort]))
def insultpop(w):
    a,u,s=covs[:,0],covs[:,1],covs[:,2]
    return w[0]*np.maximum(a-6.5,0)+w[1]*np.log1p(u/30)+w[2]*np.maximum(s-130,0)/10

# fixed grid -> precompute per-patient interpolation
DT,TMAX=0.1,8.0; TS=np.arange(0,TMAX+1e-9,DT); NS=len(TS)
IDX=[]; FRAC=[]
for p in cohort:
    t=np.clip(p["t"],0,TMAX-1e-6); j=np.floor(t/DT).astype(int); IDX.append(j); FRAC.append((t-TS[j])/DT)

GH_x,GH_w=hermgauss(5); LOGW=np.log(GH_w/np.sqrt(np.pi))
def egfr_grid(q,khf,w,tau):
    eta=np.sqrt(2)*tau*GH_x; expeta=np.exp(eta)[None,:]      # (1,nn)
    I=insultpop(w)[:,None]*expeta                            # (nP,nn) susceptibility
    N=np.repeat(N0s[:,None],len(GH_x),axis=1); store=np.empty((NS,nP,len(GH_x))); store[0]=N
    def haz(N): Nc=np.clip(N,N_FLOOR,1.0); return np.minimum(K0+khf*(1/Nc)**q+I,50.0)
    for k in range(1,NS):
        f1=-N*haz(N); f2=-(N+0.5*DT*f1)*haz(N+0.5*DT*f1)
        f3=-(N+0.5*DT*f2)*haz(N+0.5*DT*f2); f4=-(N+DT*f3)*haz(N+DT*f3)
        N=np.clip(N+DT/6*(f1+2*f2+2*f3+f4),N_FLOOR,1.0); store[k]=N
    return egfr_of_N(store)
def loglik(q,khf,w,tau):
    eg=egfr_grid(q,khf,w,tau); tot=0.0
    for i,p in enumerate(cohort):
        egi=eg[:,i,:]; j=IDX[i]; fr=FRAC[i][:,None]
        pr=(1-fr)*egi[j]+fr*egi[j+1]                         # (n_obs,nn)
        r=(pr-p["e"][:,None])/NOISE
        tot+=np.logaddexp.reduce(-0.5*np.sum(r**2,axis=0)+LOGW)
    return tot
def unpack(p):
    return 0.5+2.5/(1+np.exp(-p[0])),np.exp(p[1]),np.exp(p[2:5]),np.exp(p[5])
def logprior(p):
    lp=-0.5*((p[0]+0.4)/0.8)**2
    lp+=-0.5*((p[1]-np.log(0.012))/0.9)**2
    lp+=np.sum(-0.5*((p[2:5]-np.log([0.014,0.018,0.011]))/0.9)**2)
    lp+=-0.5*((p[5]-np.log(0.4))/0.7)**2
    return lp
def logpost(p):
    q,khf,w,tau=unpack(p); return loglik(q,khf,w,tau)+logprior(p)

# ---------------- adaptive (Haario) Metropolis ----------------
d=6; p=np.array([-0.4,np.log(0.012),*np.log([0.014,0.018,0.011]),np.log(0.4)])
lp=logpost(p); C=np.diag([0.02,0.02,0.03,0.03,0.03,0.02])**2
sd=2.38**2/d; chain=[]; acc=0; mean=p.copy(); Cemp=C.copy()
N_IT,BURN=6000,2000
print("Sampling (adaptive Haario)...")
for it in range(N_IT):
    L=np.linalg.cholesky(sd*Cemp+1e-9*np.eye(d)) if it>400 else np.linalg.cholesky(C)
    pp=p+L@rng.standard_normal(d); lpp=logpost(pp)
    if np.log(rng.random())<lpp-lp: p,lp,acc=pp,lpp,acc+1
    chain.append(p.copy())
    # online update of empirical mean/covariance
    n=it+1; dlt=p-mean; mean=mean+dlt/n
    Cemp=((n-1)*Cemp+np.outer(dlt,p-mean))/n if n>1 else Cemp
    if (it+1)%1000==0: print(f"  it {it+1:>4}  acc={acc/(it+1):.2f}")
chain=np.array(chain[BURN:])
qs=0.5+2.5/(1+np.exp(-chain[:,0])); khfs=np.exp(chain[:,1]); taus=np.exp(chain[:,5]); ws=np.exp(chain[:,2:5])
def ci(x): return np.percentile(x,[2.5,50,97.5])
print(f"\nFinal acceptance={acc/N_IT:.2f}")
print("\n--- POSTERIOR (median [95% CI]) ---")
print(f"q     = {ci(qs)[1]:.2f}  [{ci(qs)[0]:.2f}, {ci(qs)[2]:.2f}]   (true {QT})")
print(f"k_hf  = {ci(khfs)[1]:.4f} [{ci(khfs)[0]:.4f}, {ci(khfs)[2]:.4f}] (true {KT})")
print(f"tau   = {ci(taus)[1]:.2f}  [{ci(taus)[0]:.2f}, {ci(taus)[2]:.2f}]   (true {TAUT})")
print(f"w_a1c = {ci(ws[:,0])[1]:.4f}  (true {WT[0]})")

fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].hist(qs,bins=40,color="#7F77DD",alpha=0.85,density=True)
axes[0].axvline(QT,color="#1D9E75",lw=2,label=f"true {QT}")
axes[0].axvline(1.06,color="#E24B4A",lw=2,ls="--",label="EM biased 1.06")
axes[0].axvline(ci(qs)[1],color="k",lw=1.5,label=f"posterior {ci(qs)[1]:.2f}")
axes[0].set_xlabel("q"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=9)
axes[0].set_title("(A) Posterior of q (bias corrected)")
axes[1].scatter(qs,khfs,s=6,alpha=0.3,color="#7F77DD")
axes[1].scatter([QT],[KT],color="#1D9E75",s=90,marker="*",zorder=5,label="true")
axes[1].set_xlabel("q"); axes[1].set_ylabel("k_hf"); axes[1].legend(fontsize=9)
axes[1].set_title("(B) Joint posterior q–k_hf")
plt.tight_layout(); plt.savefig("../results/bayesian_model_demo.png",dpi=130)
np.save("../results/post_params.npy",np.array([ci(qs)[1],ci(khfs)[1],*np.median(ws,axis=0),ci(taus)[1]]))
print("\nFigure saved.")
