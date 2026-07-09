"""Face-validity check of the mechanistic model against REAL PROFILES (S1 Table,
Al-Shamsi 2018). Question: fed with each group's real baseline profile, does the
model predict that the 'developed CKD' group progresses to stage 3 (eGFR<60)
faster than the 'no CKD' group?"""
import numpy as np
G_MAX,ALPHA,N_FLOOR,K0=120.0,0.80,0.0030*0+0.003,0.0030  # k0
def N_of_egfr(e): return (e/G_MAX)**(1/ALPHA)
def egfr_of_N(N): return G_MAX*np.clip(N,1e-9,None)**ALPHA
# population parameters (from the Bayesian calibration)
Q,KHF,W=1.52,0.0141,np.array([0.0144,0.018,0.0108])
def insult(a1c,uacr,sbp): return W[0]*max(a1c-6.5,0)+W[1]*np.log1p(uacr/30)+W[2]*max(sbp-130,0)/10
def years_to_stage3(a1c,sbp,e0,uacr=30,dt=0.02):
    I=insult(a1c,uacr,sbp); N=N_of_egfr(e0); t=0
    while egfr_of_N(N)>60 and t<40:
        Nc=min(max(N,N_FLOOR),1.0); h=min(K0+KHF*(1/Nc)**Q+I,50.0); N=max(N-dt*N*h,N_FLOOR); t+=dt
    return t

print("REAL profiles from the S1 Table (Al-Shamsi 2018, n=491):\n")
print(f"{'Group':<22}{'HbA1c':>7}{'SBP':>6}{'eGFR0':>7}{'years to stage 3 (model)':>26}")
print("-"*68)
for nm,a1c,sbp,e0 in [("Developed CKD (n=56)",8.30,136.7,79.6),
                      ("No CKD (n=435)",6.38,130.7,100.5)]:
    y=years_to_stage3(a1c,sbp,e0)
    print(f"{nm:<22}{a1c:>7.2f}{sbp:>6.1f}{e0:>7.1f}{(f'{y:.1f}' if y<40 else '>40'):>26}")
print("\nThe model should predict much faster progression for the group that did develop CKD.")
