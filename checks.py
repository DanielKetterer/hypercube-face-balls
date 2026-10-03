import math, numpy as np, mpmath as mp
from map1d import run1d, CAP
SQ2=math.sqrt(2); A=3-2*SQ2; B=SQ2-1
def full(N):
    r=np.empty(N+1); r[0]=0.5
    for k in range(1,N):
        p=N-k; rj=r[:k]; s=k-np.arange(k)
        if p>=2:
            d=4*p*rj*rj-(p-1)*s*(1-rj)**2
            b=np.where(d>=0,((p+1)*rj-np.sqrt(np.maximum(d,0)))/(p-1),np.inf)
        else: b=s*(1-rj)**2/(4*rj)
        r[k]=min(CAP,b.min())
    j=np.arange(N); r[N]=(np.sqrt(N-j)*(1-r[:N])-r[:N]).min(); return r
if __name__=="__main__":
    # 1) fixed point 1/3 and derivative -2
    mp.mp.dps=40
    for p in [2,3,10,100,10**4]:
        f=lambda x: ((p+1)*x-mp.sqrt(4*p*x*x-(p-1)*(1-x)**2))/(p-1)
        print('p',p,' f(1/3)-1/3 =',mp.nstr(f(mp.mpf(1)/3)-mp.mpf(1)/3,5),' f\'(1/3) =',mp.nstr(mp.diff(f,mp.mpf(1)/3),12))
    # 2) full recursion 2..2000: value set, and 1D map agreement
    others=[];mism=[];lastB=None
    for N in range(2,2001):
        r=full(N); v=r[N]
        if abs(v-A)>1e-9 and abs(v-B)>1e-9: others.append(N)
        if abs(v-B)<1e-9: lastB=N
        if N>=10 and N%7==0:
            r1,_=run1d(N)
            if max(abs(np.array(r1)-r[:N]))>1e-12: mism.append(N)
    print('full recursion N=2..2000: interior not in {a,B} at',others,' last B at',lastB,' 1D-map layer mismatches (every 7th N>=10):',mism)
