import math, numpy as np
SQ2=math.sqrt(2); CAP=SQ2-1
def f(p,r):
    if p==1: return (1-r)**2/(4*r)
    d=4*p*r*r-(p-1)*(1-r)**2
    return math.inf if d<0 else ((p+1)*r-math.sqrt(d))/(p-1)
def run1d(N):
    r=[0.5]
    for k in range(1,N): r.append(min(CAP,f(N-k,r[-1])))
    rN=min(math.sqrt(N-j)*(1-r[j])-r[j] for j in (N-1,N-2)) if N>2 else None
    return r,rN
