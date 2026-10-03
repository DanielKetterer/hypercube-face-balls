import sympy as sp, mpmath as mp
mp.mp.dps=50
CAPs=sp.sqrt(2)-1
def beta_sym(p,s,rj):
    if p>=2:
        d=4*p*rj**2-(p-1)*s*(1-rj)**2
        return d, ((p+1)*rj-sp.sqrt(d))/(p-1)
    return None, s*(1-rj)**2/(4*rj)
def solve_exact(N):
    r=[sp.Rational(1,2)]; who=['base']
    for k in range(1,N):
        p=N-k; best=CAPs; bw='cap'; bv=mp.sqrt(2)-1
        for j in range(k):
            d,b=beta_sym(p,k-j,r[j])
            if d is not None and sp.N(d,40)<0: continue
            v=sp.N(b,40)
            if v<bv-mp.mpf(10)**-30: best,bv,bw=b,v,j
        r.append(sp.nsimplify(sp.simplify(best)) if False else sp.simplify(best)); who.append(bw)
    vals=[(sp.N(sp.sqrt(N-j)*(1-r[j])-r[j],40),j) for j in range(N)]
    v,j=min(vals)
    rN=sp.simplify(sp.sqrt(N-j)*(1-r[j])-r[j])
    return r,who,rN,j
rows=[]
for N in list(range(2,13))+[17]:
    r,who,rN,j=solve_exact(N)
    rN_s=sp.radsimp(sp.nsimplify(rN)) if False else rN
    rows.append((N,rN,float(rN),j,(sp.sqrt(N)-1)/2))
    print(N, sp.N(rN,15), 'bound by layer',j, ' exact:', sp.simplify(rN), ' | layers:', [str(sp.N(x,6)) for x in r[1:]])
