import numpy as np
import scipy.optimize
import scipy.integrate
G=3.925e8 #in Rsun, Msun and yrs
Rsun=6.96e8 #in m
yr=3.15e7 #in seconds

class planet:
    def __init__(self,M,Mp,R,a,e,Rp=0.1):
        self.M=M
        self.Mp=Mp
        self.R=R
        self.a=a 
        self.e=e
        self.Rp=Rp

def findT(eta,pl,t=0): #can be used to find t (when t argument omitted) or solved to find eta (when t supplied) 
    return np.sqrt((pl.a**3)/(G*pl.M)) * (eta - pl.e*np.sin(eta)) - t #=0 when eta is the correct value (used in below root solve)
def findEta(ts,pl):
    if pl.e==0:
        thisEta=np.sqrt(G*pl.M/(pl.a**3))*ts
    else:
        uBound=ts*np.sqrt((G*pl.M) / (pl.a**3)) + 2*pl.e
        lBound=ts*np.sqrt((G*pl.M) / (pl.a**3)) - 2*pl.e
        thisEta=scipy.optimize.fsolve(findT,0.5*(lBound+uBound),args=(pl,ts)) #must find eta by a (well-behaved) numeric root solve
    return thisEta
def findPhi(t,pl):
    eta=findEta(t,pl)
    return 2*np.arctan(np.sqrt((1+pl.e)/(1-pl.e))*np.tan(eta/2)) % (2*np.pi)
def findEtaPhi(Phi,pl): #may never use, but just in case we know Phi and would like to find eta (and subsequently t)
    return 2*np.arctan(np.sqrt((1-pl.e)/(1+pl.e))*np.tan(Phi/2))

def findAlpha(pl):
    beta2=8.723 # calculated from Aleksey Generozov's N=3 polytrope, including modes n=0 to n=5 (i.e. only f and p modes)
    return beta2*(pl.Mp/pl.M)*np.power(pl.R/pl.a,3)/4
def findGamma(t,pl):
    eta=findEta(t,pl)
    alpha=findAlpha(pl)
    return alpha*np.power(1-pl.e*np.cos(eta),-3)
def epsilon(gamma,theta,psi,whichMode=1): #mode=0: just m=0, mode=2: just BOTH m=+-2, mode=1: ALL THREE MODES
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    return gamma*(a*np.power(np.sin(theta)*np.cos(psi),2) + b*np.power(np.sin(theta)*np.sin(psi),2) + c*np.power(np.cos(theta),2))
def findEpsilon(t,theta,phi,pl,whichMode=1): #computes everything you need for epsilon in one 
    Phi=findPhi(t,pl)
    psi=phi-Phi
    gamma=findGamma(t,pl)
    return epsilon(gamma,theta,psi,whichMode=whichMode)
def findMu(t,pl):
    eta=findEta(t,pl)
    alpha=findAlpha(pl)
    return alpha*pl.R*np.sqrt(G*pl.M/pl.a**3)*np.power(1-pl.e*np.cos(eta),-5)
def findVelocity(t,theta,phi,pl,whichMode=1):
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    Phi=findPhi(t,pl)
    psi=phi-Phi
    mu=findMu(t,pl)
    eta=findEta(t,pl)
    lhs=2*np.sqrt(1-pl.e**2)*(np.sin(theta)**2)*np.sin(2*psi)*(a-b)
    rhs=3*pl.e*np.sin(eta)*(a*np.power(np.sin(theta)*np.cos(psi),2) + b*np.power(np.sin(theta)*np.sin(psi),2) + c*np.power(np.cos(theta),2))
    return -mu*(lhs+rhs)
    
def orbitVelocity(t,theta,phi,pl):
    Phi=findPhi(t,pl)
    psi=phi-Phi
    return np.sqrt(G*pl.M/(pl.a*(1-pl.e**2)))*(pl.Mp/pl.M)*np.sin(theta)*(np.sin(psi)+pl.e*np.sin(phi))*(Rsun/yr) # in ms-1
def deltaTide(t,theta,phi,pl,whichMode=1):
    epsilon=findEpsilon(t,theta,phi,pl,whichMode=whichMode)
    return -(5/4)*epsilon
def deltaBeam(t,theta,phi,pl):
    velocity=orbitVelocity(t,theta,phi,pl)
    return 4*velocity/3e8
def deltaReflect(t,theta,phi,pl,Ag=0.1):
    Phi=findPhi(t,pl)
    psi=phi-Phi
    eta=findEta(t,pl)
    cGamma=np.sin(theta)*np.cos(psi)
    sGamma=np.sqrt(1-cGamma**2)
    gamma=np.abs(np.arctan2(sGamma,cGamma))
    num=sGamma+(np.pi-gamma)*cGamma
    denom=np.pi*(1-pl.e*np.cos(eta))
    return Ag*np.power(pl.Rp/pl.a,2)*num/denom

#globally used variable to find thetaMax
vTheta=0
vPsi=0
def thetaMaxDouble(psi): #this (hacky) version should use global vTheta,vPhi, for double integration
    return (np.arctan(-1/(np.tan(vTheta)*np.cos(psi-vPsi))) % np.pi)
def thetaMinDouble(psi):
    return 0
def findNu(gamma,theta,psi,whichMode=1):
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    return np.sqrt(np.power(st*cp/(1 + a*gamma),2) + np.power(st*sp/(1 + b*gamma),2) + np.power(ct/(1 + c*gamma),2))
def nDotL(gamma,theta,psi,whichMode=1):
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    nu=findNu(gamma,theta,psi,whichMode=whichMode)
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    stv, ctv, spv, cpv = np.sin(vTheta), np.cos(vTheta), np.sin(vPsi), np.cos(vPsi)
    return ((st*stv*cp*cpv)/(1 + a*gamma) + (st*stv*sp*spv)/(1 + b*gamma) + (ct*ctv)/(1 + c*gamma))/nu
def nDotL0(theta,psi):
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    stv, ctv, spv, cpv = np.sin(vTheta), np.cos(vTheta), np.sin(vPsi), np.cos(vPsi)
    return st*stv*cp*cpv + st*stv*sp*spv+ ct*ctv
def findH(gamma,theta,psi,whichMode=1):
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    stv, ctv, spv, cpv = np.sin(vTheta), np.cos(vTheta), np.sin(vPsi), np.cos(vPsi)
    return gamma*(a*st*stv*cp*cpv + b*st*stv*sp*spv+ c*ct*ctv)

#to do!
def areaExactIntegrand(theta,psi,gamma,whichMode=1):
    normal=nDotL(gamma,theta,psi,whichMode=whichMode)
    thisEpsilon=epsilon(gamma,theta,psi,whichMode=whichMode)
    return normal*((1+thisEpsilon)**2)*np.sin(theta)
def areaApproxIntegrand(theta,psi,gamma,whichMode=1):
    normal=nDotL0(theta,psi)
    thisEpsilon=epsilon(gamma,theta,psi,whichMode=whichMode)
    h=findH(gamma,theta,psi,whichMode=whichMode)
    return (3*thisEpsilon*normal - h)*np.sin(theta)
def luminosityExactIntegrand(theta,psi,gamma,whichMode=1):
    normal=nDotL(gamma,theta,psi,whichMode=whichMode)
    thisEpsilon=epsilon(gamma,theta,psi,whichMode=whichMode)
    return normal*((1+thisEpsilon)**2)*(1-4*thisEpsilon)*np.sin(theta)
def luminosityApproxIntegrand(theta,psi,gamma,whichMode=1):
    normal=nDotL0(theta,psi)
    thisEpsilon=epsilon(gamma,theta,psi,whichMode=whichMode)
    h=findH(gamma,theta,psi,whichMode=whichMode)
    return (-thisEpsilon*normal - h)*np.sin(theta)
def epsilonExactIntegrand(theta,psi,gamma,whichMode=1):
    normal=nDotL(gamma,theta,psi,whichMode=whichMode)
    thisEpsilon=epsilon(gamma,theta,psi,whichMode=whichMode)
    return normal*thisEpsilon*np.sin(theta)
def epsilonApproxIntegrand(theta,psi,gamma,whichMode=1):
    normal=nDotL0(theta,psi)
    thisEpsilon=epsilon(gamma,theta,psi,whichMode=whichMode)
    return normal*thisEpsilon*np.sin(theta)
def velocityIntegrand(theta,psi,gamma,mu,eta,e,whichMode=1):
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    lhs=2*np.sqrt(1-e**2)*(np.sin(theta)**2)*np.sin(2*psi)*(a-b)
    rhs=3*e*np.sin(eta)*(a*np.power(np.sin(theta)*np.cos(psi),2) + b*np.power(np.sin(theta)*np.sin(psi),2) + c*np.power(np.cos(theta),2))
    normal=nDotL0(theta,psi)
    return -mu*(lhs+rhs)*(normal**2)*np.sin(theta)
def doubleIntegral(t,viewingTheta,viewingPhi,pl,whichIntegrand=5,whichMode=1):
    global vTheta, vPsi
    vTheta=viewingTheta
    Phi=findPhi(t,pl)
    vPsi=(viewingPhi-Phi) % (2*np.pi)
    gamma=findGamma(t,pl)
    if whichIntegrand==0:
        return scipy.integrate.dblquad(epsilonExactIntegrand,0,2*np.pi,thetaMinDouble,thetaMaxDouble,args=(gamma,whichMode))[0]
    elif whichIntegrand==1:
        return scipy.integrate.dblquad(epsilonApproxIntegrand,0,2*np.pi,thetaMinDouble,thetaMaxDouble,args=(gamma,whichMode))[0]
    elif whichIntegrand==2:
        return scipy.integrate.dblquad(areaExactIntegrand,0,2*np.pi,thetaMinDouble,thetaMaxDouble,args=(gamma,whichMode))[0]
    elif whichIntegrand==3:
        return scipy.integrate.dblquad(areaApproxIntegrand,0,2*np.pi,thetaMinDouble,thetaMaxDouble,args=(gamma,whichMode))[0]
    elif whichIntegrand==4:
        return scipy.integrate.dblquad(luminosityExactIntegrand,0,2*np.pi,thetaMinDouble,thetaMaxDouble,args=(gamma,whichMode))[0]
    elif whichIntegrand==5:
        return scipy.integrate.dblquad(luminosityApproxIntegrand,0,2*np.pi,thetaMinDouble,thetaMaxDouble,args=(gamma,whichMode))[0]
    elif whichIntegrand==6:
        mu=findMu(t,pl)
        eta=findEta(t,pl)
        return scipy.integrate.dblquad(velocityIntegrand,0,2*np.pi,thetaMinDouble,thetaMaxDouble,args=(gamma,mu,eta,pl.e,whichMode))[0]
def findAreaApprox(t,vTheta,vPhi,pl):
    Phi=findPhi(t,pl)
    a=1+findEpsilon(t,np.pi/2,vPhi+np.pi/2,pl,whichMode=1)
    b=1+findEpsilon(t,vTheta-np.pi/2,vPhi,pl,whichMode=1)
    return np.pi*a*b
def luminosity(t,vTheta,vPhi,pl,mode=1):
    return 0
def velocity(t,vTheta,vPhi,pl,mode=1):
    return 0