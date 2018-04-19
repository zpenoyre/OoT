import numpy as np
import scipy.optimize
import scipy.integrate
Gyr=3.925e8 #in Rsun, Msun and yrs
G=2946 #in Rsun, Msun and days
Rsun=6.96e8 #in m
Msun=2e30 #in kg
yr=3.15e7 #in seconds
day=86400 #in seconds

Mj=0.000954 #in Msun
Me=0.000003 #in Msun
Rj=0.10049  #in Rsun
Re=0.009158 #in Rsun

AU=214.94 #in Rsun

rUnit=1 #planet radii unit in rSun
mUnit=1 #planet mass unit in mSun
tUnit=1 #time unit in days
aUnit=1 #semi-major axis unit in rSun

exact=0

class planet:
    def __init__(self):
        self.M=1 #stellar masses
        self.R=1 #stellar radii
        self.beta=1 #from equipotential theory
        self.Mp=0.001
        self.Rp=0.1
        self.Ag=0.1
        self.a=10
        self.e=0.5
        self.vTheta=np.pi/2
        self.vPhi=0
        self.tp=0 #time of periapse (if using real data with arbitrary t=0)
        
class NonTransitingError(Exception):
    pass

def findT(eta,pl,t=0): #can be used to find t (when t argument omitted) or solved to find eta (when t supplied) [raw unit->days]
    return np.sqrt(((pl.a*aUnit)**3)/(G*pl.M)) * (eta - pl.e*np.sin(eta)) - (t-pl.tp)*tUnit #=0 when eta is the correct value (used in below root solve)
def findPeriod(pl): #returns period in chosen time units [raw units->raw units]
    return findT(2*np.pi,pl)/tUnit - pl.tp
def findTransit(pl): #returns time of center of transit [raw units->raw units]
    eta0=findEtaPhi(pl.vPhi,pl)
    return findT(eta0,pl)/tUnit
def findEta(t,pl): #[raw units->unitless]
    if pl.e==0:
        thisEta=np.sqrt(G*pl.M/((pl.a*aUnit)**3))*(t-pl.tp)*tUnit
    elif exact==0:
        eta0=(t-pl.tp)*tUnit*np.sqrt((G*pl.M) / ((pl.a*aUnit)**3))
        #eta1=pl.e*np.sin(eta0)/(1-pl.e*np.cos(eta0))
        eta1=pl.e*np.sin(eta0)
        eta2=(pl.e**2)*np.sin(eta0)*np.cos(eta0)
        eta3=(pl.e**3)*np.sin(eta0)*(1-(3/2)*(np.sin(eta0)**2))
        thisEta=eta0+eta1+eta2+eta3 #accurate up to and including O(e^3)
    elif exact==1:
        #Below is exact solution, much slower but if needed could be commented back out (and the above 4 lines removed)
        uBound=t*np.sqrt((G*pl.M) / ((aUnit*pl.a)**3)) + 2*pl.e
        lBound=t*np.sqrt((G*pl.M) / ((aUnit*pl.a)**3)) - 2*pl.e
        thisEta=scipy.optimize.fsolve(findT,0.5*(lBound+uBound),args=(pl,t)) #truly accurate but slow, must find eta by a (well-behaved) numeric root solve
    return thisEta
def findPhi(t,pl): #[raw units->unitless]
    eta=findEta(t,pl)
    return 2*np.arctan(np.sqrt((1+pl.e)/(1-pl.e))*np.tan(eta/2)) % (2*np.pi)
def findEtaPhi(Phi,pl): #may never use, but just in case we know Phi and would like to find eta (and subsequently t) [unitless->unitless]
    return 2*np.arctan(np.sqrt((1-pl.e)/(1+pl.e))*np.tan(Phi/2))

def findAlpha(pl): #[raw units->unitless]
    return pl.beta*((mUnit*pl.Mp)/pl.M)*np.power(pl.R/(aUnit*pl.a),3)/4
def findGamma(t,pl): #[raw units->unitless]
    eta=findEta(t,pl)
    alpha=findAlpha(pl)
    return alpha*np.power(1-pl.e*np.cos(eta),-3)
def findEpsilon(t,pl,whichMode=1): #computes everything you need for epsilon in one [raw units->unitless]
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    Phi=findPhi(t,pl)
    psi=pl.vPhi-Phi
    gamma=findGamma(t,pl)
    return gamma*(a*np.power(np.sin(pl.vTheta)*np.cos(psi),2) + b*np.power(np.sin(pl.vTheta)*np.sin(psi),2) + c*np.power(np.cos(pl.vTheta),2))

def findKappa(t,pl): #[raw units -> rSun/day]
    eta=findEta(t,pl)
    alpha=findAlpha(pl)
    return alpha*pl.R*np.sqrt(G*pl.M/(aUnit*pl.a)**3)*np.power(1-pl.e*np.cos(eta),-5)
def findVelocity(t,pl,whichMode=1): #[raw units -> rSun/day]
    if whichMode==1: #return the full expression (you'll almost always want this)
        a,b,c=4,-2,-2
    if whichMode==0: #return just m=0 mode
        a,b,c=1,1,-2
    if whichMode==2: #return m=2 + m=-2 contribution (these are identical, so equivalent to 2*(m=2 contribution))
        a,b,c=(3/2),-(3/2),0
    Phi=findPhi(t,pl)
    psi=pl.vPhi-Phi
    kappa=findKappa(t,pl)
    eta=findEta(t,pl)
    lhs=2*np.sqrt(1-pl.e**2)*(np.sin(pl.vTheta)**2)*np.sin(2*psi)*(a-b)
    rhs=3*pl.e*np.sin(eta)*(a*np.power(np.sin(pl.vTheta)*np.cos(psi),2) + b*np.power(np.sin(pl.vTheta)*np.sin(psi),2) + c*np.power(np.cos(pl.vTheta),2))
    return -kappa*(lhs+rhs)
    
def vOrbit(t,pl): #[raw units->ms-1]
    Phi=findPhi(t,pl)
    psi=pl.vPhi-Phi
    return np.sqrt(G*pl.M/((aUnit*pl.a)*(1-pl.e**2)))*((mUnit*pl.Mp)/pl.M)*np.sin(pl.vTheta)*(np.sin(psi)+pl.e*np.sin(pl.vPhi))*(Rsun/day) # in ms-1
def vTide(t,pl,whichMode=1): #[raw units->ms-1]
    velocity=findVelocity(t,pl,whichMode=whichMode)
    return -(107/240)*velocity*(Rsun/day)
def vSum(t,pl): #[raw units->ms-1]
    return vOrbit(t,pl)+vTide(t,pl)
def deltaTide(t,pl,whichMode=1): #[raw units->unitless]
    epsilon=findEpsilon(t,pl,whichMode=whichMode)
    return -(49+16*(pl.beta**-1))*epsilon/40
def deltaBeam(t,pl): #[raw units->unitless]
    velocity=vOrbit(t,pl)
    return -4*velocity/3e8 #-ve sign as luminosity increased when star is mvoing towards us
def deltaReflect(t,pl,secondary=1): #[raw units->unitless]
    Phi=findPhi(t,pl)
    psi=pl.vPhi-Phi
    eta=findEta(t,pl)
    cGamma=-np.sin(pl.vTheta)*np.cos(psi)
    sGamma=np.sqrt(1-cGamma**2)
    gamma=np.abs(np.arctan2(sGamma,cGamma))
    num=sGamma+(np.pi-gamma)*cGamma
    denom=np.pi*(1-pl.e*np.cos(eta))
    #checking for secondary eclipse
    eclipseFactor=np.ones_like(t)
    if secondary==1:
        r=aUnit*pl.a*(1-pl.e*np.cos(eta)) #in rSun
        d=np.abs(r*np.sqrt(1-np.power(np.sin(pl.vTheta)*np.cos(psi),2))) #projected distance in rSun
        total=np.argwhere((d<(pl.R-pl.Rp)) & ((psi%(2*np.pi)) > np.pi/2) & ((psi%(2*np.pi)) < 3*np.pi/2)) #indices of times when there is a total secondary eclipse
        eclipseFactor[total]=0
        partial=np.argwhere((d<(pl.R+pl.Rp)) & (d>(pl.R-pl.Rp)) & ((psi%(2*np.pi)) > np.pi/2) & ((psi%(2*np.pi)) < 3*np.pi/2)) #indices of times when there is a partial secondary eclipse
        #using circle cirle intersection formula from Mathworld
        firstTerm=((rUnit*pl.Rp)**2)*np.arccos((d**2 + (rUnit*pl.Rp)**2 - pl.R**2)/(2*d*(rUnit*pl.Rp)))
        secondTerm=(pl.R**2)*np.arccos((d**2 + pl.R**2 - (rUnit*pl.Rp)**2)/(2*d*pl.R))
        thirdTerm=-0.5*np.sqrt((d+(rUnit*pl.Rp)+pl.R)*(d+(rUnit*pl.Rp)-pl.R)*(d+pl.R-(rUnit*pl.Rp))*(-d+pl.R+(rUnit*pl.Rp)))
        overlap=(firstTerm+secondTerm+thirdTerm) /(np.pi*(rUnit*pl.Rp)**2)
        eclipseFactor[partial]=1-overlap[partial]
    return eclipseFactor*pl.Ag*np.power((rUnit*pl.Rp)/(aUnit*pl.a),2)*num/denom
def deltaSum(t,pl,secondary=1): #[raw unit->unitless]
    return deltaTide(t,pl)+deltaBeam(t,pl)+deltaReflect(t,pl,secondary=secondary)

def batman(pl): #returns parameters for batman (https://www.cfa.harvard.edu/~lkreidberg/batman/) to use, assuming planet transits
    eta0=findEtaPhi(pl.vPhi,pl)
    per=(findT(2*np.pi,pl)-findT(0,pl))/tUnit #orbital period in days
    t0=findT(eta0,pl)/tUnit  #time of transit in days (between 0 and P), remembering periapse is at t=tp (tp=0 unles set in planet parameters)
    rp=(rUnit*pl.Rp)/pl.R #planet radius in units of stellar radius
    a=(aUnit*pl.a)/pl.R #semi-major axis in units of stellar radius
    inc=(180/np.pi)*pl.vTheta #orbital inclination in degrees (same measure, except planet-people convention is -90 to 90 I believe)
    ecc=pl.e #orbital eccentricity (exactly the same)
    w=(90-((180/np.pi)*pl.vPhi)) % 360 #longitude of periastron in degrees
    b=(aUnit*pl.a)*(1-pl.e*np.cos(eta0))*np.cos(pl.vTheta)/pl.R #projected impact paramter of planet at closest approach
    if b>(pl.R+rUnit*pl.Rp)/pl.R:
        raise NonTransitingError("This planet does not transit!")
    return t0,per,rp,a,inc,ecc,w
    
def setTimeUnits(unit):
    global tUnit
    if unit=='days':
        tUnit=1
    elif unit=='years':
        tUnit=365.242199
    elif unit=='seconds':
        tUnit==1/86400
    else:
        print('time unit not recognized, possible choices are "days" (default), "years" or "seconds"')
        
def setPlanetUnits(unit):
    global rUnit,mUnit
    if unit=='solar':
        rUnit=1
        mUnit=1
    elif unit=='jupiter':
        rUnit=Rj
        mUnit=Mj
    elif unit=='earth':
        rUnit==Re
        mUnit==Me
    else:
        print('planet unit not recognized, possible choices are "solar" (default), "jupiter" or "earth"')
        
def setOrbitUnits(unit):
    global aUnit
    if unit=='solar':
        aUnit=1
    elif unit=='AU':
        aUnit=AU
    else:
        print('orbital distance unit not recognized, possible choices are "solar" (default), "AU"')
        
def demandExactEta(yesNo):
    global exact
    if yesNo==1:
        exact=1
        print('Solution for eta is now exact')
        print('WARNING: This is slow and unnesecary for all but very high eccentricities')
    elif yesNo==0:
        exact=0
        print('Solution for eta is now approximate (though accurate and quick)')
        print('This is probably a good idea for all but very high eccentricities')
    else:
        print('Argument not understood, enter either 0 (for approximate solution) or 1 (for exact)')
    