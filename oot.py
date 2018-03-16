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
    else:
        eta0=(t-pl.tp)*tUnit*np.sqrt((G*pl.M) / ((pl.a*aUnit)**3))
        eta1=pl.e*np.sin(eta0)/(1-pl.e*np.cos(eta0))
        eta3=(pl.e*np.sin(eta0+eta1) - eta1)/(1-pl.e*np.cos(eta0+eta1))
        thisEta=eta0+eta1+eta3 #accurate up to and including O(e^3)
        #Below is exact solution, much slower but if needed could be commented back out (and the above 4 lines removed)
        #uBound=t*np.sqrt((G*pl.M) / (pl.a**3)) + 2*pl.e
        #lBound=t*np.sqrt((G*pl.M) / (pl.a**3)) - 2*pl.e
        #thisEta=scipy.optimize.fsolve(findT,0.5*(lBound+uBound),args=(pl,t)) #truly accurate but slow, must find eta by a (well-behaved) numeric root solve
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
    return -(49+16*(pl.beta**-1))*epsilon/8
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
    eclipseFactor=1
    if secondary==1:
        r=aUnit*pl.a*(1-pl.e*np.cos(eta)) #in rSun
        d=np.abs(r*np.sqrt(1-np.power(np.sin(pl.vTheta)*np.cos(psi),2))) #in rSun
        d[d>(pl.R+pl.Rp)]=(1-1e-6)*(pl.R+pl.Rp*rUnit) #small correction to avoid unsolvable cos terms
        d[d<(pl.R-pl.Rp)]=(1+1e-6)*(pl.R-pl.Rp*rUnit)
        Atop=((rUnit*pl.Rp)**2)*np.arccos((d**2 + (rUnit*pl.Rp)**2 - pl.R**2)/(2*d*(rUnit*pl.Rp))) + (pl.R**2)*np.arccos((d**2 + pl.R**2 - (rUnit*pl.Rp)**2)/(2*d*pl.R))
        Abottom=-0.5*np.sqrt((d+(rUnit*pl.Rp)+pl.R)*(d+(rUnit*pl.Rp)-pl.Rp)*(d-pl.R+(rUnit*pl.Rp))*(-d+pl.R+(rUnit*pl.Rp)))
        eclipseFactor=(np.pi*(rUnit*pl.Rp)**2 - (Atop + Abottom))/(np.pi*(rUnit*pl.Rp)**2)
        eclipseFactor[eclipseFactor>1]=1
    return eclipseFactor*pl.Ag*np.power((rUnit*pl.Rp)/(aUnit*pl.a),2)*num/denom
def deltaSum(t,pl): #[raw unit->unitless]
    return deltaTide(t,pl)+deltaBeam(t,pl)+deltaReflect(t,pl)

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
        print('orbital distance unit not recognized, possible choices are "solar" (default), "AU")
        

def anchors(pl): #returns parameters for stellar anchor calculations (https://arxiv.org/abs/1710.07293) to use, assuming planet transits
    eta0=findEtaPhi(pl.vPhi,pl)
    per=findT(2*np.pi,pl)-findT(0,pl) #orbital period in days
    t0=(pl.tp+findT(eta0,pl)) % per #time of transit in days (between 0 and P), remembering periapse is at t=tp (tp=0 unles set in planet parameters)
    b=pl.a*(1-pl.e*np.cos(eta0))*np.cos(pl.vTheta)/pl.R #projected impact paramter of planet at closest approach
    if b>(pl.R+pl.Rp)/pl.R:
        raise NonTransitingError("This planet does not transit!")
    rho=3*pl.M*Msun/(4*np.pi*(pl.R*Rsun)**3) #mean stellar density in kgm-3
    logRho=np.log10(rho)
    Rp_R=pl.Rp/pl.R # ratio of planetary to stellar radii
    omega=(np.pi/2)-pl.vPhi 
    cos=np.sqrt(pl.e)*np.cos(omega)
    sin=np.sqrt(pl.e)*np.sin(omega)
    return t0,per,b,logRho,Rp_R,cos,sin
    