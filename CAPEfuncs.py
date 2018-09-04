## Ramanakumar Sankar
## 09/04/2018

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fixed_point
from scipy.integrate import odeint, quad
from matplotlib.projections import register_projection

class Constants:
    def __init__(self, Ratmo, g, Cp, Rw, Lv):
        self.Ratmo = Ratmo
        self.Rw = Rw
        self.g = g
        self.Cp = Cp
        self.Lv = Lv
        self.epsilon = epsilon = Ratmo/(Rw)
        self.Kappa = Ratmo/Cp


def get_lcl(p, T, qH2O, k0, dewpoint, esat, constants):
    p0 = p[k0]
    t0 = T[k0]

    q0 =qH2O[k0]
    e0 = q0*p0/(constants.epsilon + q0)
    dewpt0 = dewpoint(p0, q0)

    esdew0 = esat(dewpt0)
    q0 = esdew0/(p0-esdew0)*constants.epsilon

    # def dewpoint(p, q):
    #     e = q*p/(epsilon + q)
    #     Td = 2681.18/(12.610 - np.log10(e))
    #     return Td

    def lcl_min(p):
        Td = dewpoint(p,q0)
        # print(p, Td)
        return p0*(Td/t0)**(1./constants.Kappa)

    ## interpolate the Tp profile so we can call
    ## it as a function of pressure
    Tp = interp1d(np.log10(p), T, kind='cubic')
    # zp = interp1d(np.log10(p), z, kind='cubic')

    ## this is the minimizer where we are trying to solve
    ## for the location where q0 = qsat(p)
    ## where qsat is defined as the saturation mixing ratio

    plcl = fixed_point(lcl_min, p0, maxiter=50)
    # zlcl = zp(np.log10(plcl))
    tlcl = Tp(np.log10(plcl))
    ## get the index of k corresponding to the LCL
    klcl = np.argmin((p - plcl)**2.)

    return (plcl, tlcl, klcl)

def get_parcel_temp(p, T, k0, plcl, klcl, esat, constants):
    p0 = p[k0]
    t0 = T[k0]

    ## integrate the parcel temperature 
    ## with the dry lapse rate

    ## integrate upto the point before
    Tparcel = (T[k0])*np.ones_like(p)
    for k in range(1, klcl+1):
        ## dry lapse rate
        Tparcel[k] = t0*(p[k]/p0)**(constants.Ratmo/constants.Cp)

    ## we then correct the remaining amount based on 
    ## how much more of the atmosphere is dry
    Tparcel[klcl] = t0*(plcl/p0)**(constants.Ratmo/constants.Cp)

    pmin = p.min()
    ## calculate the moist adiabat wrt pressure
    def dTdp_moist(Ti,p):
        esi = esat(Ti)
        qi = esi/(p - esi)*constants.epsilon
        ## This equation comes from [Bakhshaii2013]_.
        dTdp = (1./p)*((constants.Ratmo*Ti + constants.Lv*qi)/\
            (constants.Cp + \
                (constants.Lv*constants.Lv*qi*constants.epsilon/\
                    (constants.Ratmo*Ti*Ti))))
        return dTdp

    ## integrate it for all points from klcl+1 to the end
    pmoist = p[klcl:]
    T0 = Tparcel[klcl]

    Tmoist = odeint(dTdp_moist, T0, pmoist)

    ## set everything above the klcl to be the newly
    ## integrated temperature
    Tparcel[klcl:] = Tmoist[:,0]

    return Tparcel

def get_CAPE_CIN(p, plcl, T, Tparcel, constants):
    ## find the buoyancy:
    ## b = Rdry*(Tatmo - Tparcel)
    ## CAPE = integral b
    b = constants.Ratmo*(Tparcel-T)
    blogp = interp1d(np.log(p), b, kind='cubic')

    ## we need to find the kLFC and kEL
    ## from MetPY:
    ## The LFC could:
        # 1) Not exist
        # 2) Exist but be equal to the LCL
        # 3) Exist and be above the LCL

    ## let's loop up from klcl to the top
    ## and find the LFC
    ## use a high resolution b to find the root
    ptemp = np.linspace(np.log(plcl), np.log(p[-1]), 100)
    btemp = blogp(ptemp)

    plfc = -10
    pel = -10
    for xi, lpi in enumerate(ptemp):
        ## get the EL
        if(btemp[xi] <= 0.):
            if((plfc != -10) & (pel == -10)):
                pel = np.exp(lpi)
                break
        ## get the LFC
        if(btemp[xi] > 0.):
            if(plfc == -10):
                plfc = np.exp(lpi)

    ## integrate b to get the CAPE
    if(plfc==-10):
        ## if there is no LFC, CAPE=0
        CAPE = 0.
        CIN = 0.
    else:
        ## integrate from zlfc to zel 
        ## to get the CAPE
        pCAPE = np.linspace(np.log(plfc), np.log(pel), 50)
        bCAPE = blogp(pCAPE)
        bCAPE[bCAPE < 0.] = 0.
        CAPE = -np.trapz(bCAPE, pCAPE)

        pCIN  = np.linspace(np.log(p[0]), np.log(plfc), 50)
        bCIN  = blogp(pCIN)
        bCIN[bCIN > 0.] = 0.
        CIN = - np.trapz(bCIN, pCIN)

    return CAPE, CIN, b

def do_CAPE(pij, Tij, qH2O, k0, dewpoint, esat, constants):
    plcl, tlcl, klcl = get_lcl(pij, Tij, qH2O, k0, dewpoint, esat, constants)

    Tparcel = get_parcel_temp(pij, Tij, k0, plcl, klcl, esat, constants)

    CAPE, CIN, b = get_CAPE_CIN(pij, plcl, Tij, Tparcel, constants)

    return Tparcel, CAPE, CIN, b