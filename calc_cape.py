## Ramanakumar Sankar
## 09/04/2018
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skewt import *
import CAPEfuncs

''' 
    CONSTANTS
    ---------
    These need to be changed for 
    different atmospheres.
    These values are for water on Earth
    (SI Units)
'''

Ratmo = 287.    # Dry gas constant
Rw = 461.       # Species gas constant
Cp = 1004.      # Dry heat capacity
g = 9.81        # Gravity
Lv = 2.501e6    # Latent heat of 
                # vaporization

## initialize the Constants struct
constants = \
    CAPEfuncs.Constants(Ratmo, g, Cp, Rw, Lv)


''' 
    DATA
    ----
    Load the data however you need
    The required variables are:
     -- pressure (in decreasing order)
     -- temperature of the atmosphere
     -- surface mass mixing ratio (q)
     -- dewpoint and sat vapor pressure
        definitions
'''

data = np.loadtxt("sounding.txt",skiprows=4)
nx = data.shape[0]

### convert to SI units
p = data[:,0]*100.
T = data[:,2] + 273.
dewpointldat = data[:,3] + 273.
RH = data[:,4]/100.; qH2O = data[:,5]/1000.

''' 
    k0 is the index to start parcel ascent
    corresponding to the dataset
'''
k0 = 0

'''
    FUNCTIONS
    ---------
    Calculate the saturation vapor pressure
    at a given temperature
    And the dewpoint at a given pressure 
    and mixing ratio (i.e. for a given 
    partial pressure of vapor)
    These values are for water
'''
esat = lambda T: 10.**(12.610 - 2681.18/T)
dewpoint = lambda p, q: 2681.18/(12.610 - \
        np.log10(q*p/(constants.epsilon + q)) \
    )

'''
    CALL THE CAPE CALCULATION
    -------------------------
    This calls the CAPE/CIN calculation
'''
Tparcel, CAPE, CIN, b = \
    CAPEfuncs.do_CAPE(p, T, qH2O, k0, dewpoint, esat, constants)


'''
    OPTIONAL: MAKE A SKEW-T PLOT
'''
## register the skewt projection
register_projection(SkewXAxes)

fig=plt.figure(figsize=(5,5)); ax = fig.add_subplot(111, projection='skewx')
plt.grid(True)

ax.plot(Tparcel, p/100., 'k-')
ax.plot(T, p/100., 'r-')
ax.set_yscale('log')
ax.set_title("CAPE=%.3f"%CAPE)
# # plt.axes().set_xscale('log')
ax.set_xlim((200.,400.))
ax.set_ylim((1000.,100.))

plt.show()
