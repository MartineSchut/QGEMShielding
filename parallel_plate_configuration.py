import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cs
import math

### constants
G = cs.gravitational_constant
hbar = cs.hbar
c = cs.speed_of_light
epsilon0 = cs.epsilon_0
kb = cs.Boltzmann
g = 2 # electronic g-factor
mub = 9.3*10**(-24) # Bohr magneton

### some experimental parameters
t = 0.25 # time of creation/annihilation of superpositions
tau = 0.5 # time that the superposition size is constant 
W = 1*10**(-6) # width of hte plate
p = 10**(-2)*1.602176634*10**(-19)*10**(-2) # assumed permanent dipole of the test mass
epsilon=5.7 # dielectric constant test mass
sdensity=3500 # density of the test mass
L = 10**(-3) # length of the plate
dt = 0.001 # time step relevant for numerical integration
timelist = np.arange(dt,(t+tau+t+dt),dt)

### functions
def column(matrix, i):
    return [row[i] for row in matrix]

# function for acceleration due to dipole interaction 
def acc_dd(r,m,p,theta):
    acc_dd = (3*p**2/(4*np.pi*epsilon0*16*m*r**4))*(1+math.cos(theta)**2)
    return acc_dd

# function for acceleration due to casimir interaction 
def acc_cas(r):
    acc = ((3*hbar*c)/(2*np.pi))*((epsilon-1)/(epsilon+2))*(3/(4*np.pi*sdensity*r**5))
    return acc

# function for acceleration due to magnetic field gradient
def dxacc(m,dB):
    dxacc = (g*mub*dB)/m
    return dxacc

# function of infinitesimal gravitational phase for superpositions parallel to the plate
def accphivar(m, dt, r1, r2, delx, W):
    infphase = ((2*G*m**2*dt)/hbar)*(1/np.sqrt((r1 + r2 + W)**2 + delx**2) - 1/(r1 + r2 + W))
    return infphase

# function of infinitesimal gravitational phase in the case of asymmetrically tilted superpositions
def accphivar2(m, dt, x1, x2, delx, W):
    infphase1 = ((G*m**2*dt)/hbar)*(2/np.sqrt((x1 + x2 + W)**2 + delx**2-(x2-x1)**2) - 1/(x1 + x1 + W) - 1/(x2 + x2 + W))
    return infphase1

# function of infinitesimal gravitational phase in the case of symmetrically tilted superpositions
def accphivar3(m, dt, x1, x2, delx, W):
    infphase2 = ((G*m**2*dt)/hbar)*(1/np.sqrt((x1 + x1 + W)**2 + delx**2-(x2-x1)**2) + 1/np.sqrt((x2 + x2 + W)**2 + delx**2-(x2-x1)**2) - 1/(x1 + x2 + W) - 1/(x1 + x2 + W))
    return infphase2

# function that outputs the generated gravitational entanglement phase and final positions for two superpositions initially a distance x1, x2 away from the plate
def phasefunc(m, p, dt, v1, v2, dx, x1, x2, phase):
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            phase = phase + accphivar(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dxmax = dx
            phase = phase + accphivar(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+2*tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            phase = phase + accphivar(mass,dt,x1,x2,dx,W)
            if x1<0:
                print('error')
                break
    return phase, x1, x2

# function that outputs the generated gravitational entanglement phase and final positions for two superpositions that are tilted (a)symmetrically with respect to the plate
def phasefunctilt(mass, p, dt, v1, v2, dx, x1, x2, phase_tilt, phase_tilt2):
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            phase_tilt = phase_tilt + accphivar2(mass,dt,x1,x2,dx,W)
            phase_tilt2 = phase_tilt2 + accphivar3(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dxmax = dx
            phase_tilt = phase_tilt + accphivar2(mass,dt,x1,x2,dx,W)
            phase_tilt2 = phase_tilt2 + accphivar3(mass,dt,x1,x2,dx,W)
        elif timelist[i]<=(t+2*tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            phase_tilt = phase_tilt + accphivar2(mass,dt,x1,x2,dx,W)
            phase_tilt2 = phase_tilt2 + accphivar3(mass,dt,x1,x2,dx,W)
            if x1<0:
                print('error')
                break
    return phase_tilt, phase_tilt2


### dephasing via Casimir and dipole interaction with the plate due to a tilte deltad2.
def dephasefunctilt(mass, p, dt, v1, v2, dx, x1, x2):
    dephase_tilt_cas = 0; dephase_tilt_dip = 0;
    for i in range(len(timelist)):
        if timelist[i]<=t:
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = 0.5*dxacc(mass,flux)*timelist[i]**2
            dephase_tilt_cas = dephase_tilt_cas + (infdephasetiltcas(mass,x1) - infdephasetiltcas(mass,x2))*dt
            dephase_tilt_dip = dephase_tilt_dip + (infdephasetiltdip(mass,x1) - infdephasetiltdip(mass,x2))*dt
        elif timelist[i]<=(t+tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dxmax = dx
            dephase_tilt_cas = dephase_tilt_cas + (infdephasetiltcas(mass,x1) - infdephasetiltcas(mass,x2))*dt
            dephase_tilt_dip = dephase_tilt_dip + (infdephasetiltdip(mass,x1) - infdephasetiltdip(mass,x2))*dt
        elif timelist[i]<=(t+2*tau):
            x1 = x1 - v1*dt - 0.5*acc_dd(x1,mass,p,0)*dt**2 - 0.5*acc_cas(x1)*dt**2
            v1 = v1 + dt*acc_dd(x1,mass,p,0) + dt*acc_cas(x1)
            x2 = x2 - v2*dt - 0.5*acc_dd(x2,mass,p,0)*dt**2 - 0.5*acc_cas(x2)*dt**2
            v2 = v2 + dt*acc_dd(x2,mass,p,0) + dt*acc_cas(x2)
            dx = dxmax - 0.5*dxacc(mass,flux)*(timelist[i]-t-tau)**2
            dephase_tilt_cas = dephase_tilt_cas + (infdephasetiltcas(mass,x1) - infdephasetiltcas(mass,x2))*dt
            dephase_tilt_dip = dephase_tilt_dip + (infdephasetiltdip(mass,x1) - infdephasetiltdip(mass,x2))*dt
    return dephase_tilt_cas, dephase_tilt_dip

def infdephasetiltcas(mass, x):
    # For distance we assume that orientation is 0 --> 1 in the +x direction, and 0 has i.c. x(0) = d-deltad2, and 1 has x(0)=d+deltad2
    Rcubed = (3*mass)/(4*np.pi*sdensity)
    Vc = ((3*c*Rcubed)/(8*np.pi))*((epsilon-1)/(epsilon+2))*(1/(x**4))
    return Vc

def infdephasetiltdip(mass, x):
    # For distance we assume that orientation is 0 --> 1 in the +x direction, and 0 has i.c. x(0) = d-deltad2, and 1 has x(0)=d+deltad2
    Vd = (p**2/(16*np.pi*epsilon0*hbar))*(1/x**3)
    return Vd
