#Print the minimization of the chi square function

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
Time=12
Length=12
vol=Time*Length
Beta=0.4
points=1000
tsep=1      # time separation in the correlators
N_fact=(2*np.pi)/Length   # Exponential Factor
f=np.zeros((Time,Length)).astype( int ) # Initializing the field with all zeroes

## Calculating the conjugate field in the momentum space

def re_1P(t,n):         # Real Part of the conjugate Field
    p=N_fact*n          # N-th momentum
    a=[]                # array of all the lattice points along x multiplied by the exponential factor
    for i in range(Length):
        a.append(f[t][i]* np.cos(p*i))        # Sites along x for a particular time slice
    return np.sum(a)                           # Summation over all x

def im_1P(t,n):         # Real Part of the conjugate Field
    p=N_fact*n          # N-th momentum
    a=[]
    for i in range(Length):
        a.append(f[t][i]* np.sin(-p*i))         # Sites along x for a particular time slice
    return np.sum(a)

file=open("Field12.txt",'r')
n=1             # N-th Momentum
corr=[ [] for t in range( Time ) ]   # Correlators for a particular correlation time of the config

# This 'for' loop goes over all the field configurations
for _ in range( points ):

    # Pick out one entire configuartion of 8X8 lattice
    config = file.readline()

    t=0    # Initializing the time slice

    # This 'for' loop goes over all time-slices
    for time_slice in config.split( ',' ):

        x=0    # Initialing the space coordinate

        for val in time_slice.split():

            f[t][x] = int( val )    # Storing the value of the field at ( t, x )
            x += 1                  # Position step

        t+=1   # Time step

    # This 'for' loop goes over all time-slices
    for t in range( Time ):

        re_O_out = re_1P( t, n )    # Outgoing state (real part)
        im_O_out = im_1P( t, n )    # Outgoing state (imaginary part)

        re_O_in  = re_1P( 0, -n )   # Incoming state (real part)
        im_O_in  = im_1P( 0, -n )   # Incoming state (imaginary part)

        # Two-point correlation function (real and imaginary parts)
        re_C2pt  = re_O_out * re_O_in - im_O_out * im_O_in
        im_C2pt  = im_O_out * re_O_in + re_O_out * im_O_in
        # Add correlator value of the t-th time-slice to the list
        corr[t].append( re_C2pt + 1.0j*im_C2pt )

file.close()


C2pt_mns = []
C2pt_err = []
C2pt_var = []
# Go over all correlation times
t_i=0
t_f=Time//2
for t in range( t_i,t_f ):

    # Compute mean values and standard errors
    C2pt_mns.append( np.mean( corr[t] ) )
    C2pt_err.append( np.std(  corr[t] ) / np.sqrt( points ) )
    C2pt_var.append((C2pt_err[t])**2)

print("The mean is: ",C2pt_mns)
print("The Variance is: ",C2pt_err)


def hypth(t,pars):
    a=pars[0]
    b=pars[1]
    return b*(np.exp(-a*t))

def chi2(a,b):
    pars=[a,b]
    delcorr=[]
    chisq=[]
    for t in range(t_i,t_f):
        h0= hypth(t,pars)
        delcorr.append((C2pt_mns[t]-h0))
        chisq.append((delcorr[t]**2)/C2pt_var[t])
    return np.sum(chisq)

chi=Minuit(chi2,a=0.22188,b=4.75506)
chi.migrad()
print(chi.values)





