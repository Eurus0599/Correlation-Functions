## Printing Chi Square plot and the correlators plot in the same

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
Time=16
Length=16
vol=Time*Length
Beta=0.4
points=20000
tsep=1      # time separation in the correlators
N_fact=(2*np.pi)/Length   # Exponential Factor
f=np.zeros((Time,Length)).astype( int ) # Initializing the field with all zeroes

## Calculating the conjugate field in the momentum space

def re_1P(t,n):         # Real Part of the conjugate Field
    p=N_fact*n          # N-th momentum
    a=[]                # array of all the lattice points along x multiplied by the exponential factor
    for i in range(Length):
        a.append(f[t][i]* np.cos(-p*i))        # Sites along x for a particular time slice
    return np.sum(a)                           # Summation over all x

def im_1P(t,n):         # Real Part of the conjugate Field
    p=N_fact*n          # N-th momentum
    a=[]
    for i in range(Length):
        a.append(f[t][i]* np.sin(-p*i))         # Sites along x for a particular time slice
    return np.sum(a)

file=open("Field1.txt",'r')
n=0             # N-th Momentum

corr=[[] for i in range(Time)]   # Correlators for a particular correlation time of the config

for _ in range(points):
    confg=file.readline()        # Picks out one entire configuartion of 8X8 lattice
    t=0                          # Initialising the time slice
    for time_slice in confg.split( ',' ):       # The time slices are split by commas
        x=0                      # Initialising the space coordinate for each time slice to go over all the lattice points
        for val in time_slice.split():
            f[t][x]=int(val)        # Storing the value of the field config
            x+=1
        t+=1
    for t in range(Time):
        re_C2pt = 0.0
        im_C2pt = 0.0
        re_O_out = re_1P( t, n )  # Outgoing state (real part)
        im_O_out = im_1P( t, n )  # Outgoing state (imaginary part)
        re_O_in  = re_1P( 0, -n )   # Incoming state (real part)
        im_O_in  = im_1P( 0, -n )   # Incoming state (imaginary part)

        # Two-point correlation function contributions
        re_C2pt = ( re_O_out * re_O_in - im_O_out * im_O_in )
        im_C2pt = ( im_O_out * re_O_in + re_O_out * im_O_in )
        
        corr[t].append( ( re_C2pt + 1.0j*im_C2pt ) )
        t+=1
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
print("The Error is: ",C2pt_err)


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

#chi=Minuit(chi2,a=0.22188,b=4.75506)
#chi.migrad()
#print(chi.values)

t_range=np.arange(0.0,8.0+0.02,0.02)
a=0.12181158591959301     # for 16X16 for n=0
b=79.62183940797468
#a=0.14265117654262183   # for 8X8 for n=0
#b=4.60682011991196

#a=0.42271267503752274                      # for 16X16 for n=1
#b=31.442199171779365

#a=0.8152281020036604   # for 8X8 for n=1
#b=7.133780394769291
Corr_fit = hypth( t_range, [ a, b ] )

#print("Fitted function is:",Corr_fit)
#print("Mean:",C2pt_mns)


plt.rcParams.update({'font.size': 18})
plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

# Initiate figure
fig, axs = plt.subplots( 1, 1, figsize=( 12, 8 ) )

axs.plot(t_range,Corr_fit)

# Errorbars of the correlation function (real part)
axs.errorbar(  np.arange(t_i,t_f) , np.real(C2pt_mns), xerr=None, yerr=C2pt_err,
                    mfc='w', ms=4, fmt='o', elinewidth=1.5, zorder=8,
                    capsize=2.0, mew=1.0, alpha=1.0, c='r' )


# Main info of the initial parameters
axs.set_title( r"$V=T\times L=$"+str(Time)+r"$\times$"+str(Length)
               + r"$\quad$ Ising model " + "$\quad$"
               + r"$\beta=$"+str(Beta),
               fontsize=18 )

# Remove top and right borders of the frame
right_side = axs.spines["right"]
right_side.set_visible(False)
top_side = axs.spines["top"]
top_side.set_visible(False)

# Scale of the y-axis
#axs.set_yscale( 'log' )

# Label of the y-axis
axs.set_ylabel( r"$C^{\rm 2pt}_n(t)$" )

# Label of the x-axis and location
axs.set_xlabel( r'$t$', fontsize=18 )
axs.xaxis.set_label_coords(.995, -0.01)

# Limits of the plot
#axs.set_ylim( 1.35, 7.0 )
axs.set_xlim( -0.5, 9.0 )

# Save plot
name = './Ising_C2pt_OneParticle&Chi.pdf'
plt.savefig( name, dpi=100, facecolor='w', edgecolor='w', format='pdf',
        transparent=True, bbox_inches='tight', pad_inches=0.1,
        metadata=None )

# Show plot
plt.show()