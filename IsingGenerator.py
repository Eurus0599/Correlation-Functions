################################################################################
#
# 06/20/2022                    Ising Model                       Marco Carrillo
#
# QuickIsing.py
#
################################################################################
#
# 'QuickIsing' is a simple implementation of an Ising model in two-dimensions
# most parameters, including the spin field, are set as global variables. An
# Ising model consists of a square lattice whose sites allocate a (scalar) field
# that take +1 or -1 values. The spacing between each lattice site is assumed to
# be unitary. The evolution of the system is given by a Metropolis algorithm.
# This method exploits the correlation of the field at neighboring sites to
# determine the probability of changing their value. Here we consider periodic
# boundary conditions.
#
################################################################################

################################################################################
#                              LIBRARIES                                       #
################################################################################

import numpy as np               ## For arrays and statistics
import matplotlib.pyplot as plt  ## For visualization

################################################################################
#                              LIBRARIES                                       #
################################################################################

################################################################################
#                          GLOBAL VARIABLES                                    #
################################################################################

TIME = 12          # Extension in the temporal dimension
LEN  = 12         # Extension in the spatial dimension
VOL  = TIME * LEN  # Number of sites in the lattice
BETA = 0.4         # 1/T with T an external emperature in units of Boltzman's
                   # constant

THERMAL =  100     # Number of updates before measurements
SEPARE  =   20     # Number of updates between measurements
POINTS  = 1000   # Number of measurements

# Ising model field initialize
field = np.zeros( ( TIME, LEN ) ).astype( int )

################################################################################
#                          GLOBAL VARIABLES                                    #
################################################################################

################################################################################
#                            METHODS BEGIN                                     #
################################################################################

#####
# METHOD:  cold_start
# Takes:   None
# Returns: field

def cold_start():

    # (T,L)-sized array with all sites valued as +1
    field = np.ones( ( TIME, LEN ) ).astype( int )

    return field

# This 'cold_start' method assumes the global variable 'field' and sets all the
# spins to the same value +1.
#####

#####
# METHOD:  hot_start
# Takes:   None
# Returns: field

def hot_start():

    # (T,L)-sized array with all sites valued randomly as +1 or -1
    field = 2 * np.random.randint( 2, size=( TIME, LEN ) ) - 1

    return field

# This 'hot_start' method assumes the global variable 'field' and sets all the
# spins to a random value -1 or +1.
#####

#####
# METHOD:  energy_density
# Takes:   field
# Returns: energy per site

def energy_density( field ):

    # Start the value of the energy as 0
    E_total = 0.0

    # Go over all time-slices
    for t in range( TIME ):

        # Go over all positions
        for x in range( LEN ):

            # Compute contributions from the interactions of the fields at the
            # given site (t,x) and at (t-1,x) and (t,x-1)
            dE_prev_x = field[ t ][ x ] * field[ t ][ (x+LEN-1)%LEN ]
            dE_prev_t = field[ t ][ x ] * field[ (t+TIME-1)%TIME ][ x ]

            # Add contribution to the energy variable
            E_total += ( dE_prev_x + dE_prev_t )

    # Return total energy per site
    return - E_total / ( 2.0 * VOL )

# This 'energy_density' method takes the 'field' and computes the ratio of total
# energy to twice the number of sites in the lattice.
#####

#####
# METHOD:  Metropolis
# Takes:   time, position
# Returns: None

def Metropolis( t, x ):

    # Values of the field at the neares neighbours of site (t,x)
    prev_x = field[ t ][ (x+LEN-1)%LEN ]
    next_x = field[ t ][ ( x + 1 )%LEN ]
    prev_t = field[ (t+TIME-1)%TIME ][ x ]
    next_t = field[ ( t + 1 ) %TIME ][ x ]

    # Energy difference from fliping the field at (t,x)
    dE = 2 * field[t][x] * ( prev_x + next_x + prev_t + next_t )

    # MC step
    if np.random.rand() <= np.exp( - dE * BETA ): field[t][x] *= -1

# This 'Metropolis' method checks the spin at a lattice site and changes it as
# dictated by the Metropolis algorithm. Can be optimized.
#####

#####
# METHOD:  NewConfiguration
# Takes:   None
# Returns: None

def NewConfiguration():

    for t in range( TIME ):
        for x in range( LEN ):
            Metropolis( t, x )

# This 'NewConfiguration' method parses all lattice sites and performs a MC step
# on the spin.
#####

#####
# METHOD:  Thermalize
# Takes:   None
# Returns: None

def Thermalize( no_updates=THERMAL ):

    for _ in range( no_updates ):
        NewConfiguration()

# This 'Thermalize' method updates the field, which started from a hot or cold
# configuration, the necessary times to take it to an equilibrium configuration.
#####

################################################################################
#                            METHODS BEGIN                                     #
################################################################################

################################################################################
#                            MAIN BEGIN                                        #
################################################################################

####################
#    MC Simulation
####################

# Uncomment the starting configuration

#case = " cold start "
#field = cold_start()
case = " hot start "
field = hot_start()

# Thermalize field
Thermalize()

file = open( "Field12.txt", "w" )

# This loop goes over the number of MC experiments to be performed
for n in range( POINTS ):

    # This loops performs a number 'SEPARE' of MC steps
    for j in range( SEPARE ):

        # Perform MC step
        NewConfiguration()

    for time_slice in field:
        for val in time_slice:
            file.write( str(val)+" " )
        file.write( "," )

    if n != POINTS-1: file.write( "\n" )

    #if n: break

file.close()

################################################################################
#                            MAIN END                                          #
################################################################################
