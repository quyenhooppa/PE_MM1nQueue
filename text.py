import simpy
#import random
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import scipy.stats as st
import math
from matplotlib.font_manager import FontProperties
from kneed import KneeLocator

NUM_REP = 5
MAXSIMTIME = 10000
CI = 0.95

#===================INDEPENDENT REPLICATIONS======================#
kn1 = 850
xi = np.array([])
for i in range (NUM_REP):
    temp = np.loadtxt( 'test%d.csv' % i, delimiter='\t')
    # mean = temp[:,1]
    mean = 0
    for j in range (kn1+1, MAXSIMTIME):
        mean += temp[j,1]
    xi = np.append(xi, mean/(MAXSIMTIME-kn1)-1)

xMean = xi.mean()

var = 0
temp = 0
for i in range (NUM_REP):
    temp += (xi[i] - xMean)**2
var = temp / (NUM_REP - 1)

z_scores = 1 - (1 - CI)/2
z = st.norm.ppf(z_scores)

print(xMean)
print(var)
print(z*math.sqrt(var/NUM_REP))