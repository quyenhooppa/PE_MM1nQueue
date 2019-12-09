import simpy
#import random
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import scipy.stats as st
import math
from matplotlib.font_manager import FontProperties
from kneed import KneeLocator

''' ------------------------ '''
''' Parameters               '''
''' ------------------------ '''
# Simulation settings
LOGGED = False
GRAPH = True
VERBOSE = False
MAXSIMTIME = 10000
NUM_REP = 5
CI = 0.95
n0 = 510

# Define parameters for the single queue M/M/1/n
LAMBDA = 5.1000
MU = 3.5000
BUFFER = 800
RHO = LAMBDA/MU


##################################################################
class Job:
    """Definition of a Job object in the queuing system

    Args:
    id (int)        : A unique ID of a job
    arrtime (int)    : Arrival time of a job
    duration (int)   : The time that job is served

    Attributes:
    id (int)        : A unique ID of a job
    arrtime (int)    : Arrival time of a job
    duration (int)   : The time that job is served

    """
    def __init__(self, id, arrtime, duration):
        self.id = id
        self.arrtime = arrtime
        self.duration = duration



##################################################################
class Server:
    """Definition of a Server object in the queuing system

    Args:
        env (simpy.core.Environment):SimPy environment
    Attributes:
        Jobs (list)                 : A queue of jobs
        serversleeping (bool)       : A flag to indicate whether the system is at idle mode or not
        totalServiceTime (int)      : A total time for serving jobs
        waitingTime (int)           : A total time of jobs for waiting to be served
        responseTime (int)          : A total time of jobs from arriving to leaving the server
        idleTime (int)              : A total time that the server is not busy
        jobsSystem (int)            : Number of jobs in the server
        jobsDone (int)              : Number of jobs successfully being served
        jobsDrop (int)              : Number of jobs being rejected by the server 

    """

    def __init__(self, env):
        self.env = env
        self.Jobs = list(())
        self.serversleeping = None
        ''' statistics '''
        self.toltalServiceTime = 0
        self.waitingTime = 0
        self.responseTime = 0
        self.idleTime = 0
        self.jobsSystem = 0
        self.jobsDone = 0
        self.jobsDrop = 0
        
        ''' register a new server process '''
        env.process( self.serve() )

    def serve(self):
        """ Server event of queuing system

        The server takes the first job in the queue to serve.
        When there is no job left in the queue, server will wait until a next job comming.

        Args:
        env (simpy.core.Environment): SimPy environment

        """
        while True:
            if len( self.Jobs ) == 0:
                ''' do nothing, just change server to idle
                and then yield a wait event which takes infinite time
                '''
                self.serversleeping = env.process( self.waiting( self.env ))
                t1 = self.env.now
                yield self.serversleeping
                ''' accumulate the server idle time'''
                self.idleTime += self.env.now - t1
            else:
                ''' get the first job to be served'''
                j = self.Jobs.pop( 0 )

                ''' sum up the waiting time'''
                #print('%d: %.2f' % (self.jobsDone + 1, self.env.now - j.arrtime))
                a = self.env.now - j.arrtime
                self.toltalServiceTime += j.duration
                ''' yield an event for the job finish'''
                yield self.env.timeout( j.duration )
                ''' sum up the jobs done '''
                self.waitingTime += a
                self.responseTime += env.now - j.arrtime
                self.jobsDone += 1
                self.jobsSystem -= 1
    

    def waiting(self, env):
        """ Waiting for a job comming to server

        Args:
        env (simpy.core.Environment):SimPy environment

        """
        try:
            if VERBOSE:
                print( 'Server is idle at %.2f' % self.env.now )
            yield self.env.timeout( MAXSIMTIME )
        except simpy.Interrupt as i:
            if VERBOSE:
                print('Server waken up and works at %.2f' % self.env.now )



##################################################################
class JobGenerator:
    """JobGenerator creates Job object in the queuing system

    Args:
        env (simpy.core.Environment)    : SimPy environment
        lam (float)                       : Arrival rate 
        mu (float)                        : Service rate
    Attributes:
        server (Server)                 : A queue of jobs
        interarrivaltime (float)          : Mean arrival time between two jobs
        servicetime (float)               : Mean service time of a job 

    """
    def __init__(self, env, lam = 2, mu = 1.99999):
        self.server = Server(env)
        self.env = env
        self.interarrivaltime = 1/lam
        self.servicetime = 1/mu

        env.process( self.generatejobs(env) )
        env.process( self.loglog(env))

    def generatejobs(self, env):
        i = 1
        while True:
            '''yield an event for new job arrival'''
            job_interarrival = random.exponential( self.interarrivaltime )
            yield env.timeout( job_interarrival )
            if (len(self.server.Jobs) < BUFFER):
                ''' generate service time and add job to the list'''
                job_duration = random.exponential( self.servicetime )
                self.server.Jobs.append( Job(i, env.now, job_duration) )
                self.server.jobsSystem += 1
                if VERBOSE:
                    print( 'job %d: t = %.2f, l = %.2f, dt = %.2f'
                        %( i, env.now, job_duration, job_interarrival ) )
                i += 1

                ''' if server is idle, wake it up'''
                if not self.server.serversleeping.triggered:
                    self.server.serversleeping.interrupt( 'Wake up, please.' )
            else:
                self.server.jobsDrop += 1
                
    ''' record queue length every 1 unit time'''
    def loglog(self, env):
        while True:
            qlog.write('%d\t%d\t%d\n' %(env.now, len(self.server.Jobs), self.server.jobsSystem))
            yield env.timeout(1)


##################################################################

''' performance metrics '''
jobDrop = 0 
serviceTime = 0
waitTime = 0
utilization = 0
resTime = 0


''' open a log file and save statistics '''
if LOGGED:
    for i in range(NUM_REP):
        qlog = open ( 'test%d.csv'  % i, 'w' )
        env = simpy.Environment()
        MyJobGenerator = JobGenerator( env, LAMBDA, MU )
        env.run( until = MAXSIMTIME )
        jobDrop += MyJobGenerator.server.jobsDrop / MAXSIMTIME
        serviceTime += (MyJobGenerator.server.toltalServiceTime / MyJobGenerator.server.jobsDone)
        waitTime += (MyJobGenerator.server.waitingTime / MyJobGenerator.server.jobsDone)
        resTime += (MyJobGenerator.server.responseTime / MyJobGenerator.server.jobsDone)
        utilization += 1.0-MyJobGenerator.server.idleTime/MAXSIMTIME
        qlog.close()
    log = open ( 'sim.csv', 'w' )
    log.write('%f\t%f\t%f\t%f\t%f\n' % (jobDrop, serviceTime, waitTime, resTime, utilization))
    log.close()


#===================INITIAL DATA DELTION======================#
'''calculate mean across replications'''
xj = []
xs = []
for i in range(MAXSIMTIME):
    xj.append(0.0)
    xs.append(0.0)
for i in range(NUM_REP):
    temp = np.loadtxt( 'test%d.csv' %i, delimiter='\t')
    for j in range(MAXSIMTIME):
        xj[j] += temp[j, 1]
        xs[j] += temp[j, 2]
for i in range(MAXSIMTIME):
    xj[i] = xj[i] / NUM_REP
    xs[i] = xs[i] / NUM_REP

'''mean of all xj'''
xmean = 0.000000
xsystem = 0.0000
for i in range(MAXSIMTIME):
    xmean += xj[i]
    xsystem += xs[i]
xmean = xmean / MAXSIMTIME

'''mean of (n-l) remaing observations and relative change with xmean'''
xl = list()
l = list()
relative = np.array([])
for i in range(MAXSIMTIME-1):
    temp = 0
    for j in range(i+1,MAXSIMTIME):
        temp += xj[j]
    l.append(i)
    xl.append(temp/(MAXSIMTIME - i - 1))
    relative = np.append(relative, ((temp / (MAXSIMTIME - i - 1)) - xmean) / xmean)
kn1 = KneeLocator(l, relative, curve='concave', direction='increasing')
#print(round(kn1.knee,0))

'''average queue length after stable state'''
qLenStable = 0
pos = round(kn1.knee,0)
for i in range (pos, MAXSIMTIME):
    qLenStable += xj[i]
qLenStable = qLenStable / (MAXSIMTIME-pos)


#===================INDEPENDENT REPLICATIONS======================#
xi = np.array([])
for i in range (NUM_REP):
    temp = np.loadtxt( 'test%d.csv' % i, delimiter='\t')
    # mean = temp[:,1]
    mean = 0
    for j in range (n0+1, MAXSIMTIME):
        mean += temp[j,1]
    xi = np.append(xi, mean/(MAXSIMTIME-n0-1))

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

##################################################################
if GRAPH:
    plt.subplot( 2, 2, 1 )
    plt.title('Queue')
    plt.xlabel( 'Time' )
    plt.ylabel( 'Queue length')
    for i in range(NUM_REP):
        log = np.loadtxt( 'test%d.csv' % i, delimiter='\t' )
        plt.step( log[:,0], log[:,1], where='post' )

    plt.subplot( 2, 2, 2 )
    plt.title('Mean across replications')
    plt.xlabel( 'Time' )
    plt.ylabel( 'X$_j$' )
    plt.step( log[:,0], xj, where='post', color = 'orange' )

    plt.subplot( 2, 2, 3 )
    plt.title('Mean X$_l$')
    plt.xlabel( 'l' )
    plt.ylabel( 'X$_l$' )
    plt.step( l, xl, where='post', color = 'black')

    plt.subplot( 2, 2, 4 )
    plt.title('Relative change')
    plt.xlabel( 'l' )
    plt.ylabel( 'change' )
    plt.step( l, relative, where='post', color = 'blue' )
    plt.plot( round(kn1.knee,0), relative[round(kn1.knee,0)], ".", color = "red")


##################################################################
# print('%5.f %d %5.f' % (RHO, BUFFER, RHO**(BUFFER+1)))
# if RHO != 1:
n = float(RHO/(1-RHO) - ((BUFFER+1)*RHO**(BUFFER+1))/(1-RHO**(BUFFER+1)))
nq = float(RHO/(1-RHO) - RHO*(1+BUFFER*RHO**(BUFFER))/(1 - RHO**(BUFFER+1)))
temp = ((1-RHO)/(1-RHO**(BUFFER+1)))*RHO**(BUFFER)
er = n/(LAMBDA*(1-temp))
ew = er - 1/MU

stats = np.loadtxt( 'sim.csv', delimiter='\t' )
jobDrop = stats[0]
serviceTime = stats[1]
waitTime = stats[2]
resTime = stats[3]
utilization = stats[4]
# print(stats)

'''print statistics'''
print('\n------------ Analytical Calculation ------------')
print('Mean no. of jobs in the system:           :%.3f' % (n))
print('Mean no. of jobs in the queue:            :%.3f' % (nq))
print('Utilization of the server                 :%.3f' % (RHO))
print('Mean response time:                       :%.3f' % (er))
print('Mean waiting time:                        :%.3f' % (ew))

print('\n------------ Simulation Calculation ------------')
print( 'Average mean no. of jobs in the system   : %.3f' % round(xsystem/MAXSIMTIME,3) )
print( 'Average mean no. jobs in the queue       : %.3f' % round(xmean,3) )
print( 'Average mean no. of jobs rejected        : %.3f' % round(jobDrop/NUM_REP,3) )
print( 'Average utilization                      : %.3f' % round(utilization/NUM_REP,3) )
print( 'Average service time per job             : %.3f' % round(serviceTime/NUM_REP,3) )
print( 'Average response time per job            : %.3f' % round(resTime/NUM_REP,3) )
print( 'Average waiting time per job             : %.3f' % round(waitTime/NUM_REP,3) )
print( 'Average queue length in stable state     : %.3f '% round(qLenStable,3) )
print( 'Transient length                         : %d\n' % round(kn1.knee,0))

if GRAPH:
    plt.tight_layout()
    plt.show()