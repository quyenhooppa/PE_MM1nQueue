import simpy
#import random
import numpy as np
from scipy.signal import argrelextrema
import numpy.random as random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from kneed import KneeLocator

''' ------------------------ '''
''' Parameters               '''
''' ------------------------ '''
LOGGED = True
GRAPH = True
VERBOSE = False
POPULATION = 50000000
SERVICE_DISCIPLINE = 'FIFO'

LAMBDA = 2.000
MU = 1.900
RHO = LAMBDA/MU
BUFFER = 20
MAXSIMTIME = 1000
NUM_REP = 4
k = 200

''' ------------------------ '''
''' DES model                '''
''' ------------------------ '''
class Job:
    def __init__(self, id, arrtime, duration):
        self.id = id
        self.arrtime = arrtime
        self.duration = duration

    def __str__(self):
        return 'Job %d at %d, length %d' %(self.id, self.arrtime, self.duration)

def SJF( job ):
    return job.duration

''' A server
 - env: SimPy environment
 - strat: - FIFO: First In First Out
          - SJF : Shortest Job First
'''
class Server:
    def __init__(self, env, strat = 'FIFO'):
        self.env = env
        self.strat = strat
        self.Jobs = list(())
        self.jobsDrop = 0
        self.serversleeping = None
        ''' statistics '''
        self.toltalServiceTime = 0
        self.waitingTime = 0
        self.responseTime = 0
        self.idleTime = 0
        self.jobsGenerate = 0
        self.jobsDone = 0
        
        ''' register a new server process '''
        env.process( self.serve() )

    def serve(self):
        while True:
            ''' do nothing, just change server to idle
              and then yield a wait event which takes infinite time
            '''
            if len( self.Jobs ) == 0:
                self.serversleeping = env.process( self.waiting( self.env ))
                t1 = self.env.now
                yield self.serversleeping
                ''' accumulate the server idle time'''
                self.idleTime += self.env.now - t1
            else:
                ''' get the first job to be served'''
                if self.strat == 'SJF':
                    self.Jobs.sort( key = SJF )
                    j = self.Jobs.pop( 0 )
                else: # FIFO by default
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
    

    def waiting(self, env):
        try:
            if VERBOSE:
                print( 'Server is idle at %.2f' % self.env.now )
            yield self.env.timeout( MAXSIMTIME )
        except simpy.Interrupt as i:
            if VERBOSE:
                print('Server waken up and works at %.2f' % self.env.now )

class JobGenerator:
    def __init__(self, env, strat, nrjobs = 10000000, lam = 2, mu = 1.99999):
        self.server = Server(env, strat)
        self.env = env
        self.nrjobs = nrjobs
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
            self.server.jobsGenerate += 1
            if (len(self.server.Jobs) < BUFFER):
                ''' generate service time and add job to the list'''
                job_duration = random.exponential( self.servicetime )
                self.server.Jobs.append( Job(i, env.now, job_duration) )
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
            qlog.write('%d\t%d\n' %(env.now, len(self.server.Jobs)))
            yield env.timeout(1)

''' performance metrics '''
#jobCreate = 0
jobDone = 0
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
        MyJobGenerator = JobGenerator( env, SERVICE_DISCIPLINE, POPULATION, LAMBDA, MU )
        env.run( until = MAXSIMTIME )
        #jobCreate += MyJobGenerator.server.jobsGenerate 
        jobDone += MyJobGenerator.server.jobsDone
        jobDrop += MyJobGenerator.server.jobsDrop
        serviceTime += (MyJobGenerator.server.toltalServiceTime / MyJobGenerator.server.jobsDone)
        waitTime += (MyJobGenerator.server.waitingTime / MyJobGenerator.server.jobsDone)
        resTime += (MyJobGenerator.server.responseTime / MyJobGenerator.server.jobsDone)
        utilization += 1.0-MyJobGenerator.server.idleTime/MAXSIMTIME
        qlog.close()

'''calculate mean across replications'''
xj = []
for i in range(MAXSIMTIME):
    xj.append(0.0)
for i in range(NUM_REP):
    temp = np.loadtxt( 'test%d.csv' %i, delimiter='\t')
    for j in range(MAXSIMTIME):
        xj[j] += temp[j, 1]
for i in range(MAXSIMTIME):
    xj[i] = xj[i] / NUM_REP

'''mean of all mean xj'''
xmean = 0.000000
for i in range(MAXSIMTIME):
    xmean += xj[i]
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


# xjmean = np.array([])
# j = list()
# for i in range (k, MAXSIMTIME-k-1):
#     temp = 0
#     for l1 in range (-k,k+1):
#         temp += xj[i+l1]
#     j.append(i)
#     xjmean = np.append(xjmean, temp / (2*k+1))
# kn2 = KneeLocator(j, xjmean, curve='concave', direction='increasing')

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
    plt.ylabel( 'X$_j$', rotation='90' )
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
    # knee1 = argrelextrema(relative, np.greater)
    # plt.plot( knee1[0][0]-1, relative[knee1[0][0]-1], ".", color = 'red' )
    plt.plot( round(kn1.knee,0), relative[round(kn1.knee,0)], ".", color = "red")

    # plt.subplot( 3, 2, 5 )
    # plt.title('Moving average')
    # plt.xlabel( 'j' )
    # plt.ylabel( 'Mean X$_j$' )
    # plt.step( j, xjmean, where='post', color = 'purple' )
    # # knee2 = argrelextrema(xjmean, np.greater)
    # # plt.plot( j[knee2[0][0]-1], xjmean[knee2[0][0]-1], ".", color = 'red' )
    # plt.plot( round(kn2.knee,0), xjmean[round(kn2.knee,0)-k], ".", color = "red")

#print('%5.f %d %5.f' % (RHO, BUFFER, RHO**(BUFFER+1)))
n = float(RHO/(1-RHO) - ((BUFFER+1)*RHO**(BUFFER+1))/(1-RHO**(BUFFER+1)))
nq = float(RHO/(1-RHO) - RHO*(1+BUFFER*RHO**(BUFFER))/(1 - RHO**(BUFFER+1)))
pb = ((1-RHO)/(1-RHO**(BUFFER+1)))*RHO**(BUFFER)
er = n/(LAMBDA*(1-pb))
ew = nq/(LAMBDA*(1-pb))

'''print statistics'''
print('\n------------ Analytical Calculation ------------')
print('Mean no. of jobs in the system:           :%.3f' % (n))
print('Mean no. of jobs in the queue:            :%.3f' % (nq))
print('Mean response time:                       :%.3f' % (er))
print('Mean waiting time:                        :%.3f' % (ew))

print('\n------------ Simulation Calculation ------------')
print( 'Average number of jobs in the system     : %.3f' % round(log[:,1].mean(),3) )
print( 'Average number of jobs in the queue      : %.3f' % round(log[:,1].mean()+1,3) )
print( 'Average number of jobs rejected          : %.3f' % round(jobDrop/NUM_REP,3) )
print( 'Average utilization                      : %.3f' % round(utilization/NUM_REP,3) )
print( 'Average service time per job             : %.3f' % round(serviceTime/NUM_REP,3) )
print( 'Average response time per job            : %.3f' % round(resTime/NUM_REP,3) )
print( 'Average waiting time per job             : %.3f' % round(waitTime/NUM_REP,3) )
print( 'Average queue length in stable state     : %.3f '% round(qLenStable,3) )
print( 'Transient length                         : %d' % round(kn1.knee,0))
#print( 'Knee of moving average of independent replications  : %d '% round(kn2.knee,0))

if GRAPH:
    plt.tight_layout()
    plt.show()