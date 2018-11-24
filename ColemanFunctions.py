# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:51:06 2018

@author: bjorn
"""

from numpy import linalg as la
from numpy import random as ran
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import networkx as nx
from collections import OrderedDict as odict
import itertools

def normalize_matrix_row(matrix):
    if matrix.ndim == 2:
        # function for normalizing a matrix over rows
        normed = copy.deepcopy(matrix)
        for i in list(range(0,np.size(matrix,0))):
            for j in list(range(0,np.size(matrix,1))):
                normed[i,j] = matrix[i,j]/np.sum(np.abs(matrix[i,:]))  
        
    elif matrix.ndim == 1:
        # function for normalizing a matrix over rows
        normed = copy.deepcopy(matrix)
        for i in list(range(0,np.size(matrix,0))):
            normed[i] = matrix[i]/np.sum(np.abs(matrix[:]))  
    
    return normed;


''' 
input functions
'''

def SetUp(n,q,m,y,a,c):
    
    outputs = {'n': [n, 'number of actors [n]'],
               'q': [q, 'Number of events [q]'],
               'm': [m, 'Number of resources [m]'],
               'y': [y, 'Directed interests of actor j in event i [y(n,q)]'],
               'x': [np.abs(y), 'Real interests of actor j in event i [x(n,q)]'],
               's': [np.sign(y), 'Directedness of interests of actor j in event i [s(n,q)]'],
               'a': [a, 'Requirement of event i of resource k [a(q,m)]'],               
               'c': [c, 'Control of resource k by actor j [c(m,n)]'],
               }
    return outputs;


def CheckParameters(inputs):
    if not all(np.sum(np.abs(inputs['y'][0]),1) == np.ones((inputs['n'][0],1))):
        #print('Warning! Interest matrix rows did not sum to one. Using normalized data.')
        inputs['y'][0] = normalize_matrix_row(inputs['y'][0])
        inputs['x'][0] = np.abs(inputs['y'][0])
    if not all(np.sum(inputs['a'][0],1) == np.ones((inputs['q'][0],1))):
        #print('Warning! Event resource requirement matrix rows did not sum to one. Using normalized data.')
        inputs['a'][0] = normalize_matrix_row(inputs['a'][0])
    if not all(np.sum(inputs['c'][0],1) == np.ones((inputs['m'][0],1))):
        #print('Warning! Resource control matrix rows did not sum to one. Using normalized data.')
        inputs['c'][0] = normalize_matrix_row(inputs['c'][0])
    
    return inputs;


def standard_variables_techno():
    # standard variables for techno, problem 1 (Coleman)
    
    # actor control over resource
    # variabelnavn: c
    c = np.matrix('1. 0. 0.;' 
                  '0.5 0.25 0.25;'
                  '0.333 0.5 0.167;'
                  '0.1 0.2 0.7')
    
    # actor interest for projects
    # valenced interests: y
    # absolute interests: x  
    y = np.matrix('0.75 0.25;' 
                  '0.25 0.75;' 
                  '0.1 0.9')
    x = normalize_matrix_row(np.abs(y))
    
    # proportion of resource needs for projects
    # variabelnavn: a
    a = np.matrix('0.4 0.333 0.167 0.1;' 
                  '0.25 0.5 0.125 0.125')

    n = np.size(x,0) # number of actors,    3
    q = np.size(x,1) # number of events,    2
    m = np.size(a,1) # number of resources, 4
    
    outputs = SetUp(n,q,m,y,a,c)
        
    return outputs;

def standard_variables_techno2():
    # standard variables for techno, problem 1 (Coleman)
    
    # actor control over resource
    # variabelnavn: c
    c = np.matrix('1. 0. 0.;' 
                  '0. 1. 0.;'
                  '0. 0. 1.')
    
    # actor interest for projects
    # valenced interests: y
    # absolute interests: x  
    y = np.matrix('0.5 0.5;' 
                  '0.333 0.667;' 
                  '0.75 0.25')
    x = normalize_matrix_row(np.abs(y))
    
    # proportion of resource needs for projects
    # variabelnavn: a
    a = np.matrix('0.5 0.250 0.250;' 
                  '0.5 0.333 0.167')

    n = np.size(x,0) # number of actors,    3
    q = np.size(x,1) # number of events,    2
    m = np.size(a,1) # number of resources, 3
    
    outputs = SetUp(n,q,m,y,a,c)
        
    return outputs;


def standard_variables_trad12():
    # standard variables from trad12 (coleman)
    
    # actor control over resource (really over event, since a is unit)
    # variabelnavn: ctT = a*c
    ct = np.matrix('0.3  0.4  0.4  0.25;'  
                  '0.2  0.3  0.4  0.35;' 
                  '0.5  0.3  0.2  0.40')
    c = np.matrix.transpose(ct)

    # actor interest for projects  
    # valenced interests: y
    # absolute interests: x  
    y = np.matrix('0.4  0.2  0.1  0.3;'
                  '-0.3  0.3  0.2  0.2;'
                  '-0.1 -0.3 -0.5  0.1')
    #y = np.matrix.transpose(y)
    x = normalize_matrix_row(np.abs(y))
        
    n = np.size(x,0) # number of actors
    q = np.size(x,1) # number of events
    m = copy.deepcopy(q) #np.size(c,1) # number of resources
    
    # proportion of resource needs for projects
    # variabelnavn: a
    a = np.identity(q)
    

    outputs = SetUp(n,q,m,y,a,c)
        
    return outputs;



''' 
Functions for derived variables 
'''

def solver_rwv(inputs):
    c=inputs['c'][0]
    a=inputs['a'][0]
    x=inputs['x'][0]
    n=inputs['n'][0]
    m=inputs['m'][0]
    q=inputs['q'][0]
    # solve for 'value' (over actors and resources) and ' power' iteratively
    
    # making initial guesses 
    # power of actors
    r = ran.rand(n) 
    r = normalize_matrix_row(r)
    # value of resources
    w = ran.rand(m)
    w = normalize_matrix_row(w)
    # value of events
    v = ran.rand(q)
    v = normalize_matrix_row(v)
    
    # parameters for solver
    correction = 0.01
    change = 10**(-6)
    
    i = 1;
    error_w = 0;
    error_r = 0;
    error_v = 0;
    corre = 1;
    
    if np.sum(np.matrix.diagonal(a))==np.size(a,0):
        # this is executed if there is no resource matrix (set to unit)
        holder = []
        w = np.ones(m)
        while corre>change and i<100:
            
            # Updating guesses
            v_ny = r*x;
            v_ny = normalize_matrix_row(v_ny)
            
            r = v_ny*c
            
            # calculating error and correcting
            error_v = v_ny-v
            v = v + error_v
            
            # calculating total correction
            corre = np.sum(np.abs(error_v))
            holder.append(corre)
            
            # updating counter
            i=i+1
            if i>99:
                print('warning! solver did not converge!')
                
    else:
        aux = []
        aux1 = []
        aux2 = []
        aux3 = []
        while corre>change and i<100:
            
            # Updating guesses
            r_ny = w*c;
            r_ny = normalize_matrix_row(r_ny)
            w_ny = v*a;
            w_ny = normalize_matrix_row(w_ny)
            v_ny = r*x;
            v_ny = normalize_matrix_row(v_ny)
            
            r = w_ny*c
            w = v_ny*a
            v = r_ny*x
            
            # calculating error and correcting
            error_r = r_ny-r
            error_w = w_ny-w
            error_v = v_ny-v
            r = r + error_r*correction
            w = w + error_w*correction
            v = v + error_v*correction
            aux1.append(error_r)
            aux2.append(error_v)
            aux3.append(error_w)
            
            # calculating total correction
            corre = np.sum(np.abs(error_r)) + np.sum(np.abs(error_w)) + np.sum(np.abs(error_v))
            aux.append(corre)
            
            # updating counter
            i=i+1
            if i>99:
                print('warning! solver did not converge!')
            
    outputs = {'r': [r, 'Total power of actor j [r(n)]'],
               'v': [v, 'Value of each event i [v(q)]'],
               'w': [w, 'Value of each resource k [w(m)]']}
        
    return outputs;
    
def ConstitutionalControl(inputs):
    # Finding Constitutional control (given control over resource, and resource requirements)
    C = inputs['a'][0]*inputs['c'][0]
    
    outputs = {'C':[C, 'Constitutional control over event i by actor j [C(q,n)]'] }       
    return outputs;
    
def FractionOfResources(inputs):
    a=inputs['a'][0]
    v=inputs['v'][0]
    w=inputs['w'][0]
    m=inputs['m'][0]
    q=inputs['q'][0]
    # calculates fraction of resource used by actors towards events 
    F = np.zeros((m,q))
    for k in list(range(0,m)):
        for i in list(range(0,q)):
            F[k,i] = a.item(i,k)*v.item(i)/w.item(k)
    outputs = {'F':[F, 'Fraction of resource k used towards event i [F(m,q)]'] }       
    return outputs;
  
def Control_ActorEvent(inputs):
    y=inputs['y'][0]
    r=inputs['r'][0]
    v=inputs['v'][0]
    n=inputs['n'][0]
    q=inputs['q'][0]
    # calculates the final control of actors on events
    c_AE = np.zeros((n,q))
    for j in list(range(0,n)):
        for i in list(range(0,q)):
            c_AE[j,i] = y.item(j,i)*r.item(j)/v.item(i)
            
    outputs ={'c_AE': [c_AE, 'Final control of actor j of each event k [y*(q,n)]']}
    return outputs;


def DerivedInterest(inputs):
    x=inputs['x'][0]
    a=inputs['a'][0]
    n=inputs['n'][0]
    m=inputs['m'][0]
    q=inputs['q'][0]
    # calculates the derived actor interest in each resource
    B = np.zeros((n,m))
    for j in list(range(0,n)):
        for k in list(range(0,m)):
            aux = 0
            for i in list(range(0,q)):
                aux = aux + x.item(j,i)*a.item(i,k)
            B[j,k] = aux
     
    outputs ={'B':[B, 'Derived interest of each actor j in each resource k [B(n,m))]']}
    return outputs;


def Control_ActorResource(inputs):
    B=inputs['B'][0]
    r=inputs['r'][0]
    w=inputs['w'][0]
    n=inputs['n'][0]
    m=inputs['m'][0]
    # calculates the final control of actors over resources
    c_AR = np.zeros((n,m))
    for j in list(range(0,n)):
        for k in list(range(0,m)):
            c_AR[j,k] = B.item(j,k)*r.item(j)/w.item(k)
         
    outputs ={'c_AR':[c_AR, 'Final control of actor j of resource k    [c_AR(m,n)]']}
    return outputs;
    
    
def Control_ActorActor(inputs): 
    # calculates the Direct control of actor on actor
    B=inputs['B'][0]
    c=inputs['c'][0]
    n=inputs['n'][0]
    m=inputs['m'][0]
    
    z = np.zeros((n,n))
    for j in list(range(0,n)):
        for h in list(range(0,n)):
            aux=0;
            for k in list(range(0,m)):
                aux = aux + B.item(j,k)*c.item(k,h)
            z[j,h] = aux
    

    outputs ={'z':[z, 'Final control of actor j by actor h [z(n,n)]'],
              'c_AA':[z, 'Final control of actor j by actor h [c_AA(n,n)]']}
    return outputs;  


def Control_EventEvent(inputs):    
    # Calculating control of event i by event j
    c_EE = inputs['c'][0]*inputs['x'][0]
    
    outputs ={'c_EE':[c_EE, 'Control of event i by event j [c_EE(n,n)]']}
    return outputs;

def PositiveOutcome(inputs):
    # Probability of positive outcome - really dont get this formula
    # variabelnavn : p_i (vektor over events)
    c_AE=inputs['c_AE'][0]
    
    P_p = 0.5 + 0.5*np.sum(c_AE,0)
    
    outputs ={'P_p':[P_p, 'Probability of positive outcome of event j [Pp(q)]']}
    return outputs;
    
    
def Increment_ExpectedRealization(inputs):
    # Calculating the increment in expected realization of interests of actor i from j
    # variabelnavn : p_hj (matrise actor-by-actor)
    s=inputs['s'][0]
    x=inputs['x'][0]
    c_AE=inputs['c_AE'][0]
    q=inputs['q'][0]
    n=inputs['n'][0]
    
    p=np.zeros((n,n))
    for j in list(range(0,n)):
        for h in list(range(0,n)):
            for i in list(range(0,q)):
                p[h,j] = p.item(h,j) + s.item(h,i)*x.item(h,i)*abs(c_AE.item(j,i))*s.item(j,i)
            
            p[h,j] = p.item(h,j)*0.5 # for normalization

    p_hj = copy.deepcopy(p)
    
    
    outputs ={'p_hj':[p_hj, 'increment in expected realization of interests of actor i from j']}
    return outputs;
    
    
def ExpectedValueOfCollectivity(inputs):
    # Expected value of collectivity - not sure about this formula (avrundingsfeil?)
    # variabelnavn : p_h
    p_hj=inputs['p_hj'][0]
    
    p_h = np.sum(p_hj,0) + 0.5;
    
    
    outputs ={'p_h':[p_h, 'Expected value of collectivity']}
    return outputs;
    
    
def ExpectedWeightedRealization(inputs):   
    # Expected wtd realiz of interests of all actors - følgefeil?
    # variabelnavn : d_i
    p_h=inputs['p_h'][0]
    r=inputs['r'][0]
    
    d_i = np.sum(np.multiply(p_h,r))
    
    outputs ={'d_i':[d_i, 'Expected weighted realization of interests of all actors']}
    return outputs;

def DirectedPowerOfCollectivity(inputs):
    # Directed power of collectivity over events
    # variabelnavn : d
    r=inputs['r'][0]
    y=inputs['y'][0]
    
    d = r*y
    
    outputs ={'d':[d, 'Directed power of collectivity on event']}
    return outputs;

def TotalPowerOfCollectivity(inputs):    
    # Total external power of collectivity
    d=inputs['d'][0]
    
    R = np.sum(np.abs(d))
    
    outputs ={'R':[R, 'Total external power of collectivity']}
    return outputs;

def MatchingAttitudes(y):
    s = np.sign(y)
    n = float(len(s))
    aux = np.mean(np.abs(np.sum(s,0)/n))
    if n%2==1:
        minimum = 1/n
    else:
        minimum = 0
        
    rang = 1-minimum
    avgmatch = (aux-minimum)/rang
    
    return avgmatch



'''
coalition functions
'''
def FeasibleCoalitions(inputs):
    n = inputs['n'][0]
    c = inputs['c'][0]
    #creating matrix of all possible coalitions    
    if n<9:
        LOLI = np.unpackbits(np.arange(2**n).astype(np.uint8)[:, None], axis = 1)[:, -n:]
    elif n<19:
        LOLI = []
        for i in list(range(0,2**n)):
            aux = format(i, '#0'+str(n+2)+'b')
            LOLI.append([int(b) for b in aux[2:]])
                           
    else:
        print('can only generate coalitions for n<=18. To many coalitions!')
        
    actors = np.array(list(range(1,n+1)))
    coalitions = actors*LOLI
    
    feasible = []
    CoalitionDict = odict({})
    for i in list(range(0,2**n)):
        coal = []
        for j in list(range(0,n)):
            if coalitions[i,j]>0:
                coal.append(coalitions[i,j]-1)
        coalitioncontrol = 0
        if len(coal)>1:
            for cc in coal:
                coalitioncontrol += c[0,cc]
        if coalitioncontrol>0.5:
            feasible.append(coal)
     
            CoalitionDict.update({str(coal): [coal, coalitioncontrol]}) 
    
    
    return CoalitionDict



def CoalitionTrad(inputs,coalitions):
    q = inputs['q'][0]
    m = inputs['m'][0]
    y = inputs['y'][0]
    c = inputs['c'][0]    
    a = inputs['a'][0]
    
    outputs = {}
    for key in list(coalitions.keys()):
        
        idx = coalitions[key][0]
        n = len(idx)
        yy = y[idx]
        cc = c[:,idx]
        
        # doing trad calculations for coalition
        subsystem = SetUp(n,q,m,yy,a,cc)
        subsystem.update(RunFullAnalysis(subsystem))
        outputs[key] = subsystem

    return outputs


def CoalitionTechno(inputs,coalitions):
    q = inputs['q'][0]
    m = len(list(coalitions.keys()))
    n = inputs['n'][0]
    y = inputs['y'][0]    
    a = np.matrix(np.zeros((q,m)))
    c = np.matrix(np.zeros((m,n)))
    cc = inputs['c'][0]
    
    outputs = {}
    i = 0
    totalControl = 0
    for key in list(coalitions.keys()):
        
        idx = coalitions[key][0]
        members = float(len(idx))
        coalitionControl = 0
        for id in idx:
            # calculating average control across issues for each actor in coalition
            c[i,id] = np.sum(cc[:,id])/q
            coalitionControl += c[i,id]    
            totalControl += coalitionControl
            
        print(coalitionControl)
        for id in list(range(0,q)):
            a[id,i] = coalitionControl/float(totalControl)
        i+=1
    print(a)
    outputs = SetUp(n,q,m,y,a,c)
    outputs.update(RunFullAnalysis(outputs))
     
    
    return outputs



def CoalitionTechno_EqualControl(inputs,coalitions):
    q = inputs['q'][0]
    m = len(list(coalitions.keys()))
    n = inputs['n'][0]
    y = inputs['y'][0]    
    a = np.matrix(np.zeros((q,m)))
    c = np.matrix(np.zeros((m,n)))
    cc = inputs['c'][0]
    
    outputs = {}
    i = 0
    for key in list(coalitions.keys()):
        
        idx = coalitions[key][0]
        members = float(len(idx))
        for id in idx:
            c[i,id] = 1/members
        
        for id in list(range(0,q)):
            a[id,i] = np.sum(cc[idx,id])
        i+=1
      
    outputs = SetUp(n,q,m,y,a,c)
    outputs.update(RunFullAnalysis(outputs))
    
    return outputs

def RunCoalitionAnalysis(n,q,m,y,a,c,descriptor):
    
    # running analysis with real interests
    outputs = SetUp(n,q,m,y,a,c)
    outputs.update(RunFullAnalysis(outputs))
    
    # saving results from full model
    fullModel = copy.deepcopy(outputs)
    
    
    # finding feasible coalitions
    coalitions = FeasibleCoalitions(outputs)
    
    
    # running trad analysis for coalitions
    coalitionOutputs_Trad = CoalitionTrad(outputs,coalitions)
    
    # printing all the coalition results to file
    keynames = list(coalitionOutputs_Trad.keys())
    order = ['n','q','C','y','z','c_EE','v','r','c_AE','P_p','p_hj','p_h','d_i','d','R']
    filepath=r"C:\Users\bjorn\OneDrive\Documents\Skole og arbeidsliv\projects\GudmundStanford\scripts\Outputs" 
    title='trad_coalition_'+descriptor+'_'
    for i in list(range(0,len(keynames))):
        OrderedOutput_file(filepath,title+keynames[i],coalitionOutputs_Trad[keynames[i]],order)
    
    
    # Running adjusted techno analysis
    coalitionOutputs_Techno = CoalitionTechno(outputs,coalitions)
    
    # printing files for techno
    order = ['n','q','m','C','y','a','c','B','v','r','w','P_p','p_hj','p_h','d_i','d','R']
    #['n','q','m','x','a','c','r','v','w','F','c_AE','B','c_AR','z']
    filepath=r"C:\Users\bjorn\OneDrive\Documents\Skole og arbeidsliv\projects\GudmundStanford\scripts\Outputs" 
    title='techno_Coalition'+descriptor+'_'
    OrderedOutput_file(filepath,title,coalitionOutputs_Techno,order)
    
    return coalitionOutputs_Trad, coalitionOutputs_Techno


def RunCoalitionAnalysis_noOutput(n,q,m,y,a,c):
    
    # running analysis with real interests
    outputs = SetUp(n,q,m,y,a,c)
    outputs.update(RunFullAnalysis(outputs))
    
    # saving results from full model
    fullModel = copy.deepcopy(outputs)
    
    
    # finding feasible coalitions
    coalitions = FeasibleCoalitions(outputs)
    
    
    # running trad analysis for coalitions
    coalitionOutputs_Trad = CoalitionTrad(outputs,coalitions)
    
    return coalitionOutputs_Trad


def ValueForOpposition(inputs,actor,Current,full):
    P = inputs[Current]['P_p'][0]
    y = inputs[full]['y'][0][actor,:]
    return np.sum(np.multiply(P*2-1,y))/float(np.size(y))
    
    
def ValueOfCoalition(inputs):
    keys = list(inputs.keys())
    n = inputs[keys[-1]]['n'][0]
    coalValue = {}
    for key in keys:
        coalmembers = [np.matrix(key)[0,i] for i in list(range(0,np.size(np.matrix(key))))]
        vals = [ValueForOpposition(inputs,i,key,keys[-1]) for i in list(range(0,n))]
        valCoal = np.sum([vals[i] for i in coalmembers])
        coalValue.update({key: 
            {'to actors': vals,
            'to coalition':valCoal,
            'to opposition':np.sum(vals)-valCoal,
            'to collective':np.sum(vals)}})
    return coalValue



def OptimalCoalition(inputs):
    
    numCoalitions = len(inputs.keys())
    coalitionNames = list(inputs.keys())
    members = [np.matrix(coalitionNames[i]) for i in list(range(0,numCoalitions))]
    numMembers = [np.size(members[a]) for a in list(range(0,numCoalitions))]
    totMembers = np.max(numMembers)
    
    coalitionChoice = [[0 for i in list(range(numCoalitions))] for j in list(range(numCoalitions))]
    savevals = [[0 for i in list(range(totMembers))] for j in list(range(numCoalitions))]
    savevals2 = [[0 for i in list(range(totMembers))] for j in list(range(numCoalitions))]
    
    c=0
    for c in list(range(0,numCoalitions)):
        # pick a seed coalition
        seed = inputs[coalitionNames[c]]
        currentMembers = [members[c][0,i] for i in list(range(0,numMembers[c]))]
        currentValue = [0 for i in list(range(0,totMembers))]
        aux = 0
        for i in list(range(0,totMembers)):
            currentValue[i] = ValueForOpposition(inputs,i,coalitionNames[c],coalitionNames[-1])
            
            
        bestChoice = copy.deepcopy(c)
        savevals[c] = copy.deepcopy(currentValue)
        # compare with other coalitions
        bestImprovement = -1
        for n in list(range(0,numCoalitions)):
            potentialMembers = [members[n][0,i] for i in list(range(0,numMembers[n]))]
            potentialValue = [0 for i in list(range(0,totMembers))]
            aux = 0
            for i in potentialMembers:
                potentialValue[i] = ValueForOpposition(inputs,i,coalitionNames[n],coalitionNames[-1])
            
            # check if new value is better for all members than in previous coalition
            aux = 1
            for i in potentialMembers:
                if potentialValue[i]<=currentValue[i]:
                    aux = 0
            newImprovement = 0
            if not aux==0:
                for i in potentialMembers:
                    newImprovement+=potentialValue[i]-currentValue[i]
                    
            # updating the values 
            if newImprovement>=bestImprovement and newImprovement>0:
                bestImprovement = copy.deepcopy(newImprovement)
                bestChoice = copy.deepcopy(n)
                savevals2[c] = copy.deepcopy(potentialValue)
                coalitionChoice[c][bestChoice] = bestImprovement
        
        
    summary = {coalitionNames[i]: {'change':coalitionChoice[i], 
           'self': savevals[i], 
           'best': savevals2[i]}for i in list(range(0,numCoalitions))}
    TPM = [summary[i]['change'] for i in coalitionNames]
    aux = copy.deepcopy(TPM)
    for i in list(range(0,numCoalitions)):
        for j in list(range(numCoalitions)):
            if not TPM[i][j] == 0:
                aux[i][j] = TPM[i][j]/np.sum(TPM[i])
    TPM = aux    
    return summary, TPM

def WinningCoalitions(TPM):
    numCoal = np.size(TPM,0)
    source = np.sum(TPM,0)
    sink = np.sum(TPM,1)
    winning = []
    for i in list(range(0,numCoal)):
        winning.append(0)
        if sink[i]==0 and source[i]>0:
            winning[i]=1
            
    return winning
    

def IsWinningMinimal(coalitionNames,winning):
    numCoalitions = len(winning)
    members = [np.matrix(coalitionNames[i]) for i in list(range(0,numCoalitions))]
    numMembers = [np.size(members[a]) for a in list(range(0,numCoalitions))]
    
    isMinimal = 1
    for i in list(range(0,numCoalitions)):
        if winning[i]==1:            
            for j in list(range(0,numCoalitions)):
                if set(coalitionNames[i]) > set(coalitionNames[j]):
                    isMinimal = 0
                                        
    return isMinimal
    
    
def DrawCoalitionMap(inputs,summary,TPM):
    numCoal = np.size(TPM,0)
    listCoals = list(range(0,numCoal)) 
    names = list(inputs.keys())
    
    Conns = [(names[i],names[j],TPM[i][j]) for i in listCoals for j in listCoals if TPM[i][j]>0]
    nodeSizes = [np.sum(summary[coal]['self']) for coal in names]
    nS = 1000*(0.1+(nodeSizes-np.min(nodeSizes))/(np.max(nodeSizes)-np.min(nodeSizes)))
    winner = WinningCoalitions(TPM)
    
    lineWidth = [(summary[coal]['change']) for coal in names]
    G = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_weighted_edges_from(Conns)
    weights = [3*G[u][v]['weight'] for u,v in G.edges()]
    fig = plt.figure()
    nx.draw(G,pos=nx.kamada_kawai_layout(G),
            with_labels=True, node_size=nS, width=weights,
            arrowsize=20, node_color=winner, alpha=0.4,
            font_size=22)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    return fig,nodeSizes,nS


    
def DrawOnlyStrongest(inputs,summary,TPM):
    numCoal = np.size(TPM,0)
    listCoals = list(range(0,numCoal)) 
    names = list(inputs.keys())
    
    Conns = [(names[i],names[j],TPM[i][j]) for i in listCoals for j in listCoals if TPM[i][j]==np.max(TPM[i]) and TPM[i][j]>0]
    nodeSizes = [np.sum(summary[coal]['self']) for coal in names]
    nS = 5000*(0.1+(nodeSizes-np.min(nodeSizes))/(np.max(nodeSizes)-np.min(nodeSizes)))
    winner = WinningCoalitions(TPM)
    
    lineWidth = [(summary[coal]['change']) for coal in names]
    G = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_weighted_edges_from(Conns)
    weights = [3*G[u][v]['weight'] for u,v in G.edges()]
    fig = plt.figure()
    nx.draw(G,pos=nx.spring_layout(G),
            with_labels=True, node_size=nS, width=weights,
            arrowsize=20, node_color=winner, alpha=0.4,
            font_size=22)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    return fig,nodeSizes,nS

'''
defined analysis pipelines
'''

def Trad12ExampleAnalysis():
    # calculating example from original Trad12 script
    outputs = standard_variables_trad12()
    
    outputs.update(solver_rwv(outputs))
    
    outputs.update(ConstitutionalControl(outputs))
    
    outputs.update(DerivedInterest(outputs))
    
    outputs.update(Control_ActorActor(outputs))
    
    outputs.update(Control_ActorEvent(outputs))
        
    outputs.update(Control_EventEvent(outputs))
    
    outputs.update(PositiveOutcome(outputs))
    
    outputs.update(Increment_ExpectedRealization(outputs))
    
    outputs.update(ExpectedValueOfCollectivity(outputs))
    
    outputs.update(ExpectedWeightedRealization(outputs))
    
    outputs.update(DirectedPowerOfCollectivity(outputs))
    
    outputs.update(TotalPowerOfCollectivity(outputs))
    
    return outputs;



def TechnoExampleAnalysis():
    # calculating example from original Techno script
    outputs = standard_variables_techno()
    
    outputs.update(solver_rwv(outputs))
    
    outputs.update(FractionOfResources(outputs))
     
    outputs.update(Control_ActorEvent(outputs))
    
    outputs.update(DerivedInterest(outputs))
    
    outputs.update(Control_ActorResource(outputs))
    
    outputs.update(Control_ActorActor(outputs))
    
    return outputs;



def RunFullAnalysis(outputs):
    
    outputs.update(CheckParameters(outputs))
    
    outputs.update(solver_rwv(outputs))
    
    outputs.update(FractionOfResources(outputs))
    
    outputs.update(Control_ActorEvent(outputs))
    
    outputs.update(DerivedInterest(outputs))
    
    outputs.update(Control_ActorResource(outputs))
    
    outputs.update(Control_EventEvent(outputs))
     
    outputs.update(Control_ActorActor(outputs))
    
    outputs.update(ConstitutionalControl(outputs))
    
    outputs.update(PositiveOutcome(outputs))
    
    outputs.update(Increment_ExpectedRealization(outputs))
    
    outputs.update(ExpectedValueOfCollectivity(outputs))
    
    outputs.update(ExpectedWeightedRealization(outputs))
    
    outputs.update(DirectedPowerOfCollectivity(outputs))
    
    outputs.update(TotalPowerOfCollectivity(outputs))
    
    return outputs;


'''
POST-PROCESSING FUNCTIONS
'''

def CalculateMetrics(origOutputs,outputs,metrics):
    
    if len(metrics)==0:
        metrics = {'control':[],
                   }
        
    # calculating difference between final directed control and interest
    control = np.matrix.tolist(np.mean(outputs['c_AE'][0]-origOutputs['y'][0],1))
    control1D = [control[i][0] for i in list(range(0,len(control)))]
    metrics['control'].append(control1D)

    return metrics;


''' 
PLOTTING FUNCTIONS
'''

def imagesc_varyInterests(data,titles,mask):
    fig = plt.figure(figsize=(20,20))
    cols = 4
    rows = 3
    
    iterator = list(range(0,len(data[0])))
    
    for i in list(range(0,len(data))):
        plt.subplot(cols,rows,i+1)
        plotdata = [[data[i][a][b]*mask[a][b] for a in iterator] for b in iterator]
        im = plt.imshow(plotdata,extent=[-1,1,-1,1],vmin=-1,vmax=1)
        plt.title(titles[i])
        plt.set_cmap('seismic')

    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

'''
OUTPUT FUNCTIONS
'''

def Outputvariables(heading,data,f,newlines):
    
    if np.size(data)>1:
        
        if type(data)==np.ndarray:
            data = np.matrix(data)
        
        print(heading,file=f)
        for line in data:
            np.savetxt(f, line, fmt='%.3f')
            
    else:
        dat = "{:.3f}".format(data)
        print(heading+ ': ' + dat,file=f)
    
    for i in list(range(0,newlines)):
        print('\n',file=f)
        
        
def TechnoOutput_file(filepath,title,inputs):
    with open(filepath+ r'\\' + time.strftime("%Y%m%d") + '_' +title+ '.txt','w+') as f:
        
        print('********************************************\n',file=f)
        print('  ' + title +  '\n',file=f)
        print('  Created on: '+ time.strftime("%d/%m/%Y") + ' ' + time.strftime("%H:%M:%S"),file=f)
        print('********************************************\n\n\n',file=f)
        
        for i in inputs:
            Outputvariables(inputs[i][1],inputs[i][0],f,1)
      
        print('********************************************\n',file=f)
        print('  ' + title +  '\n',file=f)
        print('  Created on: '+ time.strftime("%d/%m/%Y") + ' ' + time.strftime("%H:%M:%S"),file=f)
        print('********************************************\n\n\n',file=f)
    return;    
    
        
        
def OrderedOutput_file(filepath,title,inputs,order):
    
    with open(filepath+ r'\\' + time.strftime("%Y%m%d") + '_' +title+ '.txt','w+') as f:
        
        print('********************************************\n',file=f)
        print('  ' + title +  '\n',file=f)
        print('  Created on: '+ time.strftime("%d/%m/%Y") + ' ' + time.strftime("%H:%M:%S"),file=f)
        print('********************************************\n\n\n',file=f)
        
        for i in order:
            Outputvariables(inputs[i][1],inputs[i][0],f,1)
      
        print('********************************************\n',file=f)
        print('  ' + title +  '\n',file=f)
        print('  Created on: '+ time.strftime("%d/%m/%Y") + ' ' + time.strftime("%H:%M:%S"),file=f)
        print('********************************************\n\n\n',file=f)
    return;    
    

def Outputvariables_screen(heading,data,newlines):
    
    if np.size(data)>1:
        if type(data)==np.ndarray:
            data = np.matrix(data)
        print(heading)
        for line in data:
            dat = "{:.3f}".format(line)
            print(dat)
            
    else:
        dat = "{:.3f}".format(data)
        print(heading+ ': ' + dat)
    for i in list(range(0,newlines)):
        print('\n')
        
        
def OrderedOutput_screen(inputs,order):
    
    for i in order:
        Outputvariables_screen(inputs[i][1],inputs[i][0],1)
  
    return;    
    


'''
TODO LIST!
1. consider merging the two, is techno the only necessary one?
2. use independent function script
3. write clear descriptions of what input and output variables mean
4. make goodd looking output
5. make input dialog


'''
