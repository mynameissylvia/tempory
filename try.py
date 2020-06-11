#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:36:37 2019

@author: sylvia
"""
import numpy as np
from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
from gurobipy  import *
import random
import time
random.seed(250)

start = time.time()

storing = []
files=open('c101.txt')
for f in files:
    a=f.split()
    storing.append(a)
vehicle = [1,2]#range(int(storing[4][0])+1)[1:]
vehicle_sure = range(len(vehicle)+1)
vehicle_capacity = int(storing[4][1])
data = [[int(j) for j in i] for i in storing[9:]]
data[0][0]=0
# 0.'CUST NO.', 1.'XCOORD.', 2.'YCOORD.', 3.'DEMAND', 4.' READY TIME', 5.'DUE DATE', 6.'SERVICE TIME'
Tmax = data[0][5]
ALPHA=0
BETA=0  
n=30
location = [data[:][0]]+data[:][1:n+1]
del data
locate = [i+1 for i in range(n)]
prob = [min(random.random()+0.5,0.99) for i in locate]
chosen_n=30
#40 48
scenario = []


all_d = location[:]
all_d.append(location[0][:])
all_d[-1][0]=len(all_d)-1
timeing=[]
max_time=0
for i in all_d:
    t=[]
    for j in all_d:
        time_ij = ((i[1]-j[1])**2 + (i[2]-j[2])**2)**0.5
        t.append(time_ij)
        if time_ij>max_time:
            max_time=time_ij
        del time_ij
    timeing.append(t)
    del t
large_M=int(all_d[0][5]+max_time+all_d[1][6])+1
scine=range(5)


#maximum probability
max_prob_m = []
max_prob_n = []
max_prob = []
max_prob_c = 1
for i in locate:
    if prob[i-1] >= 0.5:
        max_prob_m.append('y')
        max_prob_n.append(i)
        max_prob.append(prob[i-1])
        max_prob_c*= prob[i-1]
    else:
        max_prob_m.append('n')
        max_prob.append(1-prob[i-1])
        max_prob_c*= (1-prob[i-1])
scenario.append([max_prob_n,max_prob_c])
customer_set=[max_prob_n]
customer_p = [max_prob_c]
current_n = 1
chosen=[]
chosen_s = []
limit = 0.1
i=0
sc = scenario[:]
# the limitation for the given scerario:maximum probability~25 percentage probability
#while s>=30:
 #   i=0
while i <max(100000,chosen_n):
    m = random.choice(locate)
    a = random.sample(locate,m)
    a.sort()
    p = 1
    for i1 in locate:
        if i1 in a:
            p*=prob[i1-1]
        else:
            p*=(1-prob[i1-1])
    if len(sc)<chosen_n:
        sc.append([a,p])
        customer_set.append(a)
        customer_p.append(p)
    else:
        if p > min(customer_p) and a not in customer_set:
            small = customer_p.index(min(customer_p))
            sc.pop(small)
            customer_set.pop(small)
            customer_p.pop(small)
            sc.append([a,p])
            customer_set.append(a)
            customer_p.append(p)
    i+=1
scenario = sc
print time.time()-start
"""
S = 30000000
start_time = time.time()
all_s = {}
all_s_t = {}
for i in range(S):
    m = random.choice(locate)
    a = random.sample(locate,m)
    a.sort()
    p = 1
    for i in locate:
        if i in a:
            p*=prob[i-1]
        else:
            p*=(1-prob[i-1])
    a_s = ' '.join(str(i) for i in a)
    all_s_t[a_s] = p
current_t1 = time.time()
print len(all_s_t),current_t1-start_time
f = int(len(all_s_t)/chosen_n)
import operator
sorted_s_t = sorted(all_s_t.items(), key = operator.itemgetter(1))
current_t2 = time.time()
print (current_t2 - current_t1)
chosen = []
for i in range(chosen_n):
    chosen.append(sorted_s_t[-1-i*f])
del sorted_s_t
del all_s_t
print (time.time()-current_t2)
print (time.time()-start)
"""
S=[]
P=[]
for i in scenario:
    S.append(i[0])
    P.append(i[1])

c1=0
c2=2.5
c3=10000
service=[0] + [10 for i in range(n)]+[0]
a=[10 for i in range(n)]
time_penalty = 200

Pmax=100


x={}
y={}
t={}
p={}
Q={}
w={}
d={}
delay_check={}
delay_t={}
node_s = range(n+1)
customer = [i+1 for i in range(n)]

model = Model('Stochastic')
#        model.setParam('OutputFlag',False)
model.Params.MIPGap=0.05
#model.Params.timeLimit=36000.0
w = model.addVars(node_s,vtype=GRB.BINARY,name = "w"  )

for s in range(chosen_n):

    node_s = range(n+1)
    customer_c = S[s]
    x[s] = model.addVars(node_s,node_s,vehicle_sure,vtype=GRB.BINARY,name = "x_%d" % s) 
    y[s] = model.addVars(node_s,vehicle_sure,vtype=GRB.BINARY,name = "y_%d" % s)
    t[s] = model.addVars(node_s,vehicle_sure,name = "t_%d" % s)
    d[s] = model.addVars(node_s,node_s,vehicle_sure,name = "d_%d" % s)
    p[s] = model.addVars(node_s,vehicle_sure,name = "penalty_%d" % s)
    delay_check[s] = model.addVars(node_s,vehicle_sure,vtype=GRB.BINARY,name = "delaycheck_%d" % s)
    delay_t[s] = model.addVars(node_s,vehicle_sure,name = "delaytime_%d" % s)
    model.addConstrs((quicksum(x[s][i,j,k] for j in node_s if i != j)==y[s][i,k] for i in node_s for k in vehicle_sure))
    model.addConstrs((quicksum(x[s][j,i,k] for j in node_s if i != j)==y[s][i,k] for i in node_s for k in vehicle_sure))
    model.addConstr(quicksum(y[s][0,k] for k in  vehicle_sure)<=len(vehicle_sure))
    model.addConstrs(quicksum(y[s][i,k] for k in  vehicle_sure)<=1 for i in customer)
    model.addConstrs(quicksum(y[s][i,k] for k in  vehicle_sure)>=w[i] for i in customer)
    model.addConstrs(d[s][i,j,k]<=vehicle_capacity*x[s][i,j,k] for i in node_s for j in node_s if j != i for k in vehicle)
    model.addConstrs(quicksum(d[s][i,j,k] for j in node_s if i != j)-quicksum(d[s][j,i,k] for j in node_s if i != j)==all_d[i][3]*y[s][i,k] for i in customer for k in vehicle_sure)
    model.addConstrs(d[s][i,j,k]>=0 for i in node_s for j in node_s if i != j for k in vehicle_sure)
    model.addConstrs(quicksum(service[i]*y[s][i,k] for i in customer) +quicksum(quicksum(timeing[i][j]*x[s][i,j,k] for j in node_s if j != i) for i in node_s)<= Tmax for k in vehicle)
    model.addConstrs(t[s][i,k]+service[i]+timeing[i][j]+large_M*x[s][i,j,k]<=t[s][j,k]+large_M for i in node_s for j in customer for k in vehicle)
    model.addConstrs(all_d[i][4]<= t[s][i,k] <= all_d[i][5]+Pmax for i in node_s for k in vehicle)
    model.addConstrs(quicksum(x[s][i,j,k] for k in vehicle_sure)== 0 for i in node_s for j in node_s if i == j)
    model.addConstrs(t[s][i,k]-all_d[i][5] == delay_check[s][i,k] for i in customer for k in vehicle)
    model.addConstrs(p[s][i,k] == delay_check[s][i,k] for i in customer for k in vehicle if delay_check[s][i,k] > 0 )
    model.addConstrs(p[s][i,k]>=0 for i in node_s for k in vehicle_sure)
    Q[s] = (quicksum(5*all_d[i][3]*y[s][i,k] for i in customer_c for k in vehicle if customer_c != None)
    -c1*quicksum(x[s][i,j,k]for i in node_s for j in node_s if i != j for k in vehicle)
    -c2*quicksum(timeing[i][j]*quicksum(x[s][i,j,k] for k in vehicle)for i in node_s for j in node_s if i != j)
    -c3*quicksum(x[s][i,j,0]for i in node_s for j in node_s if i != j )
    -time_penalty*quicksum(p[s][i,k] for i in customer for k in vehicle)
    +quicksum(w[i]*a[i] for i in range(n)))
model.setObjective(quicksum(Q[s]*P[s] for s in range(chosen_n))/sum(P), GRB.MAXIMIZE)
model.optimize()
count_chossen = 0
for v in model.getVars():
    if v.Varname[0] == 'w' and v.x>0:
        count_chossen +=1
        print ("%s" % (v.Varname[2:-1])),
if count_chossen == 0:
    print('No chosen customers'),
print "\n",time.time()-start

    