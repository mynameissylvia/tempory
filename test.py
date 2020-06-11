# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:41:58 2019

@author: Angel
"""
"""
from gurobipy  import *
import numpy as np 
import math 
import random
import matplotlib.pyplot as plt

#random.seed(50)
#50
#150 10 10

#initial data

node_c = 20
center = 5
vehicle_c = 4
size = 20

customer = [ i+1 for i in range(node_c-1)]
node = range(node_c)
location = [[0,0]]
location_c= []
tw_e = []
t_w = [[0,10000]]
for i in customer:
    if i<=center:
        l_x = random.uniform(-30,30)
        l_y = random.uniform(-30,30)
        t_w_e = random.uniform(0,20)
        t_w_l =  random.uniform(10,40)
        location_c.append([l_x,l_y])
        tw_e.append([t_w_e,t_w_l])
    else:
        cent = random.choice(location_c)
        cent_t = tw_e[location_c.index(cent)]
        l_x = random.uniform(-3,3)+cent[0]
        l_y = random.uniform(-3,3)+cent[1]
        t_w_e = random.uniform(-10,20)+cent_t[0]
        t_w_l =  random.uniform(-10,40)+cent_t[1]
    location.append([l_x,l_y])
    t_w.append([t_w_e,t_w_l])
del location_c
del tw_e
dist = []
time = [] 
for i in node:
    d0 = []
    t0 = []
    for j in node:
        distance = math.sqrt(math.pow(location[i][0]-location[j][0],2)+math.pow(location[i][1]-location[j][1],2))
        d0.append(distance)
        t0.append(distance/10)
        del distance
    dist.append(d0)
    time.append(t0)
    del d0
    del t0
pickup = [random.randint(0,10) for i in customer]
delivery = [random.randint(0,99) for i in customer]
revenue = [(pickup[i-1]+delivery[i-1])/20*random.choice([50,100,200]) for i in customer]
probability = [random.uniform(0,1) for i in customer]
location = np.matrix(location)
l = plt.plot(location[:,0], location[:,1], 'ro',1.0)
counting=0
for i in location:
    plt.text(i[0,0], i[0,1],counting)
    counting+=1
plt.show()



vehicle = [i + 1 for i in range(vehicle_c)]
vehicle_sure = range(vehicle_c+1)
vehicle_q = 200

service_time = 5
Tmax = 30

c1 = 5
c2 = 5
c3 = 10000000
a = 100

print revenue

large_M=10000

S = range(1)
NODES=[]



model = Model('Stochastic')
w = model.addVars(customer,vtype=GRB.BINARY,name = "w")
#w is the selected customer must be served no matter there's a needed or not
model.update()

Q=0

x={}
y={}
t={}
p={}
d={}

for s in S:
    a = 5
    node_s = node
    customer_c = random.sample(customer,a)
    customer_c.sort()
    
    x[s] = model.addVars(node_s,node_s,vehicle_sure,vtype=GRB.BINARY,name = "x_%d" % s)
    y[s] = model.addVars(node_s,vehicle_sure,vtype=GRB.BINARY,name = "y_%d" % s)
    t[s] = model.addVars(node_s,vehicle_sure,name = "arrive_time_%d" % s)
    p[s] = model.addVars(node_s,node_s,vehicle_sure,name = "pickupamount_%d" % s)
    d[s] = model.addVars(node_s,node_s,vehicle_sure,name = "deliveryamount_%d" % s)
    
    model.addConstrs((quicksum(x[s][i,j,k] for j in node_s if i != j)==y[s][i,k] for i in node_s for k in vehicle_sure))
    model.addConstrs((quicksum(x[s][j,i,k] for j in node_s if i != j)==y[s][i,k] for i in node_s for k in vehicle_sure))
    model.addConstr(quicksum(y[s][0,k] for k in  vehicle_sure)<=len(vehicle_sure))
    model.addConstrs(quicksum(y[s][i,k] for k in  vehicle_sure)<=1 for i in node_s if i != 0)
    model.addConstrs(quicksum(y[s][i,k] for k in  vehicle_sure)>=w[i] for i in node_s if i != 0)
    model.addConstrs(quicksum(time[i][j]*x[s][i,j,k] for i in node_s for j in node_s if i != j)<= Tmax for k in vehicle)
    model.addConstrs(t[s][i,k]-t[s][j,k]+service_time+time[i][j]-large_M*x[s][i,j,k]<=large_M for i in node_s for j in node_s if i != j for k in vehicle)
    model.addConstrs(t_w[i][0]<= t[s][i,k] <= t_w[i][1]for i in node_s for k in vehicle)
#    model.addConstrs(p[s][i,j,k]+d[s][i,j,k]<=vehicle_q*x[s][i,j,k] for i in node_s for j in node_s if i != j for k in vehicle)
#    model.addConstrs(quicksum(p[s][i,j,k] for j in node_s if i != j)-quicksum(p[s][j,i,k] for j in node_s if i != j)==pickup[i-1]*y[s][i,k] for i in node_s if i != 0 for k in vehicle_sure)
#    model.addConstrs(quicksum(d[s][j,i,k] for j in node_s if i != j)-quicksum(d[s][i,j,k] for j in node_s if i != j)==delivery[i-1]*y[s][i,k] for i in node_s if i != 0 for k in vehicle_sure)
    model.addConstrs(t[s][i,k]>=0 for i in node_s for k in vehicle_sure)
#    model.addConstrs(p[s][i,j,k]>=0 for i in node_s for j in node_s if i != j for k in vehicle_sure)
#    model.addConstrs(d[s][i,j,k]>=0 for i in node_s for j in node_s if i != j for k in vehicle_sure)
    q = quicksum(revenue[i-1]*quicksum(y[s][i,k] for k in vehicle_sure) for i in customer_c)-c1*quicksum(x[s][i,j,k]for i in node_s for j in node_s if i != j for k in vehicle)-c2*quicksum(dist[i][j]*quicksum(x[s][i,j,k] for k in vehicle)for i in node_s for j in node_s if i != j )-c3*quicksum(x[s][i,j,0]for i in node_s for j in node_s if i != j )
    Q +=q/len(S)
    NODES.append(customer_c)
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
storing = []
files=open('c101.txt')
for f in files:
    a=f.split()
    storing.append(a)
vehicle = range(int(storing[4][0])+1)[1:]
vehicle_sure = range(len(vehicle)+1)
vehicle_capacity = int(storing[4][1])
data = [[int(j) for j in i] for i in storing[9:]]
data[0][0]=0
# 0.'CUST NO.', 1.'XCOORD.', 2.'YCOORD.', 3.'DEMAND', 4.' READY TIME', 5.'DUE DATE', 6.'SERVICE TIME'
Tmax = data[0][5]
ALPHA=0
BETA=0  
n=25
location = [data[:][0]]+data[:][1:n+1]
location = location[:]
l = 0
occurance = [float("%.2f" %random.random()) for i in range(n)]
#for i in range
for i in location:
    i[0]=l
    l+=1
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
#S=[(i+1)*2 for i in scine]
S_t=range(5)
S=[10 for i in S_t]
#S = [random.randint(1,n-1) for i in scine]
customer_s = [random.sample(range(n+1)[1:],i) for i in S]
print customer_s
c1=1
c2=0.01
c3=10000
service=[0] + [10 for i in range(n)]+[0]
a=[5 for i in range(n)]
time_penalty = 200

Pmax=100

S_t=[1]
x={}
y={}
t={}
p={}
delay_check={}
early_check={}
delay_t={}
early_t={}
c=[1 for i in range(n)]
global ALPHA
global BETA
for s in S_t:
    start_t = time.time()
    model = Model('Stochastic')
#        model.setParam('OutputFlag',False)
    model.Params.MIPGap=0.05
    model.Params.timeLimit=30.0
    node_s = range(n+1)
    customer = [i+1 for i in range(n)]
    customer_c = customer_s[s]
    x[s] = model.addVars(node_s,node_s,vehicle_sure,vtype=GRB.BINARY,name = "x_%d" % s) 
    y[s] = model.addVars(node_s,vehicle_sure,vtype=GRB.BINARY,name = "y_%d" % s)
    t[s] = model.addVars(node_s,vehicle_sure,name = "t_%d" % s)
    p[s] = model.addVars(node_s,vehicle_sure,name = "t_%d" % s)
    delay_check[s] = model.addVars(node_s,vehicle_sure,vtype=GRB.BINARY,name = "delaycheck_%d" % s)
    delay_t[s] = model.addVars(node_s,vehicle_sure,name = "delaytime_%d" % s)
    model.addConstrs((quicksum(x[s][i,j,k] for j in node_s if i != j)==y[s][i,k] for i in node_s for k in vehicle_sure))
    model.addConstrs((quicksum(x[s][j,i,k] for j in node_s if i != j)==y[s][i,k] for i in node_s for k in vehicle_sure))
    model.addConstr(quicksum(y[s][0,k] for k in  vehicle_sure)<=len(vehicle_sure))
    model.addConstrs(quicksum(y[s][i,k] for k in  vehicle_sure)<=1 for i in customer)
    model.addConstrs(quicksum(y[s][i,k] for k in  vehicle_sure)>=c[i-1] for i in customer)
#        model.addConstrs(y[s][n+1,k]- y[s][0,k] ==0 for k in vehicle_sure)
    model.addConstrs(quicksum(service[i]*y[s][i,k] for i in customer) +quicksum(quicksum(timeing[i][j]*x[s][i,j,k] for j in node_s if j != i) for i in node_s)<= Tmax for k in vehicle)
    model.addConstrs(t[s][i,k]+service[i]+timeing[i][j]+large_M*x[s][i,j,k]<=t[s][j,k]+large_M for i in node_s for j in customer for k in vehicle)
    model.addConstrs(all_d[i][4]<= t[s][i,k] <= all_d[i][5]+Pmax for i in node_s for k in vehicle)
    model.addConstrs(quicksum(x[s][i,j,k] for k in vehicle_sure)== 0 for i in node_s for j in node_s if i == j)
    model.addConstrs(t[s][i,k]-all_d[i][5] == delay_check[s][i,k] for i in customer for k in vehicle)
    model.addConstrs(p[s][i,k] == delay_check[s][i,k] for i in customer for k in vehicle if delay_check[s][i,k] > 0 )
    model.addConstrs(p[s][i,k]>=0 for i in node_s for k in vehicle_sure)
    Q = (quicksum(5*all_d[i][3]*y[s][i,k] for i in customer_c for k in vehicle if customer_c != None)
    -c1*quicksum(x[s][i,j,k]for i in node_s for j in node_s if i != j for k in vehicle)
    -c2*quicksum(timeing[i][j]*quicksum(x[s][i,j,k] for k in vehicle)for i in node_s for j in node_s if i != j)
    -c3*quicksum(x[s][i,j,0]for i in node_s for j in node_s if i != j )
    -time_penalty*quicksum(p[s][i,k] for i in customer for k in vehicle))
    model.setObjective(Q/len(S), GRB.MAXIMIZE)
    model.optimize()
    print '\nFirst stage choosen customers:'
    count_chossen = 0
    for v in model.getVars():
        if v.Varname[0] == 'w' and v.x>0:
            count_chossen +=1
            print ("%s" % (v.Varname[2:-1])),
    if count_chossen == 0:
        print('No chosen customers'),
    for s in S:
        for v in model.getVars():
            if v.Varname[0] == 'P' and v.x>0:
                print("%s" % (v.Varname))
