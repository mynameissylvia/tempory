# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:41:58 2019

@author: Angel
"""

from gurobipy  import *
import numpy as np 
import math 
import random
import matplotlib.pyplot as plt



#initial data

node_c = 13
vehicle_c = 4
size = 20

customer = [ i+1 for i in range(node_c-1)]
node = range(node_c)
location = [[0,0]]
t_w = [[0,10000]]
for i in customer:
    l_x = random.uniform(-20,20)
    l_y = random.uniform(-20,20)
    t_w_e = random.uniform(0,20)
    t_w_l =  random.uniform(10,40)
    location.append([l_x,l_y])
    t_w.append([t_w_e,t_w_l])
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
revenue = [(pickup[i-1]+delivery[i-1])/20*random.choice([95,130,220,140,310]) for i in customer]
probability = [random.uniform(0,1) for i in customer]
location = np.matrix(location)
l = plt.plot(location[:,0], location[:,1], 'ro')
plt.show()



vehicle = [i + 1 for i in range(vehicle_c)]
vehicle_sure = range(vehicle_c+1)
vehicle_q = 200

service_time = 5
Tmax = 15

c1 = 10
c2 = 20
c3 = 10000000
a = 10000

large_M=10000

S = range(10)
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
    a = 10
    node_s = node
    customer_c = random.sample(customer,a)
    
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
    model.addConstrs(p[s][i,j,k]+d[s][i,j,k]<=vehicle_q*x[s][i,j,k] for i in node_s for j in node_s if i != j for k in vehicle)
    model.addConstrs(quicksum(p[s][i,j,k] for j in node_s if i != j)-quicksum(p[s][j,i,k] for j in node_s if i != j)==pickup[i-1]*y[s][i,k] for i in node_s if i != 0 for k in vehicle_sure)
    model.addConstrs(quicksum(d[s][j,i,k] for j in node_s if i != j)-quicksum(d[s][i,j,k] for j in node_s if i != j)==delivery[i-1]*y[s][i,k] for i in node_s if i != 0 for k in vehicle_sure)
    model.addConstrs(t[s][i,k]>=0 for i in node_s for k in vehicle_sure)
    model.addConstrs(p[s][i,j,k]>=0 for i in node_s for j in node_s if i != j for k in vehicle_sure)
    model.addConstrs(d[s][i,j,k]>=0 for i in node_s for j in node_s if i != j for k in vehicle_sure)
    q = quicksum(revenue[i-1]*quicksum(y[s][i,k] for k in vehicle_sure) for i in customer_c if i != 0)-c1*quicksum(x[s][i,j,k]for i in node_s for j in node_s if i != j for k in vehicle)-c2*quicksum(dist[i][j]*quicksum(x[s][i,j,k] for k in vehicle)for i in node_s for j in node_s if i != j )-c3*quicksum(x[s][i,j,0]for i in node_s for j in node_s if i != j )
    Q +=q/len(S)
    NODES.append(customer_c)

model.setObjective(quicksum(w[i] for i in customer)*a+Q, GRB.MAXIMIZE)
model.optimize()

print '\nFirst stage choosen customers:'
count_chossen = 0
for v in model.getVars():
    if v.Varname[0] == 'w' and v.x>0:
        count_chossen +=1
        print ("%s" % (v.Varname[2:-1])),
if count_chossen == 0:
    print('No chosen customers')

#
#for s in S:
 #   print'\nsceario %d\nnode:'%s,NODES[s]
  #  for v in model.getVars():
   #     if v.Varname[0] == 'x' and v.Varname[2] == '%d' %s and v.x>0:
    #        print("%s" % (v.Varname))