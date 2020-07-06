
"""
Created on Mon Jul  6 20:35:53 2020

@author: Angel
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:52:18 2019
Simple prior model
@author: sylvia
"""
import numpy as np
import random
import time
import copy
random.seed(250)

start = time.time()
#
text_name = []
for i in range(9):
    text="c10"+str(i+1)+".txt"
    text_name.append(text)
for i in range(8):
	text ="c20"+str(i+1)+".txt"
	text_name.append(text)
for i in range(12):
	text ="r1"+str((i+1)//10)+str((i+1)%10)+".txt"
	text_name.append(text)
#
numb1=1
numb2=1
numb3=10
numb4=2.5
numb5=20

n=20
locate = [i+1 for i in range(n)]
scenario_file="20customer_sc.txt"
s_files=open(scenario_file)
count = 0
scenario=[]
for f in s_files:
    if count%2==0:
        sa=f.split()
        map_object=map(int, sa)
        sa=list(map_object)
    else:
        sb = float(f)
        scenario.append([sa,sb])
    count +=1
S=[]
P=[]
for i in scenario:
    S.append([1]+[1 if j in i[0] else 0 for j in locate]+[1])    
    P.append(i[1])
#text_name=["c103.txt"]
for text_t in text_name:
    storing = []
    files=open(text_t)
    for f in files:
        a=f.split()
        storing.append(a)
    vehicle = 2#range(int(storing[4][0])+1)[1:]
    vehicle_sure = range(vehicle+1)
    vehicle_capacity = int(storing[4][1])
    data = [[int(j) for j in i] for i in storing[9:]]
    data[0][0]=0
    # 0.'CUST NO.', 1.'XCOORD.', 2.'YCOORD.', 3.'DEMAND', 4.' READY TIME', 5.'DUE DATE', 6.'SERVICE TIME'
    Tmax = data[0][5]
    original=0
    depote = n+1
    depote_insert = data[:][0]
    depote_insert=depote_insert[:]
    depote_insert[0]=31
    location = [data[:][0]]+data[:][1:n+1]+[depote_insert]
    location = location[:]
    del data
    chosen_n=10    
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
    c1=10
    c2=2.5
    #c3=10000
    service_value=[0] + [10 for i in range(n)]+[0]
    c3 = 20
    demand=[]
    tw=[]
    service = []
    for i in location:
        demand.append(i[3])
        tw.append(i[4:6])
        service.append(i[6])
    allow_time=100
    profit=[5*i for i in demand]
    punish_v={}
    total_score = 0
    location_items=range(len(location))
    punish_v [original]=0
    punish_v[depote]=0
        
    def initial(v_capacity,location,profit,demand,service_value,tw,time_r,service,v_count,c1,c2,c3,allow_time,original,depote):
        Tmax=tw[0][1]
        routing = []
        score_ = 0
        wait_={}
        global maxshift_c, service_start_c
        maxshift_c={}
        maxshift_c[original]=[0]*v_count
        maxshift_c[depote]=[0]*v_count
        wait_[original]=0
        wait_[depote]=0
        service_start_c = {}
        service_start_c[original]=[0]*v_count
        service_start_c[depote]=[0]*v_count
        all_place = location[:]
        all_place.remove(original)
        all_place.remove(depote)
        fix=[]
        while len(routing)<v_count and len(all_place)>1:
            route = [original]
            continue_r = True
            current_capacity = 0
            current_time = 0
            while continue_r:
                insert_l=depote
                insert_score = 0
                insert_punish=0
                for i in all_place:
                    arrive =current_time+time_r[route[-1]][i]
                    start_time = max(arrive,tw[i][0])
                    punish = max(0,start_time-tw[i][1])
                    if i != insert_l and  current_capacity+demand[i]<=v_capacity and start_time<=tw[i][1]+allow_time and start_time+service[i]+time_r[i][0]<=Tmax:
                        current_score =numb1*profit[i]+numb2*service_value[i]-numb3-numb4*(time_r[route[-1]][i])-numb5*punish
                        if current_score>insert_score:
                            insert_score = current_score
                            insert_start = start_time
                            insert_l = i
                            insert_punish=punish
                if insert_l == depote:
                    score_-=c2*(time_r[route[-1]][insert_l])
                    service_start_c[depote][len(routing)-1] = current_time+time_r[route[-1]][depote]
                    route.append(depote)
                    continue_r=False
                    break
                route.append(insert_l)
                wait_[insert_l] = max(0,tw[insert_l][0]-arrive)
                current_capacity+=demand[i]
                score_+=profit[insert_l]+service_value[insert_l]-c1-c2*time_r[route[-1]][insert_l]-c3*insert_punish
                service_start_c[insert_l] = insert_start
                current_time = insert_start+service[i]
                all_place.remove(insert_l)
                fix.append(insert_l)
                punish_v[insert_l]=insert_punish
            currentr=len(routing)-1
            for max_i in range(len(route)):
                i = (max_i)*-1-1    #from back
                loc_i = route[i]    #location of the route at i
                if loc_i == depote:
                    loc_j = route[i-1]
                    if loc_j != original:
                        maxshift_c[depote][currentr]=tw[loc_i][1]-service_start_c[loc_j]-service[loc_j]-time_r[loc_i][loc_j]
                    else:
                        maxshift_c[depote][currentr]=tw[loc_i][1]-service_start_c[loc_j][currentr]-service[loc_j]-time_r[loc_i][loc_j]
                elif route[i+1]==depote:
                    if loc_i != original:
                        maxshift_c[loc_i]=min(tw[loc_i][1]+allow_time-service_start_c[loc_i],maxshift_c[depote][currentr])
                    else:
                        maxshift_c[loc_i]=min(tw[loc_i][1]+allow_time-service_start_c[loc_i][currentr],maxshift_c[depote][currentr])
                elif loc_i==original:
                    maxshift_c[original][currentr]=wait_[route[i+1]]+maxshift_c[route[i+1]]
                else:
                    maxshift_c[loc_i] = min(tw[loc_i][1]+allow_time-service_start_c[loc_i],wait_[route[i+1]]+maxshift_c[route[i+1]])
            routing.append(route)
        return routing,score_,wait_, maxshift_c,service_start_c,all_place,fix
    start_time = time.time()
    initial_d= initial(vehicle_capacity,location_items,profit,demand,service_value,tw,timeing,service,vehicle,c1,c2,c3,allow_time,original,depote)
    for s in range(chosen_n):
        punish_v_c=copy.deepcopy(punish_v)
        route_s=copy.deepcopy(initial_d[0])
        score=initial_d[1]
        initial_score = initial_d[1]
        wait=initial_d[2].copy()
        maxshift=initial_d[3].copy()
        service_start=initial_d[4].copy()
        fix = initial_d[6][:]
        fix.append(original)
        fix.append(depote)
        notused = initial_d[5][:]
        notimprove=0
        service_start[0]=0
        nolocaloptimum = True
        remove_r = 1
        remove_s = 1
        real_exist = S[s]
        for i in route_s:
            current_i=0
            for j in range(len(i)):
                if real_exist[i[j]] <1 :
                    score -=profit[i[j]]
                    k=j
                    new_arrive=service_start[i[k]]+timeing[i[k]][i[k+1]]
                    wait[i[k+1]]=max(0,tw[i[k+1]][0]-new_arrive)
                    shift_n = service_start[i[k+1]]
                    if i[k+1]== depote:
                        service_start[i[k+1]][current_i]=new_arrive
                    else:
                        service_start[i[k+1]]=new_arrive+wait[i[k+1]]
                        new_punish=max(0,service_start[i[k+1]]-tw[i[k+1]][1])
                        score-=c3*(new_punish-punish_v_c[i[k+1]])
                        punish_v_c[i[k+1]]=new_punish
                        k=k+1
                        while i[k+1]<depote and shift_n != service_start[i[k+1]]:
                            new_arrive=service_start[i[k]]+service[i[k]]+timeing[i[k]][i[k+1]]
                            wait[i[k+1]]=max(0,tw[i[k+1]][0]-new_arrive)
                            shift_n = service_start[i[k+1]]
                            service_start[i[k+1]]=new_arrive+wait[i[k+1]]
                            new_punish=max(0,service_start[i[k+1]]-tw[i[k+1]][1])
                            score-=c3*(new_punish-punish_v_c[i[k+1]])
                            punish_v_c[i[k+1]]=new_punish
                            k=k+1
                        service_start[depote][current_i]=service_start[i[k]]+service[i[k]]+timeing[i[k]][i[k+1]]
                        r2 = k-len(i)
                        v=i
                        current_r=current_i
                        for i2 in range(len(v)):
                            place = v[r2]
                            original_msf = maxshift[place]
                            if place == depote:
                                maxshift[place][current_r] = tw[place][1]-service_start[place][current_r]
                            elif place == original:
                                maxshift[place][current_r] = maxshift[v[r2+1]]+wait[v[r2+1]]
                            elif v[r2+1] == depote:
                                maxshift[place] = min(tw[place][1]+allow_time-service_start[place],maxshift[v[r2+1]][current_r]+wait[v[r2+1]])
                            else:
                                maxshift[place]=min(tw[place][1]+allow_time-service_start[place],maxshift[v[r2+1]]+wait[v[r2+1]])
                            if r2<j-len(i) and original_msf== maxshift[place]:
                                break
                            r2-=1
            current_i+=1
        best_score = max(0,score)
        best_route=copy.deepcopy(initial_d[0])
        while notimprove<=150:
            causing_addition_punish = {}
            for i in notused:
                causing_addition_punish[i]=0
            while nolocaloptimum:
                current_r = 100
                current_rlocal=0
                maxratio = 0
                best_insert = 0
                add_score = 0
                for v in route_s:
                    cap = sum([demand[i1] for i1 in v])
                    for i in notused:
                        for i2 in range(len(v)-1):
                            j = v[i2]
                            k = v[i2+1]
                            arrival=service_start[j]+service[j]+timeing[i][j]
                            wait_i = max(0,tw[i][0]-arrival)
                            addition_time = timeing[i][j]+wait_i+service[i]+timeing[i][k]-timeing[j][k]
                            if k != depote:
                                msf = maxshift[k]
                            else:
                                msf = maxshift[k][current_rlocal]
                            if arrival<=tw[i][1]+allow_time and cap+demand[i]<vehicle_capacity and arrival+wait_i+service[i]+timeing[i][k]<=tw[k][1]+allow_time and addition_time<=maxshift[depote][current_rlocal]:
                                start = arrival+wait_i
                                shift = timeing[i][j]+wait_i+service[i]+timeing[i][k]-timeing[j][k]
                                punish=max(start-tw[j][1],0)
                                if shift<=wait[k]+msf:
                                    ratio = (profit[i]*real_exist[i]-c1-c2*(timeing[i][j]+timeing[i][k]-timeing[j][k])-c3*(punish+causing_addition_punish[i]))/shift
                                else:
                                    ratio = -10
                                if ratio > maxratio:
                                    punish_v_c[best_insert]=0
                                    punish_v_c[i]=punish
                                    best_insert = i
                                    insert_place = i2+1
                                    maxratio = ratio
                                    add_score = profit[i]*real_exist[i]-c1-c2*(timeing[i][j]+timeing[i][k]-timeing[j][k])-c3*punish
                                    best_start = start
                                    best_wait = wait_i
                                    best_shift = shift
                                    current_r =current_rlocal
                    current_rlocal +=1
                if maxratio>0:
                    original_wait={}
                    service_start_test={}
                    punish_v_c_test = {}
                    maxshift_test = {}
                    wait_t={}
                    nolocaloptimum=True
                    shift = best_shift
                    for i in v[insert_place:-1]:
                        wait_t[i]=max(0,wait[i]-shift)
                        shift = max(0,shift-wait[i])
                        if shift>0:
                            service_start_test[i]=service_start[i]+shift
                            new_punish=max(0,service_start_test[i]-tw[i][1])
                            add_score-=c3*(new_punish-punish_v_c[i])
                            punish_v_c_test[i]= new_punish
                            maxshift_test[i]=maxshift[i]-shift
                        else:
                            break
                    if add_score <0:
                        causing_addition_punish[best_insert]+=1
                        continue
                    v=route_s[current_r]
                    v.insert(insert_place,best_insert)
                    wait[best_insert]=best_wait
                    service_start[best_insert] = best_start
                    before = v[:insert_place]
                    before.reverse()
                    for i in v[insert_place+1:-1]:
                        original_wait[i]=wait[i]
                        wait[i]=max(0,wait[i]-shift)
                        shift = max(0,shift-original_wait[i])
                        if shift>0:
                            service_start[i]=service_start[i]+shift
                            new_punish=max(0,service_start[i]-tw[i][1])
                            punish_v_c[i]= new_punish
                            maxshift[i]=maxshift[i]-shift
                        else:
                            break
                    if shift>0:
                        service_start[depote][current_r]=service_start[depote][current_r]+shift
                        maxshift[depote][current_r]=maxshift[depote][current_r]-shift
                    be_i=v[insert_place+1]
                    if be_i == depote:
                        maxshift[best_insert]=min(tw[best_insert][1]+allow_time-best_start,wait[be_i]+maxshift[be_i][current_r])
                    else:
                        maxshift[best_insert]=min(tw[best_insert][1]+allow_time-best_start,wait[be_i]+maxshift[be_i])
                    r2 = insert_place-len(v)
                    for i2 in range(len(v)):
                        place = v[r2]
                        original_msf = maxshift[place]
                        if place == original:
                            maxshift[place][current_r] = maxshift[v[r2+1]]+wait[v[r2+1]]
                        elif v[r2+1] == depote:
                            maxshift[place] = min(tw[place][1]+allow_time-service_start[place],maxshift[v[r2+1]][current_r]+wait[v[r2+1]])
                        else:
                            maxshift[place]=min(tw[place][1]+allow_time-service_start[place],maxshift[v[r2+1]]+wait[v[r2+1]])
                        if original_msf== maxshift[place]:
                            break
                        r2-=1
                    score +=add_score
                    notused.remove(best_insert)
                else:
                    nolocaloptimum=False
            if best_score<(score-0.0001) and best_route != route_s:
                best_score= score
                best_route = route_s
                notimprove = 0
                remove_r=1
            else:
                notimprove +=1
            current_r = 0
            for v in route_s:
                original_v= v[:]
                start_d = min(remove_s,len(v)-2-remove_r)
                for i in range(remove_r):
                    if v[start_d] in fix:
                        continue
                    notused.append(v[start_d])
                    maxshift[v[start_d]]=0
                    wait[v[start_d]]=0
                    service_start[v[start_d]]=0
                    score-=profit[v[start_d]]*real_exist[v[start_d]]-c1-c2*(timeing[v[start_d-1]][v[start_d]]+timeing[v[start_d]][v[start_d+1]]-timeing[v[start_d-1]][v[start_d+1]])-c3*punish_v_c[v[start_d]]
                    punish_v_c[v[start_d]]=0
                    del v[start_d]
                if v != original_v:
                    j = v[start_d-1]
                    record = start_d
                    compare = start_d-len(v)
                    for i in v[start_d:]:
                        arrival_start = service_start[j]+service[j]+timeing[i][j]
                        wait[i] = max(0,tw[i][0]-arrival_start)
                        o_service_start = service_start[i]
                        if i != depote:
                            service_start[i]= wait[i]+arrival_start
                            origin_punish=punish_v_c[i]
                            punish_v_c[i]=max(0,service_start[i]-tw[i][1])
                            score-=c3*(punish_v_c[i]-origin_punish)
                            if service_start [i] ==o_service_start:
                                break
                        else:
                            service_start[i][current_r] = wait[i]+arrival_start
                        record+=1
                    r2 = record-len(v)
                    for i2 in range(len(v)):
                        place = v[r2]
                        original_msf = maxshift[place]
                        if place == depote:
                            maxshift[place][current_r] = tw[place][1]-service_start[place][current_r]
                        elif place == original:
                            maxshift[place][current_r] = maxshift[v[r2+1]]+wait[v[r2+1]]
                        elif v[r2+1] == depote:
                            maxshift[place] = min(tw[place][1]+allow_time-service_start[place],maxshift[v[r2+1]][current_r]+wait[v[r2+1]])
                        else:
                            maxshift[place]=min(tw[place][1]+allow_time-service_start[place],maxshift[v[r2+1]]+wait[v[r2+1]])
                        if r2<compare and original_msf== maxshift[place]:
                            break
                        r2-=1
                notused.sort()
                nolocaloptimum=True
                minlen=min(map(len,route_s))-2
                current_r +=1
            remove_s=remove_s+remove_r
            remove_r+=1
            if remove_s > minlen:
                remove_s-=minlen
            if remove_r== n/(3*vehicle):
                remove_r=1
        total_score+=best_score*P[s]
    total_score /=sum(P)
    file1 = open(".txt","a") 
    fix.sort()
    fix.remove(0)
    print text_t," ",len(fix)-1, " ", total_score," ",time.time()-start_time,fix
print numb1,numb2,numb3,numb4,numb5, " allowable_time = ",allow_time
