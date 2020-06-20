import numpy as np
import random
import time
import copy
random.seed(250)

start = time.time()

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
numb1=1
numb2=1
numb3=10
numb4=2.5
numb5=5

#text_name=["c202.txt"]
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
    global n
    n=100
    location = [data[:][0]]+data[:][1:n+1]
    location = location[:]
    del data
    locate = [i+1 for i in range(n)]
    prob = [(random.random()*0.5+0.5) for i in locate]
    chosen_n=20
    #40 48
    
    scenario = []
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
    S=[]
    P=[]
    for i in scenario:
        S.append([1]+[1 if j in i[0] else 0 for j in locate ])    
        P.append(i[1])
    
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
    
    global wait
    global maxshift
    global service_start
    global notused
    punish_v={}
    total_score = 0
    location_items=range(len(location))
    for i in location_items:
        punish_v [i]=0
        
    def initial(v_capacity,location,profit,demand,service_value,tw,time_r,service,v_count,c1,c2,c3,allow_time):
        Tmax=tw[0][1]
        routing = []
        score = 0
        wait={}
        maxshift={}
        maxshift[0]=tw[0][1]-service[0]
        wait[0]=0
        service_start = {}
        all_place = location[:]
        all_place.remove(0)
        fix=[]
        while len(routing)<v_count and len(all_place)>1:
            route = [0]
            continue_r = True
            current_capacity = 0
            current_time = 0
            while continue_r:
                insert_l=0
                insert_score = 0
                insert_punish=0
                for i in all_place:
                    start_time = max(current_time+time_r[route[-1]][i],tw[i][0])
                    punish = max(0,start_time-tw[i][1])
                    if i != insert_l and  current_capacity+demand[i]<=v_capacity and start_time<=tw[i][1]+allow_time and start_time+service[i]+time_r[i][0]<=Tmax:
                        current_score =numb1*profit[i]+numb2*service_value[i]-numb3-numb4*(time_r[route[-1]][i])-numb5*punish
                        if current_score>insert_score:
                            insert_score = current_score
                            insert_start = start_time
                            insert_l = i
                            insert_punish=punish
                if insert_l == 0:
                    score-=c2*(time_r[route[-1]][insert_l])
                    route.append(0)
                    continue_r=False
                    break
                route.append(insert_l)
                wait[insert_l] = max(0,insert_start-tw[insert_l][0])
                current_capacity+=demand[i]
                score+=profit[insert_l]+service_value[insert_l]-c1-c2*time_r[route[-1]][insert_l]-c3*insert_punish
                service_start[insert_l] = insert_start
                current_time = insert_start+service[i]
                all_place.remove(insert_l)
                fix.append(insert_l)
                punish_v[insert_l]=insert_punish
            for max_i in range(len(route)):
                i = (max_i)*-1-1    #from back
                loc_i = route[i]    #location of the route at i
                if loc_i != 0:
                    maxshift[loc_i] = min(tw[loc_i][1]+allow_time-service_start[loc_i],wait[route[i+1]]+maxshift[route[i+1]])
            routing.append(route)        
        return routing,score,wait, maxshift,service_start,all_place,fix
    start_time = time.time()
    initial_d= initial(vehicle_capacity,location_items,profit,demand,service_value,tw,timeing,service,vehicle,c1,c2,c3,allow_time)
    for s in range(chosen_n):
        punish_v_c=copy.deepcopy(punish_v)
        route_s=copy.deepcopy(initial_d[0])
        score=initial_d[1]
        wait=initial_d[2].copy()
        maxshift=initial_d[3].copy()
        service_start=initial_d[4].copy()
        fix = initial_d[6][:]
        fix.append(0)
        notused = initial_d[5][:]
        notimprove=0
        service_start[0]=0
        nolocaloptimum = True
        best_score = max(0,score)
        remove_r = 1
        remove_s = 1
        real_exist = S[s]
        for i in route_s:
            for j in i:
                if real_exist[j] <1:
                    score -=profit[j]
        while notimprove<=200:
            while nolocaloptimum:
                for v in route_s:
                    maxratio = 0
                    best_insert = 0
                    cap = sum([demand[i] for i in v])
                    add_score = 0
                    for i in notused:
                        for i2 in range(len(v)-1):
                            j = v[i2]
                            k = v[i2+1]
                            arrival=service_start[j]+service[j]+timeing[i][j]
                            wait_i = max(0,tw[i][0]-arrival)
                            if arrival<=tw[i][1]+allow_time and cap+demand[i]<vehicle_capacity and arrival+wait_i+service[i]+timeing[i][k]<=tw[k][1]+allow_time and arrival+service[i]+timeing[i][k]<=tw[0][1]:
                                waiting = wait_i
                                start = arrival+waiting
                                shift = timeing[i][j]+waiting+service[i]+timeing[i][k]-timeing[j][k]
                                punish=max(start-tw[j][1],0)
                                if shift<=wait[k]+maxshift[k]:
                                    ratio = (profit[i]*real_exist[i]-c1-c2*(timeing[i][j]+timeing[i][k]-timeing[j][k])-c3*punish)**3/shift
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
                                    best_wait = waiting
                                    best_shift = shift
                    if maxratio>0:
                        nolocaloptimum=True
                        v.insert(insert_place,best_insert)
                        wait[best_insert]=best_wait
                        service_start[best_insert] = best_start
                        shift = best_shift
                        before = v[:insert_place]
                        before.reverse()
                        for i in v[insert_place+1:-1]:
                            original_wait=wait[i]
                            wait[i]=max(0,wait[i]-shift)
                            shift = max(0,shift-original_wait)
                            if shift>0:
                                service_start[i]=service_start[i]+shift
                                new_punish=max(0,service_start[i]-tw[i][1])
                                score-=c3*(new_punish-punish_v_c[i])
                                punish_v_c[i]= new_punish
                                maxshift[i]=maxshift[i]-shift
                            else:
                                break
                        be_i=v[insert_place+1]
                        maxshift[best_insert]=min(tw[best_insert][1]-best_start,wait[be_i]+maxshift[be_i])
                        for l in range(len(before)):
                            if l <1:
                                j = best_insert
                            else:
                                j = before[l-1]
                            maxshift[l] = min(maxshift[l],wait[j]+maxshift[j])
                            if maxshift[l]== maxshift[l]:
                                break
                        score +=add_score
                        notused.remove(best_insert)
                        remove_r=1
                    else:
                        nolocaloptimum=False
                        break
            if best_score<score:
                best_score= score
                best_route = route_s
            notimprove +=1
            for v in route_s:
                start_d = min(remove_s,len(v)-2-remove_r)
                for i in range(remove_r):
                    if v[start_d] in fix:
                        continue
                    notused.append(v[start_d])
                    maxshift[v[start_d]]=0
                    wait[v[start_d]]=0
                    service_start[v[start_d]]=0
                    score_b = score
                    score-=profit[v[start_d]]*real_exist[v[start_d]]-c1-c2*(timeing[v[start_d-1]][v[start_d]]+timeing[v[start_d]][v[start_d+1]]-timeing[v[start_d-1]][v[start_d+1]])-c3*punish_v_c[v[start_d]]
                    punish_v_c[v[start_d]]=0
                    del v[start_d]
                    pre = start_d-1
                    loc = -1*(len(v)-start_d)
                    for i in v[start_d:-1]:
                        early_a =service_start[v[pre]]+service[v[pre]]+timeing[v[pre]][i]
                        wait[i]=max(0,tw[i][0]-early_a)
                        early_s = early_a+wait[i]
                        if service_start[i] > early_s:
                            service_start[i] = early_s
                            pre += 1
                            loc+=1
                        else:
                            break
                    while loc>-1*len(a):
                        if tw[v[loc+1]]==0:
                            shift_change=tw[v[loc]][1]-service_start[loc]
                        else:
                            shift_change=min(tw[v[loc]][1]-service_start[v[loc]],wait[v[loc+1]]+maxshift[v[loc+1]])
                        if maxshift[v[loc]]!= shift_change:
                            maxshift[v[loc]]=shift_change
                        else:
                            break
                notused.sort()
                nolocaloptimum=True
                minlen=min(map(len,route_s))-2
                remove_s=remove_s+remove_r
                remove_s+=1
                if remove_s > minlen:
                    remove_s-=minlen
                if remove_s== n/(3*vehicle):
                    remove_s=1
        total_score+=best_score*P[s]
    total_score /=sum(P)
    file1 = open(".txt","a") 
    print text_t," ", time.time()-start_time," ", total_score," ",len(fix)-1,len(route_s)
 
print numb1,numb2,numb3,numb4,numb5, " allowable_time = ",allow_time
