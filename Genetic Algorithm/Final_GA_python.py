
# coding: utf-8

# #### Imports

# In[1]:

import numpy as np
import random 
import math
import time


# #### Functions

# - Distance Function

# In[2]:

def find_distance(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


# - Cost Function

# In[3]:

def find_cost(supply_coords,demand_coords):
    cost = np.zeros((len(supply_coords),len(demand_coords)))
    for i in range(len(supply_coords)):
        for j in range(len(demand_coords)):
            cost[i,j] = find_distance(supply_coords[i][0],demand_coords[j][0],supply_coords[i][1],demand_coords[j][1])
    return cost


# - Bubble Sort

# In[4]:

def bubble_sort(arrx,cost):
    parents = []
    for i in range(len(arrx)):
        parents.append(arrx[i][1])
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(arrx) - 1):
            if arrx[i][0] > arrx[i + 1][0]:
                arrx[i], arrx[i + 1] = arrx[i + 1], arrx[i]
                parents[i],parents[i+1] = parents[i+1],parents[i]
                swapped = True
    final = []
    for i in range(len(parents)):
        final.append([np.sum(cost*parents[i]),parents[i]])
    return final


# - Initialize Parent Generation

# In[5]:

def init_Parent_Gen(population,num_supply,num_demand,source,dest,cost):
    pi = []
    for i in range(1,num_supply*num_demand+1):
        pi.append(i)  
    parents= []
    for p in range(population):
        X = np.zeros((num_supply,num_demand))
        s = source.copy()
        d = dest.copy()
        test = pi.copy()
        while(len(test)!=0):
            k = random.choice(test)
            i = int(((k-1)/len(d)))
            j = ((k-1)%len(d)) 
            X[i,j] = min(s[i],d[j])
            s[i] = s[i] - X[i,j]
            d[j] = d[j] - X[i,j]
            test.remove(k)
        parents.append([np.sum(cost*X),X])   
    return parents


# - Remove copies in the generation

# In[6]:

def remove_copies(parents):
    x = []
    final = []
    for i in range(len(parents)):
        x.append(parents[i][0])
        if i==0:
            final.append(parents[i])
        if i!=0 and x[i]!=x[i-1]:
            final.append(parents[i])
    return final


# - Roulette's selection

# In[7]:

def selection(parents):
    F = 0
    for i in range(len(parents)):
        F = F+ parents[i][0]
    prb = []
    for i in range(len(parents)):
        prb.append((F- parents[i][0])/(F*(len(parents)-1)))
    cum_prb = []
    for i in range(len(prb)):
        if i==0:
            cum_prb.append(prb[i])
        else:
            cum_prb.append(cum_prb[-1]+prb[i])
    r = random.random()
    error = 1
    e_ind = 1
    for i in range(len(cum_prb)):
        if error>abs(cum_prb[i]-r):
            error=abs(cum_prb[i]-r)
            e_ind = i
    return parents[e_ind][1]


# - Crossover

# In[8]:

def crossover(p,q,num_supply,num_demand,cost):
    D = np.zeros((num_supply,num_demand))
    R = np.zeros((num_supply,num_demand))
    for i in range(num_supply):
        for j in range(num_demand):
            D[i,j] = int((p[i,j]+q[i,j])/2)
            R[i,j] = (p[i,j]+q[i,j])%2
    horP_sum = np.sum(p,axis=1)
    verP_sum = np.sum(q,axis=0)
    R_dash = []
    R_costs = []
    for i in range(100):
        A = np.zeros((num_supply,num_demand))
        test = []
        s = np.sum(R/2,axis=1)
        d = np.sum(R/2,axis=0)
        for i in range(1,len(s)*len(d)+1):
            test.append(i)
        while(len(test)!=0):
            k = random.choice(test)
            i = int(((k-1)/len(d)))
            j = ((k-1)%len(d))
            A[i,j]=0
            if s[i]!=0 and d[j]!=0:
                    A[i,j] = 1
                    s[i] = s[i]-1
                    d[j]= d[j]-1
            if np.sum(D+A,axis=1)[i]>horP_sum[i]:
                if np.sum(D+A,axis=0)[j]>verP_sum[j]:
                    A[i,j] = 0
            test.remove(k)
        if np.sum(cost*A) not in R_costs:
            R_costs.append(np.sum(cost*A))
            R_dash.append(A)
        if len(R_dash)==2:
            break
    flag = 0
    X1 = D+R_dash[0]
    if len(R_dash)>1:
        X2 = D+R_dash[1]
        flag = 1
        return X1,X2,flag
    else:
        return X1,X1,flag


# - Mutation

# In[9]:

def mutate(a):
    n_rows, n_cols = random.randint(2,a.shape[0]),random.randint(2,a.shape[1])
    row = []
    col = []
    while len(row)<n_rows:
        x = random.randint(0,a.shape[0]-1)
        if x not in row:
            row.append(x)
    while len(col)<n_cols:
        x = random.randint(0,a.shape[1]-1)
        if x not in col:
            col.append(x)
    row.sort()
    col.sort()
    A = np.zeros((n_rows,n_cols))
    s = np.sum(a[np.ix_(row,col)],axis=1)
    d = np.sum(a[np.ix_(row,col)],axis=0)   
    test = []
    for i in range(1,len(s)*len(d)+1):
        test.append(i)    
    while(len(test)!=0):
        k = random.choice(test)
        i = int(((k-1)/len(d)))
        j = ((k-1)%len(d)) 
        A[i,j] = min(s[i],d[j])
        s[i] = s[i] - A[i,j]
        d[j] = d[j] - A[i,j]
        test.remove(k) 
    row_itr = 0
    col_itr = 0
    for i in row:
        for j in col:
            a[i,j] = A[row_itr,col_itr]            
            col_itr = col_itr+1
        row_itr = row_itr+1
        col_itr=0
    return a


# - Give offsprings from nth generation

# In[10]:

def do_cross(copy_parents,cross_num,num_supply,num_demand,cost,source,dest):
    offsprings = []
    parents = add_or_remove(copy_parents,1,cost)
    n = int(cross_num*len(copy_parents))
    while n!=1 and n!=0:
        a = selection(parents)
        b = selection(parents)
        if np.sum(cost*a)!=np.sum(cost*b):
            off1 = np.zeros((num_supply,num_demand))
            off2 = np.zeros((num_supply,num_demand))
            off1,off2,flag = crossover(a,b,num_supply,num_demand,cost)
            temp_hor = 0
            temp_vert = 0
            if flag==1:
                horOff1 = np.sum(off1,axis=1)
                vertOff1 = np.sum(off1,axis=0)
                horOff2 = np.sum(off2,axis=1)
                vertOff2 = np.sum(off2,axis=0)
                for i in range(num_supply):
                    if source[i]==horOff1[i] and source[i]==horOff2[i]:
                        temp_hor = temp_hor+1
                for i in range(num_demand):
                    if dest[i]==vertOff1[i] and dest[i]==vertOff2[i]:
                        temp_vert = temp_vert+1
                if temp_hor==num_supply and temp_vert==num_demand:
                    offsprings.append(off1)
                    offsprings.append(off2)
                n=n-2
            else:
                horOff1 = np.sum(off1,axis=1)
                vertOff1 = np.sum(off1,axis=0)
                for i in range(num_supply):
                    if source[i]==horOff1[i]:
                        temp_hor = temp_hor+1
                for i in range(num_demand):
                    if dest[i]==vertOff1[i]:
                        temp_vert = temp_vert+1
                if temp_hor==num_supply and temp_vert==num_demand:
                    offsprings.append(off1)
                n=n-1
    return offsprings


# - Add/Remove cost

# In[11]:

def add_or_remove(parents,flag,cost):
    p = []
    if flag==1:
        for i in range(len(parents)):
            p.append([np.sum(cost*parents[i]),parents[i]])
    if flag==0:
        for i in range(len(parents)):
            p.append(parents[i][1])
    return p


# In[12]:

def do_mutate(copy_parents,mut_num,cost,num_supply,num_demand,source,dest):
    n = int(mut_num*len(copy_parents))
    parents = add_or_remove(copy_parents,1,cost)
    offsprings = []
    temp_hor = 0
    temp_vert = 0
    while n!=0:
        a = selection(parents)
        off1 = mutate(a)
        horOff1 = np.sum(off1,axis=1)
        vertOff1 = np.sum(off1,axis=0)
        for i in range(num_supply):
            if source[i]==horOff1[i]:
                temp_hor = temp_hor+1
        for i in range(num_demand):
            if dest[i]==vertOff1[i]:
                temp_vert = temp_vert+1
        if temp_hor==num_supply and temp_vert==num_demand:
            offsprings.append(off1)
        n = n-1
    return offsprings


# #### INPUT 
print("WELCOME TO TRANSPORTATION OPTIMIZATION PROGRAM")
print("Authors: Anand Karunan and Pranav K Das")
print("\n")
print("\n")
time.sleep(1)
print("Please note that this program solves a balanced transportation problem.")
# In[13]:

while True:
    print("Please input the supply values:")
    source = input().split()
    for i in range(len(source)):
        source[i] = float(source[i])
    print("\n")
    print("Please input the demand values:")
    dest = input().split()
    for i in range(len(dest)):
        dest[i] = float(dest[i])
    if sum(source)==sum(dest):
        break
    else:
        print("Error: Total supply (",sum(source),") != Total Demand (",sum(dest),")")
        
print("\n")
print("Please enter the co-ordinates of supply points")
supply_coords = []
for i in range(len(source)):
    supply_coords.append(input().split())
    for j in range(2):
        supply_coords[i][j] = float(supply_coords[i][j])
print("\n")
print("Please enter the co-ordinates of demand points")
demand_coords = []
for i in range(len(dest)):
    demand_coords.append(input().split())
    for j in range(2):
        demand_coords[i][j] = float(demand_coords[i][j])

# source = [13,12]
# dest = [9, 16]
# supply_coords = [[7., 6.],[2., 5.]]
# demand_coords = [[3., 9.],[4., 6.]]
# source = [30, 70, 50]
# dest = [40, 30, 40, 40]
# cost = np.array([
#     [ 2, 2, 2, 1],
#     [10, 8, 5, 4],
#     [ 7, 6, 6, 8]
# ])
print("\n")
print("Please input initial population size")
population = int(input())
num_supply = len(source)
num_demand = len(dest)
print("\n")
print("Please input fractional crossover required in each generation")
cross_num = float(input())
print("\n")
print("Please input fractional mutation required in each generation")
mutate_num = float(input())
print("\n")
print("Please input number of generations")
num_gen = int(input())
print("\n")
print("Please input convergence criteria")
convergence_no = int(input())
print("\n")


# #### Main loop

# In[14]:

print("Initializing optimization")
time.sleep(1)
print("Optimization Started")
time.sleep(1)

start = time.time()
cost = find_cost(supply_coords,demand_coords)
parents = init_Parent_Gen(population,num_supply,num_demand,source,dest,cost)
parents = bubble_sort(parents,cost)
parents= remove_copies(parents)
no = len(parents)
p = parents.copy()
itr = 0
temp = []
endFinder = 10000
endNum =0
c=[]
m=[]


for i in range(num_gen):

    temp = []
    c = add_or_remove(do_cross(add_or_remove(p,0,cost),cross_num,num_supply,num_demand,cost,source,dest),1,cost)
    m = add_or_remove(do_mutate(add_or_remove(p,0,cost),mutate_num,cost,num_supply,num_demand,source,dest),1,cost)
    
    temp = c+m+p

    temp = bubble_sort(temp,cost)
    temp = remove_copies(temp)
    p = []
    p = temp[:population]

    itr = itr+1
    if endFinder>p[0][0]:
        endFinder=p[0][0]
        if endNum==0:
            endNum= endNum+1
        else:
            endNum=0
    elif endFinder==p[0][0]:
        endNum= endNum+1
        if (endNum-1)==convergence_no:
            # print("FINAL ANSWER")
            # print(p[0][1],np.sum(p[0][1]*cost))
            break

    print("Minimum cost in iteration",i,":",endFinder)

end = time.time()


# ### Final Output

# Minimized cost
time.sleep(1)
# In[15]:
print("\n")
print("Required transporation Matrix")
print(p[0][1])
print("Minimum cost:", p[0][0])
print("Time taken for optimization:", end-start,"sec")

time.sleep(100000)
# Time Taken

# In[17]:



