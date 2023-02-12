import math
import csv
def getDistance(centroid, point):
    dist = 0;
    for i in range(len(point)):
        dist += ((point[i]-centroid[i])**2)
    return math.sqrt(dist) # distance between 2 points

def memb(dist, i,j,m): # formula for membership
    numerator = 0 
    if(dist[j][i]!=0):
        numerator = (1/dist[j][i])**(1/(m-1))
    else:
        return 0
    denominator=0
    for k in range(len(dist)):    
        if(dist[k][i]!=0):
            denominator += (1/dist[k][i])**(1/(m-1)) 
    return numerator/denominator 

def norm_square(x,c):
    ans = 0
    for i in zip(x,c):
        ans+=((i[0]-i[1])**2)
    return ans

def calcNewCentroid(df, mem, dataSize, k, m):
    new_centroid = []
    for j in range(k):
        new_center = [0 for i in range(len(df[0]))] # the numerator part for calculating new centroid
        denominator = 0
        for i in range(dataSize):
            new_center =  [sum(y) for y in zip( new_center, [x*(mem[j][i]**m) for x in df[i]] )]
            denominator += (mem[j][i]**m)
        new_centroid.append([numerator/denominator for numerator in new_center])
    return new_centroid

def calcObjFunction(df,mem, centroid, dataSize, k, m):
    obj_func = 0
    for i in range(dataSize):
        for j in range(k):
            obj_func += (mem[j][i]**m)*(norm_square(df[i], centroid[j]))
    return obj_func

def calcDistMatrix(dist, c, df):
    row = 0
    for centroid in c: #calculating distance matrix
        colum = 0
        for point in df:
            dist[row][colum] = getDistance(centroid, point)
            colum+=1
        row+=1
    return dist

def stoppingCriteria2(center_diff):
    check = False
    for x in center_diff:
        if x == 0:
            check = True
    return check

def caclMembershipMatrix(mem, dist,dataSize,k, m):
    for i in range(k): 
        for j in range(dataSize):
            mem[i][j] = memb(dist, j, i, m)
    return mem

# --- Quantum C means -----
new_csv_file = "sample.csv"
def Qpoints(points): # extrapolation
    new_points = []
    for i in points:
        new_points.append(gen_point(i))
    while(len(new_points)!=1):
        new_lis = []
        for i in new_points[0]:
            for j in new_points[1]:
                new_lis.append(i*j)
        new_points.pop(0)
        new_points.pop(0)
        new_points.append(new_lis)
    return new_points[0]

def write_row(point):
    with open(new_csv_file, 'a') as file:
        write = csv.writer(file)
        write.writerow(point)
        file.close()
def make_head(n):
    x = []
    for i in range(2**(n-1)+1):
        x.append("x"+str(i+1))
    write_row(x)

def gen_point(x):
    return [math.cos(x), math.sin(x)] 