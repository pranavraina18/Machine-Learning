import matplotlib.pyplot as plt
import numpy as np

#Import data set
clustering_data = np.loadtxt("\clusteringData.csv", delimiter=",",dtype=float)

"""
function takes in feature data and the number of centroids
choose a random index as centeroid as return
"""
def create_centroids(data_set,K):  
    # choose a random index from row size of dataset using K to determine frequency and replacement as false 
    # to not have a dublicate index
    random_index = np.random.choice(len(data_set), size=K, replace=False)
    
    #advanced filtering to return values
    return data_set[random_index]
    
"""
function takes in 2 lists and then
calculates distance based on manhattan formula and axis
"""    
def calculate_distance(list1 , list2):
    return np.sum(np.absolute(list1-list2),axis=2)  

"""
function takes in the feature data and the current array of centroids
checks distances from centroid and returns index array based on shortest distance

function is based on broadcasting and axis
https://numpy.org/devdocs/user/basics.broadcasting.html
"""
def assign_centroids(data_set,centeroid):
    return np.argmin(calculate_distance(data_set,centeroid[:,np.newaxis]),axis=0)            

"""
function takes in the feature data , the current array of centroids and index list of centeroids 
then returns new centroid based on mean
"""
def move_centroids(data_set,centroid_index,centeroid):
    new_centroid =[]

    for i in range(len(centeroid)):
        #advance index filtering
        filter_ls = data_set[centroid_index==i]
        #mean calculation row wize
        new_centroid.append(np.mean(filter_ls, axis=0))
    return new_centroid

"""
function takes in the feature data , the current array of centroids and index list of centeroids 
then returns the current sum of distortion cost of each element with respective to the centeroid
"""
def distortion_cost(data_set,centroid_index,new_centroid_ls):
    cost = np.zeros(len(data_set))    
    for id,item in enumerate(data_set):
        # useing idexing to find which centroid the item belongs to and then fetching that centriods value
        # then apply formula to calculate distortion cost
        cost[id]= np.sum(np.absolute(item-new_centroid_ls[(int(centroid_index[id])-1)])**2) 
    
    #sum to find the final cost           
    return np.sum(cost).round()

"""
function takes in the feature data, the number of centroids, 
the number of iterations and the number of restarts , return the best solution over restart

"""
def restart_KMeans(data_set,K,iterations,restart):
    counter = 0    
    best_soln = []
    while counter < restart: 
        solutions = []
        rnd_centeroids = create_centroids(data_set,K)                
        for i in range(iterations):
            centroid_index_ls = assign_centroids(data_set,rnd_centeroids)      
            new_centeroid = move_centroids(data_set,centroid_index_ls,rnd_centeroids)
            rnd_centeroids = np.array(new_centeroid)
            cost = distortion_cost(data_set,centroid_index_ls,new_centeroid)        
            solutions.append([cost,new_centeroid])
            #checking the last 5 values of a list and using count to check if last 5 items cost in list is same
            if solutions[-5:].count(solutions[-1]) == 5 :
                best_soln.append(solutions[-1])
                counter +=1
                break
               
    return min(best_soln)


"""
Function to generate elbow plot
"""
def elbow_plot(data_set,iterations):
    d_cost = []
    K_cluster = list(range(1,21))
    for k in K_cluster:
        solutions = []
        rnd_centeroids = create_centroids(data_set,k)                
        for i in range(iterations):
            centroid_index_ls = assign_centroids(data_set,rnd_centeroids)      
            new_centeroid = move_centroids(data_set,centroid_index_ls,rnd_centeroids)
            rnd_centeroids = np.array(new_centeroid)
            cost = distortion_cost(data_set,centroid_index_ls,new_centeroid)        
            solutions.append(cost)
            if solutions[-5:].count(solutions[-1]) == 5 :
                d_cost.append(solutions[-1])
                break
      
    plt.plot(K_cluster, d_cost,'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion Cost')    
    plt.title('Elbow Plot using Distortion')
    plt.savefig("K_20.png")
    

#elbow_plot(clustering_data,300)   
restart_KMeans(clustering_data,16,200,10)    
