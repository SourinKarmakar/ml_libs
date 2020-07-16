'''
Program : K-means implementation
'''

import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import progressbar

'''
getMaxMin()
Finds minimum and maximum values for each data columns
input - data (i.e. dataframe)
returns a list of tuples containing limits of each column
'''
def get_max_min(df):
    return [(max(df[i].tolist()),min(df[i].tolist())) for i in df.columns]

'''
generate_random_points(k-value,max_min_list)
Generate 'k' randomly initialized points
returns a list random points
'''
def generate_random_points(kvalue,max_min_list):
    random_points = []
    for j in range(kvalue):
        tmp = [(random.random()*(max_min_list[i][0]-max_min_list[i][1])+max_min_list[i][1]) for i in range(len(max_min_list))]
        random_points.append(tmp)
    return random_points

'''
calculate_euclidian_distance(x1,x2)
Calculates euclidian distance between two datapoints(x1,x2)
x1 - list, x2 - list; returns a value (i.e. the distance)
'''
def calculate_euclidian_distance(x1,x2):
    return math.sqrt(sum([(i-j)**2 for i,j in zip(x1,x2)]))

'''
get_classification_labels(data,kpoints)
Assigns cluster label to each datapoint
'''
def get_classification_labels(data,kpoints):
    class_labels = [0 for i in range(len(data))]
    #For each data point
    for i in range(len(data)):
        tmp=[]
        # Calculate the distance between datapoint[i] and k_point[j]
        for j in kpoints:
            dist = calculate_euclidian_distance(j,data.iloc[i].tolist())
            tmp.append(dist) # Holds the distance between datapoint[i] with all the k_points
        # Fetches the k_point's index with least distance and assigns that k_point as label for datapoint[i]
        class_labels[i] = tmp.index(min(tmp))
    return class_labels

'''
segregate_data_per_label(k_points,data_labels)
Seperates out the datapoints belongs to each of the cluster

def segregate_data_per_label(kpoints,xlab):
    pos = [[] for i in range(len(kpoints))]
    for j in range(len(xlab)):
        pos[xlab[j]].append(j)
    return pos
'''

'''
calculate_new_centroids(data,labels,dimension,k_value)
This function takes each data point along with assigned labels for that
data point and calculates the new centroid
let's assume
x1 = [x11,x12,x13,x14] has label 0
x2 = [x21,x22,x23,x24] has label 1
x3 = [x31,x32,x33,x34] has label 0
x4 = [x41,x42,x43,x44] has label 1
so c1 cluster center for label0 is c1=[c11,c12,c13,c14]
where, c11 = (x11+x31)/2 ; c12 = (x21+x32)/2 and so on..
'''
def calculate_new_centroids(data,xlab,dim,k_value):
    kpoints = [[0 for pt in range(dim)] for k in range(k_value)]
    lens = [0 for k in range(k_value)]
    for i in range(len(xlab)):
        kpoints[xlab[i]]= list(np.array(kpoints[xlab[i]])+np.array(data.iloc[i].tolist()))
        lens[xlab[i]]+=1
    for i in range(len(kpoints)):
        kpoints[i] = list(np.divide(kpoints[i],lens[i]))
    return kpoints

'''
k-means implementation
'''
def kmeans(data,k_value,iterations=1000,mode='cluster'):
    df_dim = len(data.columns)
    # Calculating data ranges
    data_ranges = get_max_min(data)
    k_points = generate_random_points(k_value,data_ranges)
    maxiters = iterations
    x_labels = []
    # Just for printing purpose
    bar = progressbar.ProgressBar(maxval=maxiters, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    print("Clustering in progress...")
    # Run for max iterations
    for iters in range(maxiters):
        '''
            Comparing all the points with random initialized centroids
            and calculating distance between the centroids with each data point.
        '''
        #Assigning labels to each data point based on distances of it from cluster centers
        x_labels = get_classification_labels(df,k_points)
        # Calculating new centroid
        k_points = calculate_new_centroids(data,x_labels,df_dim,k_value)
        bar.update(iters+1)
    bar.finish()
    '''
    print(x_labels)
    for i in k_points:
        plt.scatter(i[0],i[1],alpha=1,color='y',marker=',')
    for i in list(set(x_labels)):
        pts = [data.iloc[j] for j in range(len(data)) if x_labels[j]==i]
        x1 = [i[0] for i in pts]
        x2 = [i[1] for i in pts]
        plt.scatter(x1,x2,alpha=1)
    plt.show()
    '''
    print(x_labels)
    if mode=='best_k':
        return [x_labels,k_points]
    return x_labels

'''
choose_best_k(data,lower_k_value,upper_k_value,lambda)

This function takes data, a range of k_values to choose best among these values
and a lambda as a factor for assigning penalty on number of clusters.

This method calculates the sum of squares of distance between centroid and
points surrounding that centroid.
L = summation(summation((xi-cj)^2 for i = all points part of cluster j) for j=no.of cluster)
then L' = L + lambda * no.of_clusters assigning penalty on no.of cluster

Then this function calculates this loss with penalty and plots for all k_values
'''
def choose_best_k(data,min_val=2,max_val=10,set_lambda=0.5):
    if min_val<1:
        print("Minimum value cannot be less than 1")
        return None
    if max_val<min_val:
        print("Maximum value cannot be less than minimum value")
        return None
    loss_list = []
    for i in range(min_val,max_val+1):
        print("Creating cluster with k = "+str(i))
        clusters = kmeans(data,i,mode='best_k')
        labels = clusters[0]
        k_points = clusters[1]
        loss = 0
        for j in range(len(data)):
            for k in range(len(data.columns)):
                loss+=((data.iloc[j].tolist()[k])-k_points[labels[j]][k])**2
        loss = loss + set_lambda*(len(k_points)**2)
        loss_list.append(loss)
        print(loss)
    print(loss_list)
    plt.plot(np.arange(min_val,max_val+1),loss_list,label='Loss value')
    plt.xlabel("K-value")
    plt.ylabel("Loss")
    plt.title("Plotting Loss v/s k-value")
    plt.legend()
    return loss_list.index(min(loss_list))+min_val

if __name__=='__main__':
    df = pd.read_csv("clus_data_dim3.csv")
    print(df.head(10))   
    # choose_best_k(df,min_val=2,max_val=5,lambda=5)
    kmeans(df,4,25)
