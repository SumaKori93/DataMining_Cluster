from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as sm

LABEL_COLOR_MAP = {-1: 'black',
                   0 : 'red',
                   1 : 'blue',
                   2 : 'orange',
                   3 : 'cyan',
                   4 : 'magenta',
                   5 : 'green',
                   6 : 'purple',
                   }

MARKER_SHAPE = {
    -1: u'1',
    1 : u'o',
    2 : u'x',
    3 : u'+',
    4 : u'CARETUP',
    5 : u'd',
    6 : u's',
    7 : u'v'
    }

# fetch data into a dataframe
dataframe=pd.read_csv('Dataset.csv', sep=',',header=None)
dataframe.columns = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba","Fe", "class"]
print(dataframe)

array = dataframe.values
attributes = dataframe.columns.values

#Plot histograms for each attribute - univariate analysis
#create a folder named "UnivariateResults" if it does not exist
if not os.path.exists('UnivariateResults'):
            os.makedirs('UnivariateResults')
i=0
for attribute in attributes:        #for each attribute in the matrix
    if(i<len(attributes)) and (attribute!='ID'):
        fig = plt.figure()
        exec("plt.hist(dataframe['%s'], bins=15,edgecolor='black', linewidth=0.7, facecolor='purple')"%attribute)
        exec("plt.title('Histogram of %s value distribution')" % attribute)
        exec("plt.xlabel('%s')" % (attribute))
        exec("plt.ylabel('Frequency')")
        fig.savefig('./UnivariateResults/'+attribute+'.png')
        plt.close(fig)
        i=i+1

print("End of Univariate analysis : Please refer UnivariateResults forlder for results")

#Plot scatter plots for each set of attributes - bivariate analysis
#create a folder named "BivariateResults" if it does not exist
if not os.path.exists('BivariateResults'):
            os.makedirs('BivariateResults')

i=0
#plot corelation scatter plots for all attributes
for i in range(len(attributes)):
    for j in range(i+1, len(attributes)):
        fig = plt.figure()
        a = (dataframe['class']==1)
        b = (dataframe['class']==2)
        c = (dataframe['class']==3)
        d = (dataframe['class']==4)
        e = (dataframe['class']==5)
        f = (dataframe['class']==6)
        g = (dataframe['class']==7)
        exec("plt.scatter(dataframe['%s'][a], dataframe['%s'][a], c='red',label='1')" %(attributes[i],attributes[j]))
        exec("plt.scatter(dataframe['%s'][b], dataframe['%s'][b], c='green',label='2')" %(attributes[i],attributes[j]))
        exec("plt.scatter(dataframe['%s'][c], dataframe['%s'][c], c='blue',label='3')" %(attributes[i],attributes[j]))
        exec("plt.scatter(dataframe['%s'][d], dataframe['%s'][d], c='cyan',label='4')" %(attributes[i],attributes[j]))
        exec("plt.scatter(dataframe['%s'][e], dataframe['%s'][e], c='yellow',label='5')" %(attributes[i],attributes[j]))
        exec("plt.scatter(dataframe['%s'][f], dataframe['%s'][f], c='orange',label='6')" %(attributes[i],attributes[j]))
        exec("plt.scatter(dataframe['%s'][g], dataframe['%s'][g], c='magenta',label='7')" %(attributes[i],attributes[j]))
        exec("fig.suptitle('%s vs %s')" %(attributes[i],attributes[j]))
        plt.xlabel(attributes[i])
        plt.ylabel(attributes[j])
        plt.legend(loc="lower right")
        fig.savefig('./BivariateResults/'+attributes[i]+'vs'+attributes[j]+'.png')
        plt.close(fig)

print("End of Bivariate analysis : Please refer BivariateResults forder for results")

# separate array into input and output components
X = array[:,1:10]   #remove the ID attribute
Y = array[:,10]   #type of glass/ class attribute
##scaler = Normalizer().fit(X)
##rescaledX = scaler.transform(X)

#Normalize the input data
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

#Clustering algorithm 1 - K means Algorithm
kmeans = KMeans(n_clusters=7, max_iter=500, n_init=20, random_state=0).fit(rescaledX)
##kmeans = KMeans(n_clusters=7).fit(rescaledX)
cluster_labels_kmeans= kmeans.predict(rescaledX)

label_color_kmeans = [LABEL_COLOR_MAP[l] for l in cluster_labels_kmeans] #set colours for each cluster

# Cluster Validation for K means
predictedY_kmeans = [x+1 for x in cluster_labels_kmeans]

confusionmatrix_kmeans=sm.confusion_matrix(Y, predictedY_kmeans) # Confusion Matrix
maxValues = [0 for i in range(confusionmatrix_kmeans.shape[0])]
for i in range(confusionmatrix_kmeans.shape[1]):       #get the max values of each predicted cluster
    for j in range(confusionmatrix_kmeans.shape[0]):
        if(confusionmatrix_kmeans[j][i]>maxValues[i]):
            maxValues[i]=confusionmatrix_kmeans[j][i]
clusterpurity_kmeans = sum(maxValues)/float(sum(sum(confusionmatrix_kmeans)))    #compute cluster purity
print "Clustering purity for K-means clustering :", clusterpurity_kmeans
sil_score_kmeans =  silhouette_score(rescaledX, cluster_labels_kmeans) #compute silhouette score
print "Silhouette coefficient for K-means clustering :", sil_score_kmeans
# End of Cluster Validation

#Clustering algorithm 2 - DBSCAN Algorithm
db = DBSCAN(eps=0.4, min_samples=4, metric='manhattan').fit(rescaledX)   #eps = radius, min_sampes = number of data points that needs to be present around it so that it's a core point
cluster_labels_dbscan = db.fit_predict(rescaledX)

label_color_dbscan = [LABEL_COLOR_MAP[l] for l in cluster_labels_dbscan] #set colours for each cluster

# Cluster Validation for DBSCAN
predictedY_dbscan = [x+1 for x in cluster_labels_dbscan]

confusionmatrix_dbscan=sm.confusion_matrix(Y, predictedY_dbscan) # Confusion Matrix
maxValues = [0 for i in range(confusionmatrix_dbscan.shape[0])]
for i in range(confusionmatrix_dbscan.shape[1]):       #get the max values of each predicted cluster
    for j in range(confusionmatrix_dbscan.shape[0]):
        if(confusionmatrix_dbscan[j][i]>maxValues[i]):
            maxValues[i]=confusionmatrix_dbscan[j][i]
clusterpurity_dbscan = sum(maxValues)/float(sum(sum(confusionmatrix_dbscan)))    #compute cluster purity
print "Clustering purity for DBSCAN clustering :", clusterpurity_dbscan
sil_score_dbscan =  silhouette_score(rescaledX, cluster_labels_dbscan)
print "Silhouette coefficient for DBSCAN clustering :", sil_score_dbscan
# End of Cluster Validation


#Clustering Algorithm 3 - Agglomerative algorithm
agglomerativeModel = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage = "complete")
agglomerativeModel.fit(rescaledX)
cluster_labels_agglomerative=agglomerativeModel.fit_predict(rescaledX)

label_color_agglomerative = [LABEL_COLOR_MAP[l] for l in cluster_labels_agglomerative]

#Cluster Validation for Agglomerative algorithm
predictedY_agglomerative = [x+1 for x in cluster_labels_agglomerative]

confusionmatrix_agglomerative=sm.confusion_matrix(Y, predictedY_agglomerative) # Confusion Matrix
maxValues = [0 for i in range(confusionmatrix_agglomerative.shape[0])]
for i in range(confusionmatrix_agglomerative.shape[1]):       #get the max values of each predicted cluster
    for j in range(confusionmatrix_agglomerative.shape[0]):
        if(confusionmatrix_agglomerative[j][i]>maxValues[i]):
            maxValues[i]=confusionmatrix_agglomerative[j][i]
clusterpurity_agglomerative = sum(maxValues)/float(sum(sum(confusionmatrix_agglomerative)))    #compute cluster purity
print "Clustering purity for Agglomerative clustering :", clusterpurity_agglomerative
sil_score_agglomerative =  silhouette_score(rescaledX, cluster_labels_agglomerative)
print "Silhouette coefficient for Agglomerative clustering :", sil_score_agglomerative
# End of Cluster Validation

#Visualization
##if not os.path.exists('ClusteringVisualisation'):
##    os.makedirs('ClusteringVisualisation')

#Visualization function definition
marker_array = [MARKER_SHAPE[l] for l in Y]

def PlotVisualizationForClustering(title, label_color):
    fig = plt.figure(title)
    ax = Axes3D(fig)
    if(title == 'Original points before clustering'):
        ax.scatter(rescaledX[:, 1], rescaledX[:, 5], rescaledX[:, 7], marker='o', c=label_color) #Have taken RI, Si and Ca as the most representative features
    else:
        for _s, c, _x, _y, _z in zip(marker_array, label_color, rescaledX[:, 1], rescaledX[:, 5], rescaledX[:, 7]):
            ax.scatter(_x, _y, _z, marker=_s, c=c)

##    ax.scatter(rescaledX[:, 1], rescaledX[:, 5], rescaledX[:, 7], c=label_color, marker = marker_array )
    ax.set_xlabel('RI')
    ax.set_ylabel('Si')
    ax.set_zlabel('Ca')
    plt.show()
    plt.close(fig)


label_color_original = [LABEL_COLOR_MAP[1] for i in range(rescaledX.shape[0])]
#Visualization of clustering - function calls
PlotVisualizationForClustering('Original points before clustering',label_color_original)
PlotVisualizationForClustering('K-means with 7 clusters',label_color_kmeans)
PlotVisualizationForClustering('DBSCAN clusters',label_color_dbscan)
PlotVisualizationForClustering('Agglomerative clusters',label_color_agglomerative)

#End of Visualization
