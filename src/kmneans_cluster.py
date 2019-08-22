#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans



data = pd.read_csv('datasets/2019_castle_hills_multi_label.csv', low_memory=False)

print(data.columns)

data_to_cluster = data[['Market_val','BLUSAGECUBICFT','PropertySquareFootage','BCAD_Residential_Footprint','METERSZ','EDU']]
data_to_cluster = data_to_cluster[(data_to_cluster['Market_val'] > 0)]
data_to_cluster = data_to_cluster[(data_to_cluster['BLUSAGECUBICFT'] > 0)]
print(data_to_cluster.head())
print(data_to_cluster.columns)

#print(data_to_cluster['Number_of_Baths'].unique())
print(data_to_cluster['BLUSAGECUBICFT'].min())
print(data_to_cluster['BLUSAGECUBICFT'].max())
print(data_to_cluster['Market_val'].max())
print(data_to_cluster['Market_val'].min())

kmeans = KMeans(n_clusters=4,init='random',random_state=0,n_init=100,n_jobs=4).fit(data_to_cluster.values)
centroids = kmeans.cluster_centers_
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data_to_cluster['Market_val'],data_to_cluster['PropertySquareFootage'],data_to_cluster['BLUSAGECUBICFT'],c=kmeans.labels_.astype(float),s=50,alpha=0.5)
ax.scatter(centroids[:,0],centroids[:,2],centroids[:,1],c='red')
ax.set_xlabel('Market Value')
ax.set_ylabel('PropertySquareFootage')
ax.set_zlabel('Billed Consumption')
plt.show()

# plt.scatter(data_to_cluster['Market_val'],data_to_cluster['PropertySquareFootage'],c=kmeans.labels_.astype(float),s=50,alpha=0.5)
# plt.xlabel('Market value')
# plt.ylabel('Billed consumption in cubic ft.')
# plt.title('Unsupervised Machine Learning K-means clustering of Castle Hills Residential homes')
# plt.scatter(centroids[:,0],centroids[:,2],c='red',s=50,label='Center points')
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.show()




