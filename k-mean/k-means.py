import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans


def euclidianDistance(x, y):
	'''
	Euclidian distance between x, y
	--------
	Return
	d: float
	'''

	return np.sqrt(np.sum((x-y)**2))

class k_means:
	def __init__(self, k = 2, threshold = 0.001, max_iter = 300, has_converged = False):

		''' 
		Class constructor

		Parameters
		----------
		- k: number of clusters. 
		- threshold (percentage): stop algorithm when difference between previous cluster 
		and new cluster is less than threshold
		- max_iter: number of times centroids will move
		- has_converged: to check if the algorithm stop or not
		'''


		self.k = k
		self.threshold = threshold
		self.max_iter = max_iter
		self.has_converged = has_converged

	def initCentroids(self, X):
		''' 
		Parameters
		----------
		X: input data. 
		'''
		self.centroids = []

		#Starting clusters will be random members from X set
		indexes = np.random.randint(0, len(X)-1,self.k)
		self.centroids = X[indexes]
		


	def updateCentroids(self, cur_centroids):
		'''
		Class constructor

		Parameters
		----------
		cur_centroids: list of new centroids

		'''
		self.has_converged = True

		for c in range(0, self.k):
			prev_centroid = self.centroids[c]
			cur_centroid  = cur_centroids[c]
			#checking if % of difference between old position and new position is more than threshold

			d = np.sum((cur_centroid - prev_centroid)/prev_centroid * 100.0)

			if  d > self.threshold:
				self.has_converged = False
				self.centroids = cur_centroids

	def fit(self, X):
		'''
		FIT function, used to find clusters

		Parameters
		----------
		X: input data. 
		'''
		#Init list cluster centroids
		self.initCentroids(X)
		

		#Main loop
		for i in range(self.max_iter):  
			#Centroids for this iteration
			cur_centroids = []

			for centroid in range(0,self.k):
				#List samples of current cluster
				samples = []

				for k in range(len(X)):
					d_list = []
					for j in range(self.k):
						#print(X[k])
						d_list.append(euclidianDistance(self.centroids[j], X[k]))

					# Cluster has minimal distance between its centroid and data sample
					c = d_list.index(min(d_list))

					#Store sample to list
					if c == centroid:
						samples.append(X[k])   

				#New centroids of each cluster is calculated by mean of all samples closest to it
				new_centroid = np.average(np.array(samples), axis = 0)

				cur_centroids.append(new_centroid)
				

			self.updateCentroids(cur_centroids)

			if self.has_converged:
				break

		#Each cluster represented by its centroid
		return np.array(self.centroids)

	def predict(self, data):
		''' 
		Parameters
		----------
		data: input data.

		Returns:
		----------
		pred: list cluster indexes of input data 
		'''

		pred = []
		for i in range(len(data)):
			# Create list distances between centroids and data sample
			d_list = []
			for j in range(len(self.centroids)):

				# Calculate distances between current data sample and centroid(using euclidian distance) 
				# Store to d_list
				#TODO 
				d_list.append(euclidianDistance(self.centroids[j], data[i]))

			# Store the Cluster has minimal distance between its centroid and current data sample to pred
			
			pred.append(d_list.index(min(d_list)))
			
		return np.array(pred)
		
np.random.seed(8) 

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3



original_label = np.asarray([0]*N + [1]*N + [2]*N).T

def visualize(X, label, title):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    #you can fix this dpi 
    plt.figure(dpi=120)
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.title(title)
    plt.plot()
    plt.show()
    
visualize(X, original_label, 'Original Label')

model1=k_means(k=3)
print('Centers found by your model:')
print(model1.fit(X))
model1.fit(X)

pred=model1.predict(X)

visualize(X, pred, 'Your model label')
