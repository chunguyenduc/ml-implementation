import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt



class LinearRegression():
	"""
    Parameters:
    -----------
    n_iterations: The number of training iterations the algorithm will tune the weights for.
    learning_rate: The step length that will be used when updating the weights.
	w: Our weights
	mse: Cost values for each iteration
    """
	def __init__(self, n_iterations = 1000, learning_rate = 0.01):
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate
		self.w = None
		self.mse = 0
	
	def init_weight(self, n_features):
		# Init weights all zero values
		self.w = np.zeros(n_features)
		
	def fit(self, X, y):
	
		# Insert one more column value 1 for bias
		X = np.insert(X, 0, 1, axis=1)
		
		n_samples, n_features = X.shape
		
		self.training_errors = []
		self.init_weight(n_features=X.shape[1])
		
		# Do gradient descent for n_iterations
		for i in range(self.n_iterations+1):
		
			#Calculate y prediction
			y_pred = np.dot(X, self.w)
			
			# Calculate Gradient Descent for Mean Squared Error
			self.mse = np.mean(0.5 * (y - y_pred)**2)
				
			grad = (1/n_samples) * np.dot(X.T, (y_pred-y))
				
			#Update weights
			self.w -= self.learning_rate * grad
				
		
	def predict(self, X):
		X = np.insert(X, 0, 1, axis=1)
		y_pred = X.dot(self.w)
		return y_pred
		
	
def main():
	np.random.seed(3)
	
	df = pd.read_csv("Salary.csv")


	houseX = np.array(df['YearsExperience'])

	X = houseX[:, np.newaxis]
	y = np.array(df['Salary'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

	# Our model
	model = LinearRegression(n_iterations=1000, learning_rate = 0.043)
	model.fit(X_train, y_train)


	y_pred = model.predict(X_test)

	# sklearn model
	model1 = linear_model.LinearRegression()
	model1.fit(X_train, y_train)

	y_pred1 = model1.predict(X_test)
	
	# Compare our weights and sklearn weights
	print('Our model weights: ', model.w)
	print('Sckit learn weights: ', model1.intercept_, model1.coef_)

	print('\n')

	# Compare our output and sklearn output and actual output
	print('Our model output:	', np.round(y_pred[:5]))
	print('Sckit learn output:	', np.round(y_pred1[:5]))
	print('Actual output:		', y_test[:5])
	
	
	
if __name__ == "__main__":
	main()