import numpy as np
import matplotlib.pyplot as plt
import math

def main():
	print('START Q2_D\n')

	#cleaning the data from file
	def clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

	def original_data(filename):
		with open(filename, 'r') as file:
			df = file.readlines()
			clean_input = list(map(clean_data, df))
			file.close()
		return clean_input


	
	#Importing data
	df = original_data('../datasets/Q1_B_train.txt')
	x = []
	y = []
	for something in df:
		x.append(float(something[0]))
		y.append(float(something[1]))


	
	x_arr = np.array(x).reshape(-1)
	y_arr = np.array(y).reshape(128, 1)

	x_train = []
	x_train.append(x_arr)
	y_train = y_arr

	x_train_array = np.array(x_train)
	x_train_array = x_train_array.transpose()
	df_test = original_data('../datasets/Q1_C_test.txt')

	X_te = []
	y_te = []

	for data in df_test:
		X_te.append(float(data[0]))
		y_te.append(float(data[1]))
	x_te_arr = np.array(X_te).reshape(-1)
	y_te_arr = np.array(y_te).reshape(-1)

	x_test = []
	x_test.append(x_te_arr)
	y_test = y_te_arr

	x_test_array = np.array(x_test)
	x_test_array= x_test_array.transpose()

	#calculating weight based on gamma
	def wm(point, x_train_array, gamma): 
		
		m = x_train_array.shape[0] 
			
		w = np.mat(np.eye(m)) 
			
		for i in range(m): 
			xi = x_train_array[i] 
			d = (-2 * gamma * gamma) 
			w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d) 
			
		return w
#prediction
	def predict(x_train_array, y_train, point, gamma = 0.204): 
		
		m = x_train_array.shape[0] 

		x_train_array_ = np.append(x_train_array, np.ones(m).reshape(m,1), axis=1) 
		
		point_ = np.array([point, 1], dtype=object) 
		
		w = wm(point_, x_train_array_, gamma)
		
			
		theta = np.linalg.pinv(x_train_array_.T*(w * x_train_array_))*(x_train_array_.T*(w * y_train)) 
			
		pred = np.dot(point_, theta) 
			
		return theta, pred
#predict the test data
	def plot_predictions(x_train_array, y_train, gamma, nval):
		df_test = original_data('../datasets/Q1_C_test.txt')
		X_te = []
		y_te = []

		for data in df_test:
			X_te.append(float(data[0]))
			y_te.append(float(data[1]))
		x_te_arr = np.array(X_te).reshape(-1)
		y_te_arr = np.array(y_te).reshape(-1)

		x_test = []
		x_test.append(x_te_arr)
		y_test = y_te_arr

		x_test_array = np.array(x_test)
		x_test_array= x_test_array.transpose()

		
		preds = [] 
			
		for point in x_test_array: 
			
			theta, pred = predict(x_train_array, y_train, point, gamma) 
			
			preds.append(pred)
		print(theta)      
		x_test_array = np.array(x_test_array).reshape(nval,1)
		preds = np.array(preds).reshape(nval,1)

		
	#calculating error function
	def error(x_test_array, y_test, gamma):
		preds = []
		nval = x_test_array.shape[0]
		for x in x_test_array:
			theta, pred = predict(x_train_array, y_train, x, gamma)
			preds.append(pred)
		preds = np.array(preds).reshape(nval,1)
		
		MSE = np.square(np.subtract(y_test,preds)).mean() 
	
		RMSE = math.sqrt(MSE)
		print("Root Mean Square Error:\n")
		print(RMSE)


	#D
	#Calling the function to calculate error with first 20 dataset
	def error(x_test_array, y_test, gamma):
		preds = []
		nval = x_test_array.shape[0]
		
		for x in x_test_array:
		
			theta, pred = predict(x_train_array[:20], y_train[:20], x, gamma)
			preds.append(pred)
		preds = np.array(preds).reshape(nval,1)
		MSE = np.square(np.subtract(y_test,preds)).mean() 
	
		RMSE = math.sqrt(MSE)
		print("Root Mean Square Error:\n")
		print(RMSE)

	error(x_test_array, y_test, 0.204)
	print("It is more locally weighted then other based on the evidences we can decided that simple linear regression curve fits the better dataset than Locally weighted")

	#Q2E
	print("Q2_E:")
	print("We can not infer that the data doesn,t perfome better with the function theta which corresponds with the regression but it can be deduced that the function does not locally weights and works better in Simple Linear Regressio")
	print('END Q2_D\n')
if __name__ == "__main__":
    main()
