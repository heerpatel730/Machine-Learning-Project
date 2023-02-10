
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def main():
	print('START Q1_AB\n')


	#Declaring value of k
	k = 10
	#Cleaning data for training
	def clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

	def original_data(filename):
		with open(filename, 'r') as file:
			df = file.readlines()
			clean_input = list(map(clean_data, df))
			file.close()
		return clean_input


	def readFile(dataset_path):
		input_data = original_data(dataset_path)
		return input_data

	
	
	def h(x,theta):
		return np.matmul(x, theta)
	#Cost Implementing
	def cost_function(x, y, theta):
		val = h(x, theta)
		return (val - y).T@(val -y)/(2*y.shape[0])

	#Creating line Eq
	def yline(x, y, theta, learning_rate=0.1, num_epochs=10):
		m = x.shape[0]
		J_all = []

		for _ in range(num_epochs):
			h_x = h(x, theta)
		
			cost_ = (1/m)*(x.T@(h_x - y))
			theta = theta - (learning_rate) * cost_
			J_all.append(cost_function(x, y, theta))

		return theta, J_all

	
	#Test Data
	def test(d, x, theta):
		
		total = 0
		k = 10
		for i in range(1, d + 1 ):
			for k in range(1, k + 1):
				s = np.square(np.sin(k * i * x))
			tsin = theta[i] * s
			total += tsin 
			
		y_pred = theta[0] * 1 + total

		return y_pred
		
	#Reading train data
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
	d  = 6
	k = 10
	for i in range(1, d + 1):
		for k in range(1, k + 1):
			rsin = np.square(np.sin(k * i * x_arr))
			x_train.append(rsin.reshape(-1))

	x_train_array = np.array(x_train)
	x_train_array = x_train_array.transpose()

	df_test = original_data('../datasets/Q1_C_test.txt')
	
	


	
	for i in range(0,7):
		mu = []
		std = []
		d = i
	
		XX = x_train_array[:, :d+1]
		
		YY = y_train
		
		plt.suptitle(f'For d={i}', fontsize = 20)
		plt.plot(XX,YY,'bo')
		plt.show()
		



	print('END Q1_AB\n')
		


if __name__ == "__main__":
    main()