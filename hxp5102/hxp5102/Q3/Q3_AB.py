import numpy as np
import math
import matplotlib.pyplot as plt

def main():
	print('START Q3_AB\n')
	
	learning_rate =  0.01
	epoch_num = 1000
	#Cleaning of data
	def clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

	def original_data(filename):
		with open(filename, 'r') as file:
			df = file.readlines()
			clean_input = list(map(clean_data, df))
			file.close()
		return clean_input


	

	#Implementing Sigmoid function
	def sigmoid(theta, input):
		
		h = np.dot(theta, input)
		t = 1 / (1 + math.exp(-h))
		return t


	# the logistic function
	def logistic_regression_fit(x_train_array):
		
		misclassification_error = 0
		print("Training")
		theta = np.random.uniform(low=-0.1, high=0.1, size=(x_train_array.shape[1]))
		
		for epoch_i in range(epoch_num):
			
			index = np.random.randint(0, x_train_array.shape[0])
			y_hat = sigmoid(theta, x_train_array[index])
			y = 1 if train_labels[index] == 77 else 0

			if y_hat <= 0.5 and y == 1 or y_hat > 0.5 and y == 0:
				misclassification_error += 1
			
			theta += learning_rate * (y - y_hat) * x_train_array[index]

			
		print("End of Training")
		return theta
	#Reading the data from txt file
	df = original_data('../datasets/Q3_data.txt')



	x = []
	y = []
	z = []
	za= []



	for something in df:
		x.append(float(something[0]))
		y.append(float(something[1]))
		z.append(float(something[2]))
		za.append(float(ord(something[3])))




	za_arr = np.array(za).reshape(120, 1)



	gender = za_arr


	heights = np.array(x).reshape(-1)
	weights = np.array(y).reshape(-1)
	age = np.array(z).reshape(-1)

	x_train = []
	x_train.append(heights)
	x_train.append(weights)
	x_train.append(age)
	x_train_array = np.array(x_train)
	x_train_array = x_train_array.transpose()

	train_labels = np.asarray(gender)
	theta_5 = logistic_regression_fit(x_train_array)

	#Plotting 3d graph for height, weight and age
	g =[]
	for i in za:
		if i == 77 :
			g.append('red')
		else:
			g.append('blue')

	zz = plt.axes(projection = '3d')
	zz.scatter3D(x,y,z,c=g)
	zz.set_xlabel('H')
	zz.set_xlabel('w')
	zz.set_xlabel('A')

	aval = np.linspace(1.5,2.0,800)
	bval = np.linspace(50,100,800)
	s,d = np.meshgrid(aval,bval)

	p = (-theta_5[0]-theta_5[1]*s-theta_5[2]*d)
	zz.plot_surface(s,d,p)
	plt.show()
	
	print('END Q3_AB\n')


if __name__ == "__main__":
    main()
