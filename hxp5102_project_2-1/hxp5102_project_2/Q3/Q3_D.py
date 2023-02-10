import numpy as np
import math

def main():
	print('START Q3_D\n')
	


	learning_rate =  0.01
	epoch_num = 1000

	def clean_data(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

	def original_data(filename):
		with open(filename, 'r') as file:
			df = file.readlines()
			clean_input = list(map(clean_data, df))
			file.close()
		return clean_input




	# the logistic function
	def sigmoid(theta, input):
		
		h = np.dot(theta, input)
		return 1 / (1 + math.exp(-h))

	def logistic_regression_fit(x_train_array):
		
		misclassification_error = 0
		print("Training")
		theta = np.random.uniform(low=-0.1, high=0.1, size=(x_train_array.shape[1]))
		print(theta)
		for epoch_i in range(epoch_num):
			
			index = np.random.randint(0, x_train_array.shape[0])
			y_hat = sigmoid(theta, x_train_array[index])
			y = 1 if train_labels[index] == 77 else 0

			if y_hat <= 0.5 and y == 1 or y_hat > 0.5 and y == 0:
				misclassification_error += 1

			theta += learning_rate * (y - y_hat) * x_train_array[index]

			if (epoch_i + 1) % 100 == 0:
				print("Avg Misclassification Error: {} on epoch {}".format(misclassification_error / 100, epoch_i + 1))
				if misclassification_error / 1000 <= 0.01:
					break
				misclassification_error = 0
		print("End of Training")
		return theta

	df = original_data('../datasets/Q3_data.txt')


	x = []
	y = []
	za= []



	for something in df:
		x.append(float(something[0]))
		y.append(float(something[1]))
		#z.append(float(something[2]))
		za.append(float(ord(something[3])))




	za_arr = np.array(za).reshape(120, 1)



	gender = za_arr


	heights = np.array(x).reshape(-1)
	weights = np.array(y).reshape(-1)



	x_train = []
	x_train.append(heights)
	x_train.append(weights)


	x_train_array = np.array(x_train)
	x_train_array = x_train_array.transpose()


	train_labels = np.asarray(gender)
	theta = logistic_regression_fit(x_train_array)


	def classification(sample):
		y_hat = sigmoid(theta, sample)
		pred_class = 'M' if y_hat >= 0.5 else 'W'
		return pred_class

	count = 0

	for i in range(len(x_train_array)):
		x_test= x_train_array[i][:3]
		y_test = df[i][-1]
	
		df2 = []
		for j in range(len(df)):
			if df[j][:3] !=x_test:
				df2.append(df[j])
	

		prediction = classification(x_test)
		if y_test == prediction:
			count += 1

	print("{}/{} correct predictions using all features".format(count, len(df)))

	print(count/len(df)*100)
	print('END Q3_D\n')


if __name__ == "__main__":
    main()
    