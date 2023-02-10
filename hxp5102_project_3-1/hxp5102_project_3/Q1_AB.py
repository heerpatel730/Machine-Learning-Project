import math
from random import seed
from random import randrange
import matplotlib.pyplot as plt
import numpy as np 
from math import log

def main():
    print('START Q1_AB\n')
    def clean_data(line):
        return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

    def original_data(filename):
        with open(filename, 'r') as file:
            df = file.readlines()
            clean_input = list(map(clean_data, df))
            file.close()
        return clean_input

    def convert(dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())
    #training data
    df = original_data('datasets/Q1_train.txt')
    for i in range(len(df[0])-1):
        convert(df, i)
    for sub in df:
        sub[-1] = float(ord(sub[-1]))
    #testing data
    dff = original_data('datasets/Q1_test.txt')
    for i in range(len(dff[0])-1):
        convert(dff, i)
    for sub in dff:
        sub[-1] = float(ord(sub[-1]))
    #data split
    def splitdata(dataset, algorithm, *args):
        scores = list()
        count = 0
        for i in dataset:
            ttdata = list(dataset)
            ttdata.remove(i)
            dtest = list()
            crow = list(i)
            dtest.append(crow)
            crow[-1] = None
            predicted = algorithm(ttdata, dtest, *args)
            if predicted[0] == i[-1]:
                count +=1
            accuracy = count/float(len(dataset))
            scores.append(accuracy)
		
        return scores

    #dataset split on features
    def testsplit(i, v, dataset):
        l, r = list(), list()
        for row in dataset:
            if row[i] < v:
                l.append(row)
            else:
                r.append(row)
        return l, r
 
    # cal the entropy
    def Entropy(g, classes):
        n_instances = float(sum([len(group) for group in g]))
        entro = 0.0
        for group in g:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                #if p is equal to zero or less than it will take 0 as entropy
                if p > 0 :
                    x = p
                    score += p * math.log2(p)
                    #information gain
                    entro += (- score) * (size / n_instances)
                else:
                    entro = 0
        return entro
    min = 10 
    # Select the best split as threshold
    def ssplit(dataset):
        cv = list(set(row[-1] for row in dataset))
        aa, ab, ac, ag = 999, 999, 999, None
        for i in range(len(dataset[0])-1):
            for row in dataset:
                g = testsplit(i, row[i], dataset)
                entro = Entropy(g, cv)
                if entro < ac:
                    aa, ab, ac, ag = i, row[i], entro, g
        return {'i':aa, 'v':ab, 'g':ag}
 
    # Create a last node v
    def tt(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
    
    # child splits for a node 
    def split(node, maxidepth, min, depth):
        l, r = node['g']
        del(node['g'])
        # check for a no split
        if not l or not r:
            node['l'] = node['r'] = tt(l + r)
            return
        #  max depth
        if depth >= maxidepth:
            node['l'], node['r'] = tt(l), tt(r)
            return
        # left child
        if len(l) <= min:
            node['l'] = tt(l)
        else:
            node['l'] = ssplit(l)
            split(node['l'], maxidepth, min, depth+1)
        # right child
        if len(r) <= min:
            node['r'] = tt(r)
        else:
            node['r'] = ssplit(r)
            split(node['r'], maxidepth, min, depth+1)
 
    # decision tree
    def treeform(train, maxidepth, min):
        root = ssplit(train)
        split(root, maxidepth, min, 1)
        return root
 
    # prediction for decision tree
    def predict(node, row):
        if row[node['i']] < node['v']:
            if isinstance(node['l'], dict):
                return predict(node['l'], row)
            else:
                return node['l']
        else:
            if isinstance(node['r'], dict):
                return predict(node['r'], row)
            else:
                return node['r']
 
    
    def dtree(train, test, maxidepth, min):
        tree = treeform(train, maxidepth, min)
    
        predictions = list()
        for row in test:
            prediction = predict(tree, row)
            predictions.append(prediction)
        return(predictions)

    maxidepth = 5
    st=[]
    ss = []
    for maxidepth in range(1, 6):
        print("For depth=",maxidepth)
        scores = splitdata(df, dtree, maxidepth, min)
        # print('Scores: %s' % scores)
        training_accuracy = sum(scores)/float(len(scores))
        print('Train :' , training_accuracy)
        st.append(training_accuracy)
        scores1 = splitdata(dff, dtree, maxidepth, min)
        # print('Scores: %s' % scores1)
        testing_accuracy = sum(scores1)/float(len(scores1))
        print('Test:' , testing_accuracy)
        ss.append(testing_accuracy)

    maxidepth_list = [i for i in range(1,6)]
    x = np.arange(len(maxidepth_list)) + 1 

    plt.plot(x, st)
    plt.plot(x, ss)
    plt.xlabel('maxidepth') 
    plt.ylabel('accuracy') 
    plt.legend() 
    plt.show()
    print("If there is major difference between training and testing accuracy at that time model overfitts but here the difference between both is minimum so there is no overfiting but from all the 5 depts we can say that depth 1 has more overfitted data")
    print('END Q1_AB\n')


if __name__ == "__main__":
    main()
