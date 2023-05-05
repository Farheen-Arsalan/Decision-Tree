import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import datasets, svm,tree,metrics

data = datasets.load_iris()

print(data.data.shape)

X = data.data 
y = data.target

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=10)

print(Xtrain.shape,Xtest.shape)
print(ytrain.shape,ytest.shape)

# create the decision tree model 

treeModel = tree.DecisionTreeClassifier()
treeModel = treeModel.fit(Xtrain,ytrain)

opTrain = treeModel.predict(Xtrain) # passing training data to get
opTest = treeModel.predict(Xtest)
# its predicted values for training accuracy 
trAcc = metrics.accuracy_score(opTrain,ytrain)
print('Training accuracy : ',trAcc)

# visualize the tree plot
plt.figure(num=1,figsize=(20,15))
tree.plot_tree(treeModel,filled=True,rounded=True)

# testing accuracy 
tstAcc = metrics.accuracy_score(ytest,opTest)
print('Testing Accuracy: ',tstAcc)

# create the randomforest model 
rfModel = ensemble.RandomForestClassifier(n_estimators=50)
rfModel = rfModel.fit(Xtrain,ytrain)
opTrainrf = rfModel.predict(Xtrain)
opTestrf = rfModel.predict(Xtest)

trrAccRf = metrics.accuracy_score(ytrain,opTrainrf)
tstAccRf = metrics.accuracy_score(ytest,opTestrf)
print('Training Accuracy: ',trrAccRf)
print('Testing Accuracy: ',tstAccRf)
