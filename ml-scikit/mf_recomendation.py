# Load libraries
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#-------------------

filename = 'C:/Users/prana/Desktop/dataset-mf-3.csv'
#raw_data = open(filename, 'rt')
#data = numpy.loadtxt(raw_data, delimiter=",")

names = ['Amount in K', 'Risk(0.1 to 1)', 'Family Status(0.1 to 1)', 'Years(1-10)', 'class']
dataset = read_csv(filename, names=names)


print(dataset)

#--------------------


# shape
print(dataset.shape)
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

#------------------------
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

###### ------------------

# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#######-------------------------------

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#####-------------------------------



#FIXME
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# FIXME : Do for CART

dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

####-----------------------------------

mappingfilename = 'C:/Users/prana/Desktop/dataset-mapping.csv'
d = {}
with open(mappingfilename) as f:
    for line in f:
        (key, val) = line.split(",")
        d[int(key)] = val


example_x = [[5,0.75,0.5,3]]
predicted_class1 = dtc.predict(example_x)
print(predicted_class1)
print(d[int(predicted_class1)])

example_x = [[5,0.75,0.5,3]]
predicted_class2 = knn.predict(example_x)
print(predicted_class2)


print(d[int(predicted_class2)])


###***********************************************************************************

import numpy
from pandas import read_csv
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
filename = 'C:/Users/prana/Desktop/dataset-mf-3.csv'
#raw_data = open(filename, 'rt')
#data = numpy.loadtxt(raw_data, delimiter=",")

names = ['Amount in K', 'Risk(0.1 to 1)', 'Family Status(0.1 to 1)', 'Years(1-10)', 'class']
dataset = read_csv(filename, names=names)


#print(dataset)
print(dataset.head(30))
array = dataset.values
X = array[:,0:4]

# feature extraction
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features)
# summarize selected features
#print(features[0:5,:])
