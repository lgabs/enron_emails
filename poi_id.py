#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_attributes = ['salary',
					    'total_payments', 
					    'bonus', 
					    'total_stock_value', 
					    'exercised_stock_options', 					   
					    'long_term_incentive', 
					    'director_fees']
email_attributes = [ 
					'from_poi_to_this_person', 
					'from_this_person_to_poi', 
					'shared_receipt_with_poi']

target_label = ['poi']
# target label must come first, then the rest are features
features_list = target_label + financial_attributes + email_attributes 

### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop('TOTAL')

### Create a new feature: rario total_payments to salary
for w in list(data_dict.keys()):
	data_dict[w]['total_payments_by_salary'] = float(data_dict[w]['total_payments'])/float(data_dict[w]['salary'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Verify if any entry is null
print(pd.isnull(data).any())

### Try a varity of classifiers using a pipeline strategy, also 
### using different parameteres to tune the best fit for each
### classifier

# ----------- PIPELINE ARCHITECTURE --------------
#from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------- DEFINE A CLASSIFIER METHOD FOR PIPELINE -----------------
### Decision Tree Classifier
'''
steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('TreeClf', DecisionTreeClassifier())]
pipeline = Pipeline(steps) # define pipeline object
parameteres = {'PCA__n_components': [2, 3], 'TreeClf__criterion': ['gini', 'entropy'], 'TreeClf__splitter': ['best', 'random']}
'''

### Naive Bayes Classifier
### For the priors probabilities, we tried 0.50/0.50 and 0.88/0.12 (this is the ratio in the original dataset)

steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('NB', GaussianNB())]
pipeline = Pipeline(steps) 
parameteres = {'NB__priors': [None, [0.50, 0.50], [0.88, 0.12]],'PCA__n_components': [4]}

### SVM Classifier
'''
steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('SVC', SVC())]
pipeline = Pipeline(steps) # define pipeline object
parameteres = {'PCA__n_components': [4], 'SVC__kernel': ['rbf', 'poly', 'linear', 'sigmoid', 'precomputed'], 'SVC__C':[0.001,0.1,10,100,10e5], 'SVC__gamma':[0.1,0.01]}
'''
'''
### Random Forest
steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('RFC', RandomForestClassifier())]
pipeline = Pipeline(steps) # define pipeline object
parameteres = {'PCA__n_components': [2], 'RFC__n_estimators': [5], 'RFC__criterion': ['entropy']}
'''


##  -------------CREATE THE CLASSIFIER WITH GRIDSEARCH CV  ------------------------

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(pipeline, param_grid = parameteres, iid = False, cv=3)

##  ------------- GET TRAINING AND TESTING SETS  ------------------------
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify = labels)

##  ------------- FIT THE MODEL AND MEASURE SCORES  ------------------------

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("\nClassifier: {}".format(clf))
print("accuracy_score: {}".format(accuracy_score(labels_test, pred)))
print("recall_score: {}".format(recall_score(labels_test, pred)))
print("confusion_matrix: \n{}".format(confusion_matrix(labels_test, pred)))
print("best parameteres: {}".format(clf.best_params_))

### Dump the classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
