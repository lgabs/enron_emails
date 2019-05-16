

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Selecting what features we'll use. At first, we'll gather all features to make systematic selection later. 
### However, we'll only work with numerical features to test multiple classifiers, so we'll drop 'email_address',
### which is obviously not useful for classification since everyone should have a unique email address.
### features_list is a list of strings, each of which is a feature name, but the first feature must be "poi".
financial_attributes = ['salary', 
						'deferral_payments', 
						'total_payments',
						'loan_advances', 
						'bonus', 
						'restricted_stock_deferred', 
						'deferred_income',
						'total_stock_value', 
						'expenses',
						'exercised_stock_options', 
						'other',
						'long_term_incentive', 
						'restricted_stock', 
						'director_fees']

email_attributes     = ['to_messages',  
						'from_poi_to_this_person', 
						'from_messages', 
						'from_this_person_to_poi', 
						'shared_receipt_with_poi']

target_label = ['poi']
features_list = target_label + financial_attributes + email_attributes 

### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Remove the outlier found
data_dict.pop('TOTAL')

'''
### Create a new feature: rario total_payments to salary
for w in list(data_dict.keys()):
	data_dict[w]['total_payments_by_salary'] = round(float(data_dict[w]['total_payments'])/float(data_dict[w]['salary']),2)
	if np.isnan(data_dict[w]['total_payments_by_salary']):
		data_dict[w]['total_payments_by_salary'] = 0.0
features_list.append('total_payments_by_salary')
'''
### Store to my_dataset for easy export below.

my_dataset = data_dict

### Verify the proportion of nulls in a the graph -------------------------------

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=False, sort_keys = True)
labels, features = targetFeatureSplit(data)

print(pd.isnull(data).any())

df_data = pd.DataFrame(data, columns = features_list)
missing_ratio = df_data.apply(lambda x: x.isnull().sum()/len(x))

plt.tight_layout()
missing_ratio.plot(kind= 'bar')
plt.xlabel("variable"); plt.ylabel("missing values (%)")
plt.show()
	
## Now, we drop missing and null values to follow the classification problem ----------------
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# look at main infos with head() method
df = pd.DataFrame(data, columns = features_list)
print(df.info())

#print(df.total_payments_by_salary.hist()); plt.show()


### Try a varity of classifiers using a pipeline strategy, also 
### using different parameteres to tune the best fit for each
### classifier

# ----------- PIPELINE ARCHITECTURE --------------
from sklearn.decomposition import PCA
#from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------- DEFINE A CLASSIFIER METHOD FOR PIPELINE -----------------
'''
### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('TreeClf', DecisionTreeClassifier())]
pipeline = Pipeline(steps) # define pipeline object
param_grid = {'PCA__n_components': [2, 3], 'TreeClf__criterion': ['gini', 'entropy'], 'TreeClf__splitter': ['best', 'random']}
'''

### Naive Bayes Classifier
### For the priors probabilities, we tried 0.50/0.50 and 0.88/0.12 (this is the ratio in the original dataset)

from sklearn.naive_bayes import GaussianNB
steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('NB', GaussianNB())]
pipeline = Pipeline(steps) 
param_grid = {'NB__priors': [None, [0.50, 0.50], [0.88, 0.12]],'PCA__n_components': [2,4,6,8,14]}


### SVM Classifier
'''
#from sklearn.svm import SVC
steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('SVC', SVC())]
pipeline = Pipeline(steps) # define pipeline object
param_grid = {'PCA__n_components': [4], 'SVC__kernel': ['rbf', 'poly', 'linear', 'sigmoid', 'precomputed'], 'SVC__C':[0.001,0.1,10,100,10e5], 'SVC__gamma':[0.1,0.01]}
'''
'''
### Random Forest
from sklearn.ensemble import RandomForestClassifier
steps = [('scaler', StandardScaler()), ('PCA', PCA()), ('RFC', RandomForestClassifier())]
pipeline = Pipeline(steps) # define pipeline object
param_grid = {'PCA__n_components': [2], 'RFC__n_estimators': [5], 'RFC__criterion': ['entropy']}
'''


##  -------------CREATE THE CLASSIFIER WITH GRIDSEARCH CV  ------------------------

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(pipeline, param_grid = param_grid, iid = False, cv=3)

##  ------------- GET TRAINING AND TESTING SETS  ------------------------
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify = labels)

##  ------------- FIT THE MODEL AND MEASURE SCORES  ------------------------

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("\nClassifier: {}".format(clf))
print("accuracy_score: {}".format(accuracy_score(labels_test, pred)))
print("recall_score: {}".format(recall_score(labels_test, pred)))
print("F1_score: {}".format(f1_score(labels_test, pred)))
print("confusion_matrix: \n{}".format(confusion_matrix(labels_test, pred)))
print("best parameteres: {}".format(clf.best_params_))

# Looking at PCA importances and main features after discovering the best n_components for the best result from pipeline tests
if steps[1][0] == 'PCA':
	pca = PCA(n_components = 2)
	pca.fit(features_train)
	print("\n\nexplained variance ratios:")
	print(pca.explained_variance_ratio_)
	print(abs(pca.components_))

### Dump the classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
