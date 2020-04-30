import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#sklearn.ensemble contains classifiers for RandomForest, AdaBoost, ExtraTreesClf
from sklearn import ensemble as ek

# from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



""" Select features using ExtraTreesClassifier & validate with sklearn
ML models to see that the features are good enough for a Deep Learning Model """
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

os.chdir(r"E:\00Malicious-PEfile-Detection")
dataset = pickle.loads(open(os.path.join('Data/dataset.pkl'),'rb').read())
print(dataset.groupby(dataset['legitimate']).size())
dataset['legitimate'].replace('Benign', 1, inplace=True)
dataset['legitimate'].replace('Malignant', 0, inplace=True)


#dataset = pd.read_csv('E:/00Malicious-PEfile-Detection/Data/updated_data.csv', low_memory=False)
X = dataset.drop(['Name','md5','legitimate'],axis=1).values
y = dataset['legitimate'].values

#train/dev/test split
X_train, X_split, y_train, y_split = train_test_split(X, y , test_size=0.2, random_state = 0)
X_dev, X_test, y_dev, y_test = train_test_split(X_split, y_split , test_size=0.5, random_state = 0)


#GroupBy no. of Malignant & Benign files to verify False_negatives & False_positives
count = 0; count1 = 0
for val in y_dev:
	if val == 'Benign':
		count += 1
	else:
		count1 += 1
print("Benign: " + str(count))
print("Malignant: " + str(count1))


#Select Features
extratrees = ek.ExtraTreesClassifier(criterion = 'entropy', random_state = 10).fit(X,y)
model = SelectFromModel(extratrees, prefit = True)
#Transform train/dev/test set with selected features
X_train = model.transform(X_train)
X_dev = model.transform(X_dev)
X_test = model.transform(X_test)
nbfeatures = X_train.shape[1]
# print(str(X_train.shape) + "	" + str(X_dev.shape) + "	" + str(X_test.shape))

features = []
index = np.argsort(extratrees.feature_importances_)[::-1][:nbfeatures]

for f in range(nbfeatures):
	print("%d. feature %s (%f)" % (f + 1, dataset.columns[2+index[f]], extratrees.feature_importances_[index[f]]))
	features.append(dataset.columns[2+f])


#Normalize data on X_train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)	
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


### ----------------------------------
### Comment out the block to evaluate performance on these models
### or use the selected features in the Deep Learning model
# #Train & evaluate model	
# model = {"GNB": GaussianNB(),
		 # "DecisionTree": DecisionTreeClassifier(max_depth=10),
		 # "RandomForest": ek.RandomForestClassifier(n_estimators=50),
		 # "Adaboost": ek.AdaBoostClassifier(n_estimators=50)
# }	

# results = {}
# for algo in model:
	# clf = model[algo]
	# clf.fit(X_train,y_train)
	# ress = clf.predict(X_dev)
	# ac_score = accuracy_score(ress,y_dev)
	# print ("%s : %s " %(algo, ac_score))
	# results[algo] = ac_score
	
# winner = max(results, key=results.get)
# print("Best model: " + winner)

# Write the features nd Data to a file
split_data = [X_train.T, X_dev.T, X_test.T, y_train, y_dev, y_test]
open(os.path.join("Models (FeatureSelectionTechnique)/1ExtraTreesClassifier/features.pkl"), 'wb').write(pickle.dumps(features))
open(os.path.join("Models (FeatureSelectionTechnique)/1ExtraTreesClassifier/split_data.pkl"), 'wb').write(pickle.dumps(split_data))


# clf = model[winner]
# ress = clf.predict(X_dev)
# c_matrix = confusion_matrix(ress, y_dev)
# true_negatives, false_positives, false_negatives, true_positives = c_matrix.ravel()

# try:
	# total_predictions = true_negatives + false_negatives + false_positives + true_positives
	# accuracy = 1.0*(true_positives + true_negatives)/total_predictions
	# precision = 1.0*true_positives/(true_positives+false_positives)
	# recall = 1.0*true_positives/(true_positives+false_negatives)
	# f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
	# f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
	# print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
	# print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
	# print("")
# except:
	# print("Got a divide by zero when trying out:", clf)
	# print("Precision or recall may be undefined due to a lack of true positive predicitons.")

# print("False positive rate: " + str((false_positives/(false_positives+true_negatives))*100))
# print("False negative rate: " + str((false_negatives/(false_negatives+true_positives))*100))
	