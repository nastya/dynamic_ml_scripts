#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import pandas as pd
import os
import json
import math
import sys
from collections import defaultdict
import statsmodels.formula.api as sm
import interesting_api

'''
    Usage: ./classification.py <directory with feature vectors regarding actions model>
                               <directory with feature vectors regarding API model>
                               <file listing malware family names>
'''

path = "/home/nastya/test_scripts/compare_dyn_models/compare_short/fvs_exp_drebin_anserverbot_droiddream_23.08/"
if len(sys.argv) > 1:
	path = sys.argv[1]

api_fvs_path = "/home/nastya/test_scripts/compare_dyn_models/api_fvs/merge/"
if len(sys.argv) > 2:
	path = sys.argv[2]

family_names = ["anserverbot", "opfake", "plankton", "droiddream"]
if len(sys.argv) > 3:
	family_names = open(sys.argv[3], 'r').read().split()

count = 0

def is_essential_model(model_api):
	# Essential model has permission-protected API in its dynamic trace
	for elem in interesting_api.interesting_api:
		elem = elem.replace('/', '.').replace(';', '.')[1:]
		if elem in model_api and model_api[elem] == 1:
			return True
	return False

def print_results_tex(results):
	table = ''
	for name in results:
		numbers = results[name]
		try:
			table += name.encode('utf-8') + ' '
		except UnicodeDecodeError:
			table += 'name' + ' '
		for number in numbers:
			table += '& '+ str(round(number, 3) * 100) + ' \% '
		table += "\\\\" + '\n'
		table += "\\hline\n"
	return table

results = {}
table_tex = ''
def format_tex(name, numbers):
	global table_tex
	try:
		table_tex += name.encode('utf-8') + ' '
	except UnicodeDecodeError:
		table_tex += 'name' + ' '
	for number in numbers:
		table_tex += '& '+ str(round(number, 3) * 100) + ' \% '
	table_tex += "\\\\" + '\n'
	table_tex += "\\hline\n"
	if not name in results:
		results[name] = []
	results[name].append(numbers)

def split(X_train, X_test, y_train, y_test, test_indexes):
	last_train_id = 0
	last_test_id = 0
	for j in range(count):
		if not j in test_indexes:
			X_train[last_train_id] = X[j]
			y_train[last_train_id] = y[j]
			last_train_id += 1
		else:
			X_test[last_test_id] = X[j]
			y_test[last_test_id] = y[j]
			last_test_id += 1


def column(matrix, i):
    return [row[i] for row in matrix]

def get_metrics(cm):
	fp = 1 - 1.0 * cm[0][0] / np.sum(cm[0,:])
	fn = np.sum(cm[1:,0]) * 1.0 / np.sum(cm[1:,:])
	acc = np.sum(np.diagonal(cm)) * 1.0 / np.sum(cm)
	#Computing precision and recall for binary classification
	precision = np.sum(cm[1:,1:]) * 1.0 / (np.sum(cm[1:,1:]) + np.sum(cm[0,1:]))
	recall = np.sum(cm[1:,1:]) * 1.0 / np.sum(cm[1:,:])
	wc = 0
	for i in range(1, len(cm)):
		for j in range(1, len(cm[i])):
			if j == i:
				continue
			wc += cm[i][j]
	wc = wc * 1.0 / sum(sum(cm[x]) for x in range(1,len(cm)))
	return (fp, fn, wc, acc, precision, recall)


base_models_list = []
def load_models(filename, prefix):
	global base_models_list
	for m in open(filename, 'r').readlines():
		base_models_list.append(prefix + '/' + m[:-1])

for family_name in family_names:
	load_models("base_models/" + family_name + "_models.txt", "base_models/")

filtered_models = []
model_names = []

action_model_fv_len = 0

for i,f in enumerate(os.listdir(path)):
	model = json.loads(open(path + f, 'r').read())
	if not os.path.isfile(api_fvs_path + f + '.json'):
		#no corresponding API model
		continue

	model_api = json.loads(open(api_fvs_path + f + '.json', 'r').read())
	if not 'benign' in model["f_name"] and not is_essential_model(model_api):
		continue
	action_model_fv_len = len(model["fv"])
	if model["fv"][0] == 0 or (not is_essential_model(model_api) and not 'benign' in model["f_name"]):
		continue #filtering empty models
	if model["f_name"] in base_models_list:
		continue #filtering models used for fv building
	filtered_models.append(f)
	model_names.append(model["f_name"])

count = len(filtered_models)

api = interesting_api.api_after_feature_selection
row_length = action_model_fv_len + len(api.keys())

X = np.empty((count, row_length))
y = np.empty(count)


_count = [0] * 5
#TODO maybe round fvs here
for i,f in enumerate(filtered_models):
	model = json.loads(open(path + f, 'r').read())
	model_api = json.loads(open(api_fvs_path + f + '.json', 'r').read())
	api_fv = []
	for elem in sorted(api):
		api_fv.append(model_api[elem])
	X[i] = np.array(model["fv"] + api_fv)

	for j in range(len(family_names)):
		family_name = family_names[j]
		if family_name in model["f_name"]:
			y[i] = j + 1
			_count[j + 1] += 1
	if 'benign' in model["f_name"]:
		y[i] = 0
		_count[0] += 1

for j in range(len(family_names)):
	family_name = family_names[j]
	print family_name + " samples:", _count[j+1]
print 'benign samples:', _count[0]

_j = 0
while _j < len(X[0]):
	value = X[0][_j]
	equal = True
	for _i in range(len(X)):
		if X[_i][_j] != value:
			equal = False
			break
	if equal:
		print _j, 'column is equal values'
		X = np.delete(X, _j, 1)
		row_length -= 1
		continue
	_j += 1


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print X

def experiment():
	import random

	indexes = range(count)
	random.shuffle(indexes)
	K = 4
	test_indexes = []
	fold_size = int(count/K)
	for i in range(K):
		test_indexes.append(indexes[i * fold_size: (i + 1)*fold_size])


	X_train = np.empty((count - fold_size, row_length))
	y_train = np.empty(count - fold_size)
	X_test = np.empty((fold_size, row_length))
	y_test = np.empty(fold_size)



	fp = [0]*K
	fn = [0]*K
	wc = [0]*K
	acc = [0]*K
	precision = [0]*K
	recall = [0]*K

	from sklearn.naive_bayes import GaussianNB
	for i in range(K):
		split(X_train, X_test, y_train, y_test, test_indexes[i])
		# Fitting Naive Bayes to the Training set
		classifier = GaussianNB()
		classifier.fit(X_train, y_train)


		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		# Making the Confusion Matrix
		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		(fp[i], fn[i], wc[i], acc[i], precision[i], recall[i]) = get_metrics(cm)
	print 'Naive Bayes avg values: fp ', sum(fp)/K, 'fn ', sum(fn)/K, 'wc ', sum(wc)/K, 'acc', \
		sum(acc)/K, 'precision', sum(precision)/K, 'recall', sum(recall)/K
	format_tex(u'Байесовский', (sum(fp)/K,  sum(fn)/K, sum(wc)/K, sum(acc)/K, sum(precision)/K, sum(recall)/K))

	# Fitting K-NN to the Training set
	from sklearn.neighbors import KNeighborsClassifier
	for i in range(K):
		split(X_train, X_test, y_train, y_test, test_indexes[i])

		classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean', weights = 'uniform', p = 3)
		classifier.fit(X_train, y_train)

		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		# Making the Confusion Matrix
		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		(fp[i], fn[i], wc[i], acc[i], precision[i], recall[i]) = get_metrics(cm)
	print 'KNN avg values: fp ', sum(fp)/K, 'fn ', sum(fn)/K, 'wc ', sum(wc)/K, 'acc', \
		sum(acc)/K, 'precision', sum(precision)/K, 'recall', sum(recall)/K
	format_tex(u'KNN (k=5)', (sum(fp)/K,  sum(fn)/K, sum(wc)/K, sum(acc)/K, sum(precision)/K, sum(recall)/K))


	# Fitting SVM to the Training set
	from sklearn.svm import SVC
	for i in range(K):
		split(X_train, X_test, y_train, y_test, test_indexes[i])
		classifier = SVC(kernel = 'linear', C = 1000)
		classifier.fit(X_train, y_train)

		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		# Making the Confusion Matrix
		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		(fp[i], fn[i], wc[i], acc[i], precision[i], recall[i]) = get_metrics(cm)
	print 'SVM avg values: fp ', sum(fp)/K, 'fn ', sum(fn)/K, 'wc ', sum(wc)/K, 'acc', \
		sum(acc)/K, 'precision', sum(precision)/K, 'recall', sum(recall)/K
	format_tex(u'SVM', (sum(fp)/K,  sum(fn)/K, sum(wc)/K, sum(acc)/K, sum(precision)/K, sum(recall)/K))

	# Fitting Kernel SVM to the Training set
	from sklearn.svm import SVC
	for i in range(K):
		split(X_train, X_test, y_train, y_test, test_indexes[i])

		classifier = SVC(kernel = 'poly', C = 10, gamma = 0.8, degree = 3)
		classifier.fit(X_train, y_train)

		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		(fp[i], fn[i], wc[i], acc[i], precision[i], recall[i]) = get_metrics(cm)
	print 'Kernel SVM avg values: fp ', sum(fp)/K, 'fn ', sum(fn)/K, 'wc ', sum(wc)/K, 'acc', \
		sum(acc)/K, 'precision', sum(precision)/K, 'recall', sum(recall)/K
	format_tex(u'SVM (ядро poly)', (sum(fp)/K,  sum(fn)/K, sum(wc)/K, sum(acc)/K, sum(precision)/K, sum(recall)/K))

	# Fitting Random Forest Classification to the Training set
	from sklearn.ensemble import RandomForestClassifier
	for i in range(K):
		split(X_train, X_test, y_train, y_test, test_indexes[i])

		classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy',
								min_samples_leaf = 2, max_features = 0.75)
		classifier.fit(X_train, y_train)



		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		(fp[i], fn[i], wc[i], acc[i], precision[i], recall[i]) = get_metrics(cm)
	print 'Random Forest avg values: fp ', sum(fp)/K, 'fn ', sum(fn)/K, 'wc ', sum(wc)/K, 'acc', \
		sum(acc)/K, 'precision', sum(precision)/K, 'recall', sum(recall)/K
	format_tex(u'Ансамбль деревьев решений', (sum(fp)/K,  sum(fn)/K, sum(wc)/K, sum(acc)/K, sum(precision)/K, sum(recall)/K))

	# Fitting Decision Tree Classification to the Training set
	from sklearn.tree import DecisionTreeClassifier
	for i in range(K):
		split(X_train, X_test, y_train, y_test, test_indexes[i])

		classifier = DecisionTreeClassifier(criterion = 'entropy', max_features = 0.75, min_samples_leaf = 3)
		classifier.fit(X_train, y_train)

		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		(fp[i], fn[i], wc[i], acc[i], precision[i], recall[i]) = get_metrics(cm)
	print 'DecisionTree avg values: fp ', sum(fp)/K, 'fn ', sum(fn)/K, 'wc ', sum(wc)/K, 'acc', \
		sum(acc)/K, 'precision', sum(precision)/K, 'recall', sum(recall)/K
	format_tex(u'Дерево решений', (sum(fp)/K,  sum(fn)/K, sum(wc)/K, sum(acc)/K, sum(precision)/K, sum(recall)/K))

	from xgboost import XGBClassifier
	for i in range(K):
		split(X_train, X_test, y_train, y_test, test_indexes[i])

		classifier = XGBClassifier(n_estimators = 20, colsample_bytree = 0.75, max_depth = 5, booster = 'gbtree')
		classifier.fit(X_train, y_train)

		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		(fp[i], fn[i], wc[i], acc[i], precision[i], recall[i]) = get_metrics(cm)
	print 'XGBoost avg values: fp ', sum(fp)/K, 'fn ', sum(fn)/K, 'wc ', sum(wc)/K, 'acc', \
		sum(acc)/K, 'precision', sum(precision)/K, 'recall', sum(recall)/K
	format_tex(u'Градиентный бустинг', (sum(fp)/K,  sum(fn)/K, sum(wc)/K, sum(acc)/K, sum(precision)/K, sum(recall)/K))
	print table_tex

if __name__ == '__main__':
	experiment()
