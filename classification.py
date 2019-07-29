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

path = "/home/nastya/test_scripts/compare_dyn_models/compare_short/fvs_exp_drebin_anserverbot_droiddream_23.08/"
if len(sys.argv) > 1:
	path = sys.argv[1]

api_fvs_path = "/home/nastya/test_scripts/compare_dyn_models/api_fvs/merge/"
if len(sys.argv) > 2:
	path = sys.argv[2]

row_length = 261
count = 0

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

load_models("anserverbot_models.txt", "/home/nastya/dyn_experiment/anserverbot/")
load_models("opfake_models.txt", "/home/nastya/dyn_experiment/opfake/")
load_models("plankton_models.txt", "/home/nastya/dyn_experiment/plankton/")
load_models("droiddream_models.txt", "/home/nastya/dyn_experiment/droiddream/")



filtered_models = []
model_names = []

empty_api_anserverbot = open("empty_api_vectors_anserverbot.txt", 'r').read().split()
empty_api_opfake = open("empty_api_vectors_opfake.txt", 'r').read().split()
empty_api_plankton = open("empty_api_vectors_plankton.txt", 'r').read().split()
empty_api_droiddream = open("empty_api_vectors_droiddream.txt", 'r').read().split()

essential_models = json.loads(open("essential_models_api.json", 'r').read())

for i,f in enumerate(os.listdir(path)):
	model = json.loads(open(path + f, 'r').read())
	if not os.path.isfile(api_fvs_path + f + '.json'):
		#print model["f_name"], 'no corresponding API model'
		continue
	if not 'benign' in model["f_name"] and not model["f_name"] in essential_models: #wrong condition
		continue
	if model["fv"][0] == 0 or f in empty_api_anserverbot or f in empty_api_opfake or f in empty_api_plankton or f in empty_api_droiddream:
		if model["f_name"] in base_models_list:
			#print 'Base model too small, filtered out', model["f_name"]
			pass
		if f in empty_api_anserverbot or f in empty_api_opfake or f in empty_api_plankton or f in empty_api_droiddream:
			#print model["f_name"], 'filtered, empty api'
			pass
		else:
			print model["f_name"], 'filtered, model too small'
			pass
		continue #filtering empty models
	if model["f_name"] in base_models_list:
		continue #filtering models used for fv building
	filtered_models.append(f)
	model_names.append(model["f_name"])
	
count = len(filtered_models)

X = np.empty((count, row_length))
y = np.empty(count)

api = interesting_api.api_after_feature_selection

count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
count_0 = 0
count_1_indexes = []
count_2_indexes = []
count_3_indexes = []
count_4_indexes = []
count_5_indexes = []
count_0_indexes = []
#TODO maybe round fvs here
for i,f in enumerate(filtered_models):
	model = json.loads(open(path + f, 'r').read())
	model_api = json.loads(open(api_fvs_path + f + '.json', 'r').read())
	api_fv = []
	for elem in sorted(api):
		api_fv.append(model_api[elem])
	X[i] = np.array(model["fv"] + api_fv)

	if 'anserverbot' in model["f_name"]:
		y[i] = 1
		count_1 += 1
		count_1_indexes.append(i)
	if 'opfake' in model["f_name"]:
		y[i] = 2
		count_2 += 1
		count_2_indexes.append(i)
	if 'plankton' in model["f_name"]:
		y[i] = 3
		count_3 += 1
		count_3_indexes.append(i)
	if 'droiddream' in model["f_name"]:
		y[i] = 4
		count_4 += 1
		count_4_indexes.append(i)
	if 'benign' in model["f_name"]:
		y[i] = 0
		count_0 += 1
		count_0_indexes.append(i)

print 'Anserverbot samples:', count_1
print 'Opfake samples:', count_2
print 'Plankton samples:', count_3
print 'DroidDream samples:', count_4
print 'benign samples:', count_0

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
