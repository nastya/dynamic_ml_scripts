#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import pandas as pd
import os
import json
import math
import sys
import time
from collections import defaultdict
import statsmodels.formula.api as sm

path = "fvs_exp_drebin_anserverbot_droiddream_23.08/"
api_fvs_path = "/home/nastya/test_scripts/compare_dyn_models/api_fvs/merge/"

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

api = {
    "android.app.Dialog.setCancelable": 24, 
    "java.lang.Object.notifyAll": 10, 
    "java.lang.String.valueOf": 885, 
    "android.app.AlertDialog$Builder.setIcon": 19, 
    "android.app.WallpaperManager.setResource": 3, 
    "android.media.AudioManager.isWiredHeadsetOn": 1, 
    "android.content.res.Resources.openRawResource": 11, 
    "android.util.Log.w": 31, 
    "android.util.Log.v": 217, 
    "android.os.PowerManager.newWakeLock": 12, 
    "java.net.URL.openConnection": 265, 
    "android.util.Log.e": 61, 
    "android.util.Log.d": 251, 
    "android.content.Intent.addFlags": 19, 
    "android.preference.PreferenceManager.getDefaultSharedPreferences": 158, 
    "java.security.MessageDigest.digest": 283, 
    "org.json.JSONTokener.nextValue": 11, 
    "java.io.OutputStreamWriter.write": 4, 
    "android.os.Handler.post": 68, 
    "java.lang.String.substring": 586, 
    "java.util.regex.Pattern.matches": 3, 
    "android.os.Handler.postDelayed": 69, 
    "android.view.View.getId": 383, 
    "android.app.Activity.onDestroy": 181, 
    "android.app.Activity.onCreate": 1066, 
    "java.lang.String.endsWith": 172, 
    "java.lang.Class.forName": 417, 
    "android.app.Activity.dispatchKeyEvent": 3, 
    "android.content.ContentResolver.query": 4, 
    "java.lang.String.indexOf": 533, 
    "android.content.Context.getString": 148, 
    "android.app.Activity.onKeyDown": 41, 
    "java.util.ArrayList.size": 510, 
    "java.lang.StringBuilder.toString": 1053, 
    "android.view.LayoutInflater.inflate": 110, 
    "android.content.res.Resources.getString": 109, 
    "android.graphics.Paint.setColor": 51, 
    "android.widget.Toast.show": 108, 
    "android.widget.VideoView.start": 1, 
    "android.widget.ProgressBar.setProgress": 72, 
    "android.widget.AdapterView.getCount": 51, 
    "android.net.wifi.WifiManager$WifiLock.acquire": 1, 
    "android.os.Handler.sendMessage": 223, 
    "android.app.Notification.setLatestEventInfo": 20, 
    "android.provider.Browser.clearSearches": 19, 
    "java.io.BufferedReader.close": 50, 
    "java.util.Timer.cancel": 32, 
    "java.util.HashMap.put": 276, 
    "android.telephony.TelephonyManager.listen": 16, 
    "android.speech.SpeechRecognizer.setRecognitionListener": 1, 
    "android.app.ProgressDialog.setMessage": 48, 
    "android.telephony.TelephonyManager.getLine1Number": 87, 
    "android.widget.ProgressBar.getProgress": 8, 
    "java.io.BufferedReader.readLine": 70, 
    "android.location.LocationManager.getBestProvider": 4, 
    "android.content.Intent.setData": 28, 
    "android.content.Intent.setClass": 23, 
    "android.location.LocationManager.getLastKnownLocation": 18, 
    "android.app.ProgressDialog.show": 15, 
    "android.net.ConnectivityManager.getAllNetworkInfo": 168, 
    "android.provider.Settings$Secure.getString": 32, 
    "android.app.Dialog.setContentView": 5, 
    "android.app.Activity.onStart": 271, 
    "java.io.FileOutputStream.write": 259, 
    "java.lang.Object.toString": 7, 
    "android.view.KeyEvent.getAction": 21, 
    "android.telephony.TelephonyManager.getDeviceSoftwareVersion": 35, 
    "android.content.Intent.putExtra": 378, 
    "java.util.Random.nextInt": 65, 
    "android.telephony.TelephonyManager.getCellLocation": 1, 
    "android.app.KeyguardManager.newKeyguardLock": 4, 
    "android.app.AlertDialog$Builder.setMessage": 244, 
    "java.lang.Class.getName": 215, 
    "java.io.File.createNewFile": 40, 
    "android.net.ConnectivityManager.getActiveNetworkInfo": 258, 
    "android.media.MediaPlayer.stop": 17, 
    "java.lang.String.length": 679, 
    "android.location.LocationManager.isProviderEnabled": 18, 
    "android.webkit.WebView.setWebViewClient": 75, 
    "android.app.Service.onDestroy": 90, 
    "java.net.URL.openStream": 44, 
    "android.widget.ImageView.setImageResource": 384, 
    "android.os.Handler.obtainMessage": 141, 
    "java.lang.String.toLowerCase": 125, 
    "android.app.AlertDialog$Builder.setCancelable": 188, 
    "android.os.PowerManager$WakeLock.release": 6, 
    "android.telephony.TelephonyManager.getDeviceId": 167, 
    "android.app.Dialog.show": 286, 
    "java.io.InputStreamReader.close": 3, 
    "android.widget.TextView.getText": 17, 
    "android.app.WallpaperManager.setBitmap": 2, 
    "android.telephony.gsm.SmsManager.sendTextMessage": 2, 
    "java.io.OutputStreamWriter.flush": 27, 
    "java.lang.System.loadLibrary": 8, 
    "android.content.Intent.getAction": 40, 
    "android.telephony.TelephonyManager.getSimSerialNumber": 30, 
    "android.database.sqlite.SQLiteDatabase.query": 186, 
    "android.net.wifi.WifiManager.getConnectionInfo": 39, 
    "java.util.regex.Pattern.compile": 61, 
    "android.app.AlertDialog$Builder.setTitle": 251, 
    "java.lang.String.replace": 177, 
    "android.app.PendingIntent.getBroadcast": 115, 
    "android.app.AlertDialog$Builder.setNeutralButton": 6, 
    "android.app.Dialog.findViewById": 5, 
    "android.net.Uri.fromFile": 18, 
    "java.util.StringTokenizer.nextToken": 19, 
    "java.lang.Boolean.booleanValue": 260, 
    "java.lang.Integer.intValue": 564, 
    "android.media.MediaPlayer.start": 31, 
    "android.view.MotionEvent.getY": 57, 
    "java.lang.String.startsWith": 401, 
    "android.media.AudioManager.isBluetoothA2dpOn": 2, 
    "android.content.ContentValues.put": 32, 
    "java.lang.Object.getClass": 405, 
    "android.widget.Toast.makeText": 113, 
    "android.app.Service.onCreate": 119, 
    "android.database.sqlite.SQLiteDatabase.compileStatement": 43, 
    "java.io.FileOutputStream.close": 239, 
    "android.speech.SpeechRecognizer.startListening": 1, 
    "java.lang.String.toString": 314, 
    "java.io.InputStream.read": 189, 
    "android.widget.ImageView.setScaleType": 124, 
    "android.app.ActivityManager.restartPackage": 1, 
    "java.lang.String.trim": 200, 
    "java.lang.Long.parseLong": 259, 
    "android.content.res.Resources.getDrawable": 258, 
    "android.view.MotionEvent.getX": 74, 
    "java.io.OutputStream.flush": 40, 
    "java.lang.Integer.parseInt": 323, 
    "android.media.MediaPlayer.release": 12, 
    "android.view.Window.setFlags": 116, 
    "java.lang.Object.wait": 76, 
    "java.io.File.getParent": 33, 
    "android.widget.EditText.getText": 23, 
    "java.util.ArrayList.add": 505, 
    "android.app.KeyguardManager$KeyguardLock.reenableKeyguard": 2, 
    "android.app.ActivityManager.getRunningTasks": 3, 
    "android.content.Intent.setAction": 118, 
    "android.app.AlertDialog$Builder.show": 21, 
    "java.lang.StringBuffer.append": 443, 
    "java.lang.Thread.start": 488, 
    "java.lang.String.equalsIgnoreCase": 184, 
    "android.telephony.TelephonyManager.getSubscriberId": 113, 
    "android.net.Uri.parse": 424, 
    "java.io.InputStreamReader.read": 44, 
    "java.lang.Integer.toString": 142, 
    "android.net.ConnectivityManager.getNetworkInfo": 31, 
    "java.lang.Thread.sleep": 55, 
    "java.util.regex.Pattern.matcher": 59, 
    "java.lang.StringBuilder.append": 1054, 
    "android.media.MediaRecorder.setAudioSource": 2, 
    "android.widget.TextView.setTextColor": 229, 
    "android.webkit.WebView.loadUrl": 77, 
    "java.io.FileInputStream.close": 45, 
    "android.widget.ImageView.setVisibility": 12, 
    "java.lang.Long.valueOf": 305, 
    "android.widget.VideoView.setVideoURI": 3, 
    "android.net.wifi.WifiManager.getWifiState": 9, 
    "android.os.PowerManager$WakeLock.acquire": 8, 
    "android.content.Intent.getExtras": 167, 
    "android.app.Activity.onResume": 223, 
    "android.widget.ProgressBar.incrementProgressBy": 16, 
    "android.os.Vibrator.vibrate": 14, 
    "android.accounts.AccountManager.getAccounts": 2, 
    "java.lang.reflect.Field.get": 176, 
    "android.app.AlertDialog$Builder.setPositiveButton": 270, 
    "java.lang.Long.longValue": 207, 
    "java.io.File.mkdirs": 21, 
    "android.app.ProgressDialog.setProgress": 8, 
    "android.content.Intent.setDataAndType": 7, 
    "android.app.AlertDialog$Builder.setNegativeButton": 92, 
    "android.net.wifi.WifiManager.isWifiEnabled": 6, 
    "android.net.wifi.WifiManager$WifiLock.release": 1, 
    "java.lang.StringBuffer.toString": 408, 
    "android.widget.TextView.setText": 689, 
    "android.content.Intent.getStringExtra": 39, 
    "android.media.MediaPlayer.pause": 11, 
    "java.lang.Integer.valueOf": 592, 
    "android.telephony.TelephonyManager.getNetworkOperator": 26, 
    "java.lang.Class.getMethod": 519, 
    "android.telephony.TelephonyManager.getSimCountryIso": 10, 
    "java.lang.String.replaceAll": 39, 
    "java.util.HashMap.get": 260, 
    "android.app.Dialog.setTitle": 13, 
    "android.app.ActivityManager.killBackgroundProcesses": 4, 
    "java.lang.String.contains": 163, 
    "java.util.Timer.schedule": 118, 
    "java.util.Calendar.get": 66, 
    "android.provider.Browser.clearHistory": 19, 
    "java.lang.String.charAt": 337, 
    "android.media.MediaPlayer.reset": 1, 
    "android.telephony.TelephonyManager.getSimOperatorName": 15, 
    "java.io.File.getPath": 246, 
    "java.lang.String.format": 49, 
    "android.view.KeyEvent.getKeyCode": 11, 
    "java.util.StringTokenizer.hasMoreTokens": 18, 
    "java.lang.Boolean.valueOf": 476, 
    "android.view.MotionEvent.getAction": 114, 
    "android.database.sqlite.SQLiteDatabase.execSQL": 175, 
    "android.app.KeyguardManager$KeyguardLock.disableKeyguard": 3, 
    "android.provider.Settings$System.putString": 1, 
    "android.telephony.SmsManager.sendTextMessage": 130, 
    "java.net.URL.getContent": 2, 
    "java.lang.String.getBytes": 386, 
    "java.io.OutputStreamWriter.close": 37, 
    "android.hardware.Camera.open": 4, 
    "android.provider.Settings$System.putInt": 1, 
    "java.lang.String.equals": 693, 
    "android.graphics.Paint.setStrokeWidth": 27, 
    "java.io.OutputStream.write": 26, 
    "android.graphics.Paint.setAntiAlias": 13, 
    "android.content.res.AssetManager.open": 194, 
    "java.io.File.exists": 295, 
    "android.app.AlertDialog$Builder.create": 265, 
    "java.net.URLEncoder.encode": 82, 
    "android.location.LocationManager.requestLocationUpdates": 12, 
    "android.app.ProgressDialog.setProgressStyle": 10, 
    "java.lang.String.split": 576, 
    "java.lang.reflect.Method.invoke": 517, 
    "android.accounts.AccountManager.getAccountsByType": 6, 
    "android.app.NotificationManager.notify": 15, 
    "android.view.Display.getWidth": 150, 
    "android.content.res.AssetManager.list": 24, 
    "android.media.RingtoneManager.setActualDefaultRingtoneUri": 5
}


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

		time_start = time.time()
		# Predicting the Test set results
		y_pred = classifier.predict(X_test)
		time_stop = time.time()
		print 'Time:', time_stop-time_start, len(X_test)

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
