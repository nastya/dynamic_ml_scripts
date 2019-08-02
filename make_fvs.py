#!/usr/bin/python
import os
import sys
import json
import gc
from DynamicActionsModel import AppModel

'''
    Usage: ./make_fvs.py <directory with dynamic models for processing> <directory for storing feature vectors>
                         <file listing malware family names>
'''

models = []

models_for_processing_dir = "/home/nastya/dyn_experiment/benign_test_100/"
if len(sys.argv) > 1:
	models_for_processing_dir = sys.argv[1]

fvs_dir = "test_time/" # where we store generated feature vectors
if len(sys.argv) > 2:
	fvs_dir = sys.argv[2] + '/'

family_names = ["anserverbot", "opfake", "plankton", "droiddream"]
if len(sys.argv) > 3:
	family_names = open(sys.argv[3], 'r').read().split()

if not os.path.exists(fvs_dir):
	os.makedirs(fvs_dir)

def load_models(filename, prefix):
	global models
	for m in open(filename, 'r').readlines():
		models.append(AppModel(prefix + m[:-1]))


def process_dir(directory):
	global models
	count = 0
	for d in os.listdir(directory):
		if os.path.isdir(directory + '/' + d) and not os.path.exists(fvs_dir + d):
			try:
				print 'Processing', directory + '/' + d
				model = AppModel(directory + '/' + d)
				print 'Built model'
				fv = []
				count += 1
				if (count % 50 == 0):
					gc.collect()
				fv.append(model.length())
				fv.append(model.avg_chain_length())
				fv.append(model.max_chain_length())
				for index,m in enumerate(models):
					print 'Comparing with model', index
					res_comp = model.compare(m)
					fv.append(res_comp[0])
					fv.append(res_comp[1])
				fvs_dict = {}
				fvs_dict["f_name"] = directory + '/' + d
				fvs_dict["fv"] = fv
				f = open(fvs_dir + d, 'w')
				f.write(json.dumps(fvs_dict, indent = 4))
				f.close()
			except e:
				print 'Exception', e, 'occured'
	gc.collect()


# Here we call load_models with the following arguments:
# 1) list of filenames of models (here we specify only base models)
# 2) directory where those models are stored

for family_name in family_names:
	load_models("base_models/" + family_name + "_models.txt", "base_models/")

print 'Amount of models:', len(models)

# Generate feature vectors for all models stored in this directory
process_dir(models_for_processing_dir)
