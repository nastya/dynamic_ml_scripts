#!/usr/bin/python
import sys
import interesting_api
import json
import os

'''
    Usage: ./make_fvs.py <directory with dynamic models for processing> <directory for storing feature vectors>
'''

models_for_processing_dir = "/home/nastya/dyn_experiment/benign_test_100/"
if len(sys.argv) > 1:
    models_for_processing_dir = sys.argv[1]

fv_save_dir = 'api_fvs/'
if len(sys.argv) > 2:
    fv_save_dir = sys.argv[2] + '/'


iapi = interesting_api.interesting_api_20
for i in range(len(iapi)):
    iapi[i] = iapi[i][1:].replace('/', '.').replace(';', '.')

def count_api_fvs(model_dir):
    if not os.path.exists(fv_save_dir):
        os.makedirs(fv_save_dir)

    bound_size = 100

    for m in os.listdir(model_dir):
        if os.path.isdir(model_dir + m):
            print 'Processing ', model_dir + m
            if os.stat(model_dir + m + '/d_model_api.json').st_size < bound_size:
                continue
            fv = {}
            for api in iapi:
                fv[api] = 0
            try:
                model = json.loads(open(model_dir + m + '/d_model_api.json','r').read())
            except:
                continue
            for state in model:
                for event in model[state]:
                    for thread in model[state][event]["reaction"]:
                        for api in model[state][event]["reaction"][thread]:
                            if api in iapi:
                                fv[api] = 1
            f = open(fv_save_dir + m + '.json', 'w')
            f.write(json.dumps(fv, indent = 4))
            f.close()

count_api_fvs(models_for_processing_dir)
