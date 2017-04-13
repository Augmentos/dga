import cPickle as pickle

from zipfile import ZipFile
import os
import random
import tldextract
import pandas as pd
domains = []
labels = []

DATA_FILE = 'traindata.pkl'


def get_data():
	if (not os.path.isfile(DATA_FILE)):
		df = pd.read_csv("alexa-top-1m.csv",header=None)
		df = df[1:]
		df.columns = ['Rank','Domain']
		domains+=df.Domain.tolist()
		labels += ['benign']*len(df.Domain)
		df = pd.read_csv("dgadataset.csv",header=None)
		df = df[1:]
		df.columns = ['Domain','Botnet']
		domains+=df.Domain.tolist()
		labels += df.Botnet.tolist()
		print 'Dumping file'
		pickle.dump(zip(labels, domains), open(DATA_FILE, 'w'))
		print 'Dumping Completed'
	return pickle.load(open(DATA_FILE))
