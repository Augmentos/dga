import itertools
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import mylstm as lstm

from scipy import interp
from sklearn.metrics import roc_curve, auc
RESULT_FILE = 'results.pkl'

def run_experiments(islstm=True, nfolds=10):
	"""Runs all experiments"""
	
	lstm_results = None

	
	if islstm:
		lstm_results = lstm.run(nfolds=nfolds)

	





def create_figs(islstm=True, nfolds=10, force=False):
	if force or (not os.path.isfile(RESULT_FILE)):
		lstm_results = run_experiments(islstm, nfolds)

		




if __name__ == "__main__":
	create_figs(nfolds=1)