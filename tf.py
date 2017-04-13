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

	return lstm_results





def create_figs(islstm=True, nfolds=10, force=False):
	if force or (not os.path.isfile(RESULT_FILE)):
		lstm_results = run_experiments(islstm, nfolds)

		results = {'lstm': lstm_results}

		pickle.dump(results, open(RESULT_FILE, 'w'))

	else:
		results = pickle.load(open(RESULT_FILE))
		

		
	

	if results['lstm']:
		lstm_results = results['lstm']
		fpr = []
		tpr = []
		for lstm_result in lstm_results:
			t_fpr, t_tpr, _ = roc_curve(lstm_result['y'], lstm_result['probs'])
			fpr.append(t_fpr)
			tpr.append(t_tpr)
		lstm_binary_fpr, lstm_binary_tpr, lstm_binary_auc = calc_macro_roc(fpr, tpr)

	# Save figure
	from matplotlib import pyplot as plt
	with plt.style.context('bmh'):
		plt.plot(lstm_binary_fpr, lstm_binary_tpr,
				 label='LSTM (AUC = %.4f)' % (lstm_binary_auc, ), rasterized=True)
		
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate', fontsize=22)
		plt.ylabel('True Positive Rate', fontsize=22)
		plt.title('ROC - Binary Classification', fontsize=26)
		plt.legend(loc="lower right", fontsize=22)

		plt.tick_params(axis='both', labelsize=22)
		plt.savefig('results.png')

def calc_macro_roc(fpr, tpr):
	"""Calcs macro ROC on log scale"""
	# Create log scale domain
	all_fpr = sorted(itertools.chain(*fpr))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(len(tpr)):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)        






if __name__ == "__main__":
	create_figs(nfolds=1)