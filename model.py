import util
import time
import sys
import numpy as np
import re
import pandas as pd
import scipy as sp
from operator import itemgetter
import pickle
from scipy import interp
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

def grid_search_lr(txt_lst, y):
	pipeline = Pipeline([
	('vect', CountVectorizer(stop_words='english', analyzer = analyzer, ngram_range=(1, 3)))
		,('tfidf', TfidfTransformer())
	, ('clf', linear_model.LogisticRegression(tol=1e-8, penalty='l2', C=7))
	])

	# specify parameters and distributions to sample from
	param_grid = { 
					'vect__ngram_range':[None, (1, 2), (1,3),(1,4)],
					'clf__C': np.arange(1, 15, 2), 
					'clf__tol': [  1e-4, 1e-5, 1e-8, 1e-10],#, 1e-7] } ] # default 1e-4 0.0001
          			'clf__penalty':['l2','l1']
			}
	# run randomized search
	grid_search = GridSearchCV(pipeline, param_grid,
	                          verbose = 1,  cv = 5, n_jobs=1 )
	start = time.time()
	grid_search.fit(txt_lst, y)
	print("Grid Search took %.2f seconds"
	      " parameter settings." % ((time.time() - start)))
	report(grid_search.grid_scores_, n_top = 5)
	return grid_search

def grid_search_rfc(txt_lst, y):
	pipeline = Pipeline([
	('vect', CountVectorizer(stop_words='english', analyzer = analyzer))
	,('tfidf', TfidfTransformer())
	, ('clf', RandomForestClassifier(n_estimators=100))
	])

	param_grid = { 
					'vect__ngram_range':[None, (1, 2), (1,3),(1,4)],
			   		"clf__max_depth": np.arange(3,30,5),
	              "clf__min_samples_split": np.arange(1, 11,5),
	              "clf__min_samples_leaf": np.arange(2, 11,5),
	              "clf__criterion": ["gini", "entropy"]
			}
	# run randomized search
	n_iter_search = 20
	grid_search = GridSearchCV(pipeline, param_grid,
	                          verbose = 1,  cv = 5, n_jobs=1 )
	start = time.time()
	grid_search.fit(txt_lst, y)
	print("Grid Search took %.2f seconds for %d candidates"
	      " parameter settings." % ((time.time() - start), n_iter_search))
	report(grid_search.grid_scores_, n_top = 5)
	return grid_search


def randomized_search_rfc(txt_lst, y):
	# build a classifier
	pipeline = Pipeline([
	('vect', CountVectorizer(stop_words='english', analyzer = analyzer, ngram_range=(1, 3)))
	,('tfidf', TfidfTransformer())
	, ('clf', RandomForestClassifier(n_estimators=100))
	])

	# specify parameters and distributions to sample from
	param_dist = {  
				'vect__ngram_range':[None, (1, 2), (1,3),(1,4)],
				"clf__max_depth": map(lambda x: int(x), np.logspace(1, 4, 10)), #sp.stats.randint(10,1000),
	              "clf__max_features": map(lambda x: int(x), np.logspace(0, 3, 10)),
	             "clf__min_samples_split": sp.stats.randint(1, 11),
	              "clf__min_samples_leaf": sp.stats.randint(2, 11),
	              "clf__criterion": ["gini", "entropy"]
	              }
	n_iter_search = 50
	grid_search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
	                          verbose = 1, n_iter=n_iter_search, cv = 5, n_jobs=1, scoring = 'accuracy')
	start = time.time()
	grid_search.fit(txt_lst, y)
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings." % ((time.time() - start), n_iter_search))
	report(grid_search.grid_scores_, n_top = 5)
	return grid_search

def get_final_model_pickle(txt_lst, y):
	pipeline = Pipeline([
	('vect', CountVectorizer(stop_words='english', analyzer = analyzer, ngram_range=(1, 3)))
		,('tfidf', TfidfTransformer())
	, ('clf', linear_model.LogisticRegression(tol=1e-8, penalty='l2', C=7))
	])
	pipeline.set_params(vect__ngram_range=None, clf__tol=0.001, clf__penalty= 'l2', clf__C= 6)
	pipeline.fit(txt_lst, y)
	pickle.dump(pipeline,open('model.pkl','w'))

def read_model(fname='model.pkl'):
	pipeline = pickle.load(open(fname, 'r'))
	return pipeline

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def analyzer(doc):
	'''
	INPUT: list of string
	OUTPUT: list of list
	Tokenize, lowercase the document.
	'''
	mytokenizer = TfidfVectorizer(stop_words='english').build_tokenizer()
	#regex = re.compile('<.+?>|[^a-zA-Z]')
	#clean_doc = regex.sub(' ', doc)
	#return [word for word in \
	#	mytokenizer(clean_doc.strip().lower()) if len(word) >= 2]
	doc = ' '.join([word for word in doc.split() if  word[:4] != 'http'])
	return [word for word in \
		mytokenizer(doc.strip().lower()) if len(word) >= 2 ]

def plot_roc_cv(classifier, X, y, n_folds, cv):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating curve')
    plt.legend(loc="lower right")
    plt.show()

def run_estimators(txt_lst, y):
	'''
	run all the estimators and print metrics for test set
	    - INPUT: 
	        train_X: matrix
	        train_y: vector
	        test_X, test_y
	    - OUTPUT: None
	'''
	test_split = 0.25
	train_txt, test_txt, train_y, test_y = train_test_split(txt_lst, \
	        y, test_size=test_split)
	print train_txt.shape
	#print txt_lst[0]

	pipeline = Pipeline([
	('vect', CountVectorizer(stop_words='english', analyzer = analyzer)) #, ngram_range=(1, 2)))
	#('vect', CountVectorizer(stop_words='english', analyzer = analyzer, ngram_range=(1, 3)))
		,('tfidf', TfidfTransformer())
	])
	pipeline.fit(train_txt, train_y) 

	features = np.array(pipeline.named_steps['vect'].get_feature_names())
	train_X = pipeline.transform(train_txt)
	test_X = pipeline.transform(test_txt)
	print 'training set: ', train_X.shape    

	estimators = [  
	#linear_model.LogisticRegression()#,
	linear_model.LogisticRegression(tol=1e-8, penalty='l2', C=7)
	, MultinomialNB()
	, SVC()
	, RandomForestClassifier()
	, AdaBoostClassifier() 
	#, GradientBoostingClassifier() 
	, KNeighborsClassifier(3)
	,SVC(kernel="linear", C=0.025)
	,SVC(gamma=2, C=1)
	]

	metrics_name = ['model'
	, 'accuracy'
	,'precision'
	,'recall'
	,'f1']
	print ', '.join(metrics_name)
	for estimator in estimators:
		t0 = time.time()
		try:
			estimator = estimator.fit(train_X, train_y)
			test_y_pred = estimator.predict(test_X)
			metrics = [accuracy_score(test_y, test_y_pred) 
			,precision_score(test_y, test_y_pred) #, average='binary')
			, recall_score(test_y, test_y_pred)
			, f1_score(test_y, test_y_pred)]
			str_metrics = ['%.3f' % (m) for m in metrics]
			print '%s %s'  %(estimator.__class__.__name__, str_metrics)
		except Exception, e:
			print 'errror in model %s: %s'  % (estimator.__class__.__name__, str(e))
		t1 = time.time() # time it
		print "finish in  %4.4fmin for %s " %((t1-t0)/60,estimator.__class__.__name__)

if __name__ =='__main__':
	df = util.read_labeled_data()
	pd.set_option('max_colwidth',500)
	sys.stdout.flush()

	txt_lst = df.text
	y = df.label
	print df.label.sum()/df.shape[0]

	#run_estimators(txt_lst, y)
	#randomized_search_rfc(txt_lst, y)
	#grid_search_lr(txt_lst, y)
	#grid_search_rfc(txt_lst, y)
	get_final_model_pickle(txt_lst, y)
