import json
import pandas as pd
import sys
import random
import numpy as np
from parser import *

class JobCatVecotrizer(object):
	def __init__(self):
		self.pat = {'medical':['nurse','healthcare','physician', ' rn ']
		, 'engineer':['engineer','software','developer',' it ']
		,'analysis':['analyst','analytics','data scientist']
		, 'marketing/sales':['sales','business develop','biz dev', 'marketing','social media','biz dev']
		, 'driver':['truck','driver']
		, 'admin': ['receptionist','admin','admistrative']
		}
		self.job_categories = sorted(self.pat.keys())

	def get_posi_m(self, X):
		'''
			- INPUT: list of texts
			- OUTPUT: 2d array:  n_elements x n_features
		'''
		return np.array([self.get_posi(txt) for txt in X])

	def get_posi(self, x):
		'''
			- INPUT: single string 
			- OUTPUT: list 
		'''
		job_cat_vector = []
		for job in self.job_categories:
			is_match = np.any([p in x for p in self.pat[job]])
			#if is_match:
			job_cat_vector.append(int(is_match))
		return job_cat_vector

def is_ds(x):
	'''
	x: text of the tweet
	'''
	pat = ['data scientist', 'analyst', 'analytics']
	t = [p in x for p in pat]
	return np.any(t)

def read_unique_users(fnames = ['tweets_test.json']):
	d = {}
	users = []
	unique_k = 'user_screen_name'
	for fname_u in fnames:
		fh_in_u = open(fname_u, 'r')
		for line in fh_in_u:
			#print 'line',line[:-1]
			try:
				data = json.loads(line[:-1])
				if not (data[unique_k] in d):
					d[data[unique_k]] = 1
					users.append(data)
			except:
				pass
		fh_in_u.close()
	print 'read in %u users' % len(users)
	df = pd.DataFrame(users)
	df['user_location'] = df['user_location'].apply(lambda x: ' '.join([u.strip().lower() \
		for u in x.replace(',',' ').split()]))
	return df

def read_tweets(fnames = ['users_test.json']):
	tweets = []
	for fname_t in fnames:
		fh_in_t = open(fname_t,'r')	
		for line in fh_in_t:
			try:
				tweet = json.loads(line[:-1])
				if is_job_related(tweet):
					tweet['hash_tags'] = ','.join(tweet.get('hash_tags',[]))
					tweet['urls'] = ','.join(tweet.get('urls',[]))
					tweets.append(tweet)
			except:
				pass
		fh_in_t.close()
	print 'read in %d tweets' % len(tweets)
	return  pd.DataFrame(tweets)

def clean_json(fname, kname):
	'''
	remove duplicate kname
	'''
	fh_in = open(fname, 'r')
	fname_out = fname.replace('.','_new.')
	fh_out = open(fname_out, 'w')
	d = {}
	for line in fh_in:
		data = json.loads(line[:-1])
		if data:
			if not (data[kname] in d):
				fh_out.write(line)
				d[data[kname]] = 1
	fh_in.close()
	fh_out.close()

def pre_process_tweets(df_tweets):
	idx = df_tweets.index[df_tweets['id_str'].isnull()]
	df_tweets = df_tweets.drop(idx, axis = 0)
	#df_tweets = df_tweets.dropna(axis = 0, how = 'all')
	df_tweets['str_text'] = df_tweets['text'].apply(lambda x: x.encode('ascii','ignore'))
	df_tweets['is_rt'] = df_tweets['str_text'].apply(lambda x: x[:2] == 'RT')
	df_tweets['has_hashtag'] = df_tweets['hash_tags'].apply(lambda x: not x == '')
	df_tweets['n_hashtag'] = df_tweets['hash_tags'].apply(lambda x: len(x) - len(x.replace(',','')))
	return df_tweets

def get_data_for_labeling():
	df_users, df_tweets = read_user_tweets(fname_u ='users.json', fname_t = 'tweets.json' )
	df_tweets = pre_process_tweets(df_tweets)
	cond_rt = df_tweets.is_rt 
	df_tweets = df_tweets[~cond_rt]
	fh_out = open('tweet_to_label.txt', 'w')
	n_samples = min(2000,df_tweets.shape[0] )
	idx = random.sample(range(df_tweets.shape[0]),  n_samples)
	d = {}
	for x in idx:
		l = [s for s in df_tweets.iloc[x].str_text.split()]
		if len(l) < 4:
			print df_tweets.iloc[x]
		else:
			str_cleaned = ' '.join(l)
			if str_cleaned not in d:
				d[str_cleaned] = 1
				fh_out.write('\t'.join(['',str_cleaned, df_tweets.iloc[x].created_by, df_tweets.iloc[x].urls]))
				fh_out.write('\n')
	fh_out.close()

def get_quality_tweets(df_tweets):
	df_tweets['str_text'] = df_tweets['str_text'].apply(lambda x: ' '.join([e.lower() \
	               for e in x.split() if e[:4]!= 'http']))
	n = df_tweets.shape[0]
	idx = get_unique_index(df_tweets, 'str_text')
	users_w_high_cnts  = get_val_w_high_cnts(df_tweets, 'created_by').index.values	
	df_tweets = df_tweets.loc[idx]
	cond = map(lambda x: not (x in users_w_high_cnts), df_tweets['created_by'])
	print "before %d after %d" % (n, np.sum(cond))
	return df_tweets[cond] 

def get_unique_index(df_tweets, col_name):
    lst_values_to_exclude = get_val_w_high_cnts(df_tweets, col_name)

    df_tweets = df_tweets.reset_index()
    idx = df_tweets.groupby(col_name)['index'].max()
    df_tweets.set_index('index')

    cond = map(lambda x: not(x in lst_values_to_exclude), idx.index)
    idx_filtered = idx[cond].values
    print "%d tweets, %d unique %s, %d after filtering " % (df_tweets.shape[0], len(idx),col_name,  len(idx_filtered))
    return idx_filtered

def get_val_w_high_cnts(df_tweets, col_name):
    grp = df_tweets.groupby(col_name)['id_str'].count()
    cnt_max = np.mean(grp)+ 3* np.std(grp)
    print 'count by %s mean %.3f std %.3f cut_off %.3f' % (col_name, np.mean(grp), np.std(grp), cnt_max)
    lst_values_to_exclude = grp[grp>cnt_max].index.values
    return grp[grp>cnt_max].sort(ascending = False, inplace=False)

def read_labeled_data():
	n_rows = 1500
	fname = 'tweets_labelled.txt'
	fh_in = open(fname, 'r')
	df = pd.read_csv(fname, delimiter='\t')
	df = df.iloc[:n_rows]
	df.columns = ['label', 'text', 'created_by', 'url']
	df['label'] = df['label'].fillna(0)
	return df

if __name__ == '__main__':
	#get_data_for_labeling()
	pass
