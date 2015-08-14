import json
import numpy as np

def parse_tweet(tweet):
	'''
		- INPUT: 
			tweet: dictonary
	'''
	t, u = {}, {}
	if tweet.has_key('text'):
		text = tweet['text']
		retweet_count = tweet['retweet_count']
		created_at = tweet['created_at']
		hash_tags = [ h['text'] for h in tweet['entities']['hashtags']]
		favorite_count = tweet['favorite_count']
		created_by = tweet['user']['screen_name']
		id_str = tweet['id_str']
		urls = [u['expanded_url'] for u in tweet['entities']['urls']]
		in_reply_to = tweet['in_reply_to_status_id_str']
		tweet_keys = '''
		text 
		retweet_count 
		created_at 
		hash_tags 
		favorite_count 
		created_by
		id_str
		urls 
		in_reply_to
		 '''.split()
		t = {}
		for k in tweet_keys:
			t[k] = eval(k)

		user_screen_name = tweet['user']['screen_name']
		user_name = tweet['user']['name']
		user_created_at = tweet['user']['created_at']
		user_description = tweet['user']['description']
		user_friends_count = tweet['user']['friends_count']
		user_followers_count = tweet['user']['followers_count']
		user_following = tweet['user']['following']
		user_location= tweet['user']['location']
		user_url = tweet['user']['url']
		user_status_cnt_life = tweet['user']['statuses_count']

		user_keys = '''
		user_screen_name  
		user_name 
		user_created_at 
		user_description 
		user_friends_count 
		user_followers_count 
		user_following 
		user_location 
		user_url 
		user_status_cnt_life 
		'''.split()	
		u = {}
		for k in user_keys:
			u[k] = eval(k)
	return t, u

def is_job_related(t):
	flag = False
	if t:
		l = []
		l.append(np.any(['job' in e.lower() for e in t['hash_tags']]))
		l.append(np.any(['hire' in e.lower() for e in t['hash_tags']]))	
		l.append(np.any(['hiring' in e.lower() for e in t['hash_tags']]))		
		l.append(np.any(['/job' in url.lower() for url in t['urls']]))
		url_pat = '''
			jobs/
			/jobs.
			/jobs/
			/job/
			jobs.
			/jobs
			jobs.com
			jobvite
			jobdetail
			?job=
			/career
			career/
			careers/
		'''.split()
		for p in url_pat:
			l.append(np.any([p in url.lower() for url in t['urls']]))
		#print l
		#print np.any(l)
		flag = np.any(l)
	return flag

if __name__ == '__main__':
	tweets = []
	fname = 'raw_tweets.json'
	fh_out_u = open('users.json', 'w')
	fh_out_t = open('tweets.json','w')
	with open(fname) as fh_in:
		for line in fh_in:
			#print line
			line.strip()
			if len(line) > 20:
				tweet = json.loads(line)
				t, u = parse_tweet(tweet)
				if is_job_related(t):
					json.dump(t, fh_out_t, skipkeys=True)
					fh_out_t.write('\n')
					json.dump(u, fh_out_u, skipkeys=True)	
					fh_out_u.write('\n') 
	fh_out_u.close()
	fh_out_t.close()
