'''
ipython get_tweets.py 600
listern for 600 seconds every hour
'''

from __future__ import absolute_import

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json
import parser
import sys
import datetime
from configobj import ConfigObj
config = ConfigObj('config')
access_token, access_token_secret = config['access_token'], config['access_token_secret']
consumer_key, consumer_secret = config['consumer_key'], config['consumer_secret']

class Counter():
    def __init__(self, hr_max = 12 + 9, n_runs = 2, flag_test = True):
        self.flag_test = flag_test
        self.hr_max = hr_max
        if self.flag_test:
            self.hr_max = 61
        self.hr_last = -1.0
        self.max_n_runs = n_runs
        self.i_run = 0

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, start_time, hr_max = 12 + 9, time_limit = 60, flag_test = True):
        self.flag_test = flag_test
        self.fh_out_u = open('users_stream.json', 'a')
        self.fh_out_t = open('tweets_stream.json','a')        
        self.time = start_time
        self.limit = time_limit
        self.step = 600
        self.cnt  = 0

        if self.flag_test:
            self.hr_max = 61
            self.step = 1

    def on_data(self, data):
        while (time.time() - self.time) < self.limit:
            try:
                data = data.strip()
                if len(data)> 20:
                    self.dump_user_tweet(json.loads(data))
                    #return True
                i = int((time.time() - self.time)/self.step)
                if i> 0 & i % 2 == 0:
                    print (datetime.datetime.now(), time.time() - self.time)
                return True
            except:
                time.sleep(5)
                pass             
            
        self.fh_out_u.close()
        self.fh_out_t.close()
        print self.cnt, ' tweets saved'
        return False
        exit()

    def on_error(self, status):
        print(status)

    def dump_user_tweet(self, data):
        #data = data.strip()
        tweet, user = parser.parse_tweet(data)
        if parser.is_job_related(tweet):
        #if 1 == 1:
            json.dump(tweet, self.fh_out_t, skipkeys=True)
            self.fh_out_t.write('\n')
            json.dump(user, self.fh_out_u, skipkeys=True)   
            self.fh_out_u.write('\n')
            self.cnt = self.cnt+1

if __name__ == '__main__':
    time_limit = int(sys.argv[1])  # listern k seconds per run

    flag_test = False
    if flag_test: 
        n_run = 2
    counter = Counter(hr_max = 12 + 9, n_runs = 14, flag_test = flag_test)

    unit = 60
    timer = lambda x: x.hour  
    if flag_test:
        unit = 1
        timer = lambda x: x.minute

    while timer(datetime.datetime.now()) < counter.hr_max:
        hr = timer(datetime.datetime.now())
        if hr > counter.hr_last:
            start_time = time.time()
            l = StdOutListener(start_time,  time_limit = time_limit, flag_test = flag_test)
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            stream = Stream(auth, l)
            stream.filter(track=['jobs', 'hiring', 'job'], languages=['en'])

            counter.i_run = counter.i_run + 1
            counter.hr_last = hr
            print 'run ', counter.i_run 

            if counter.i_run >= counter.max_n_runs:
                break  
            print 'done this rounds wait 40'
            time.sleep(40 * unit)#*60)
        else:
            print 'wait for new hour 5'
            time.sleep(5 * unit)#*60)
            pass


 