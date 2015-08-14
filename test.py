from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from pymongo import MongoClient
import time
import json

from configobj import ConfigObj
config = ConfigObj('config')
access_token, access_token_secret = config['access_token'], config['access_token_secret']
consumer_key, consumer_secret = config['consumer_key'], config['consumer_secret']

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, db, start_time, time_limit = 60 ):
        self.db = db
        self.time = start_time
        self.limit = time_limit
        self.buffer = ''

    def on_data(self, data):
        while (time.time() - self.time) < self.limit:
            try:
                saveFile = open('raw_tweets.json', 'a')
                saveFile.write(data)
                saveFile.write('\n')
                saveFile.close() 
                print (time.time() - self.time)
                return True
            except:
                time.sleep(5)
                pass
        exit()

    def on_tweet(self, data):
            tweet = json.loads(data)
            self.db.twitter.insert(data)

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    time_limit = argv[1]

    dbname = 'twitter'
    table_name = 'twitter'
    client = MongoClient()
    db = client[dbname]

    start_time = time.time()
    l = StdOutListener(db, start_time, time_limit = time_limit)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=['jobs', 'hiring', 'job'], languages=['en'])