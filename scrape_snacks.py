import requests  # ref: http://docs.python-requests.org/en/latest/
from bs4 import BeautifulSoup
import pprint
'''
TODO:

'''
from pymongo import MongoClient

def test_mongo(db_name, db_collection, client):
    #client = MongoClient()
    # Initiate Database
    db = client[db_name]
    # Initiate collection
    tab = db[db_collection]

    #tab.insert({'name': 'data_scientist', 'job': 'rich_man'})
    print list(tab.find())
    #tab.update({'name': 'data_scientist'}, {'$set': {'job': 'jobless'}}, multi = True)
    #print list(tab.find())

def get_db_collection(db_name, collection_name, client):
	'''
	create if does not exists; else return handler
	'''
	#client = MongoClient()
	# Initiate Database
	db = client[db_name]
	# Initiate collection
	tab = db[collection_name]
	return tab

def insert_one_doc(doc_dict, tab):
	try:
    	tab.insert(doc_dict)
    except DuplicateKeyError:
    	print "Duplicate keys"

def single_query(url):
    response = requests.get(url)
    if response.status_code != 200:
        print 'WARNING', url,' ', response.status_code
    else:
        return response  # convert to json objects   j.keys() to get all keys

def get_urls(url):
	r = single_query(url)
	soup = BeautifulSoup(r.text, 'html.parser') 
	key_elements = soup.select('div#indexholder ol > li')
	return [e.a.get('href','') for e in key_elements]
'''
Name
Number of snack
Flavor
Cuisine
Series
Composition
Date it became an official snack
Description (text before taste description)
Taste description
'''
def get_1_snack(url):
	#r = requests.get(url)
	#r.status_code
	r = single_query(url)
	soup = BeautifulSoup(r.text, 'html.parser')  # r.text   content
	#<dl> <dt> key

	key_elements = soup.select('dl dt')
	#print key_elements
	keys = [e.text for e in key_elements]
	#print keys

	# <dl> <dd><a> description
	val_elements = soup.select('dl dd')
	#print '\n', val_elements
	'''
	print ''
	for e in val_elements:
		print e.a
	print ''
	'''
	#vals = [[a.text for a in e.findAll('a')] for e in val_elements  ]
	vals = []
	for e in val_elements:

		val_lst = []
		'''
		for a in e.findAll('a'):
			print a
			print a.text
			if len(a.text)>0:
				val_lst.append(a.text)
			else:
				alt_text = a.select('img')[0].get('alt','')
				val_lst.append(alt_text)
		'''
		val_lst.append(e.text)
		for img in e.findAll('img'):
			val_lst.append(img.get('alt',''))
		val_lst = [e.strip() for e in val_lst]
		vals.append(val_lst)

	#print vals
	dict_info = dict(zip(keys, vals))

	descriptions = (soup.select('div#rightstuff')[0]).text.split('\n')
	
	dict_info['title'] = descriptions[1]

	dict_info['description'] = descriptions[3]
	dict_info['taste'] = descriptions[4]
	dict_info['officital'] = descriptions[6].split('on ')[1].replace('.','')

	return dict_info

if __name__ == '__main__':

	client = MongoClient()
	db_name = 'snack_db'
	collection_name = 'snacks'
	db_collection = get_db_collection(db_name, collection_name, client)
	#delete current contents in the table
	db_collection.remove({})

	urls = get_urls('http://www.snackdata.com/')

	'''
	urls = [ 'http://www.snackdata.com/corn_dog',
	'http://www.snackdata.com/seaweed']
	'''
	root = 'http://www.snackdata.com'
	for i, url in enumerate(urls):
		snack_attributes = get_1_snack(root + url)
		#pprint.pprint(snack_attributes)
		insert_one_doc(snack_attributes, db_collection)
		if i % 5 == 0:
			print 'stored %i snaks, last %s' % (i, url)
	#test_mongo(db_name, collection_name, client)

	print db_collection.count()

	#db.connection.drop_database('test')
	'''
	c.drop_database('mydatabase')
	c['mydatabase'].drop_collection('mycollection')
	'''
