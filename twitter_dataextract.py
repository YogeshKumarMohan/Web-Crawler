# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import twython
from twython import Twython
from twython import TwythonError
import pandas
import time

t = Twython(app_key='UA0nzn84w89guYmC51LRauSXP', #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
    app_secret='UPC7z3RGdCorXkxYcQV7iFF3rNPjXQBtOT6ZqRZRH3zc7MXDU3',
    oauth_token='84988532-EnkI8CjniUffFcl03gsoCXoL01f8LoUQRglggIgiY',
    oauth_token_secret='rD4AsZ7DVKBSsxX1Ae2dTjLtoKAP38WrQRLZ64zLXgZE8')
 

user_timeline = t.get_user_timeline(screen_name="DignityHealthAZ",include_rts=True)



tweets1 = []
for tweet in user_timeline:
    tweets1.append(tweet['text'])


ids= []
for tweet in user_timeline:
    ids.append(tweet['id'])



ids_dict = {}

for tweet in user_timeline:
    ids_dict[tweet['id']] = tweet['text']
    
    
t = Twython(app_key='UA0nzn84w89guYmC51LRauSXP', #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
    app_secret='UPC7z3RGdCorXkxYcQV7iFF3rNPjXQBtOT6ZqRZRH3zc7MXDU3',
    oauth_token='84988532-EnkI8CjniUffFcl03gsoCXoL01f8LoUQRglggIgiY',
    oauth_token_secret='rD4AsZ7DVKBSsxX1Ae2dTjLtoKAP38WrQRLZ64zLXgZE8')
    

ids = pandas.read_csv("id_samp.csv")
idss =[]
idss = list(ids["IDS"])

import twython
from twython import Twython
from twython import TwythonError
import pandas
import time

x=951
retweets6 = []
page_deleted = []
for i in range(1,8):
    t = Twython(app_key='UA0nzn84w89guYmC51LRauSXP', #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
    app_secret='UPC7z3RGdCorXkxYcQV7iFF3rNPjXQBtOT6ZqRZRH3zc7MXDU3',
    oauth_token='84988532-EnkI8CjniUffFcl03gsoCXoL01f8LoUQRglggIgiY',
    oauth_token_secret='rD4AsZ7DVKBSsxX1Ae2dTjLtoKAP38WrQRLZ64zLXgZE8')
    ids = idss[x:x+59]
    for item in ids: 
        retweeter=[] 
        try:
            xx = t.get_retweets(id=item)
        except TwythonError:
            page_deleted.append(item)              
            continue 
        
        retweeter.append(item)
        for retweet in xx:
            retweeter.append(retweet['user']['description'])
        retweets6.append(retweeter)
    x += 60
    time.sleep(920)
 

RE.append(xy['text'])
for t in xy:
    RE.apend(t['user']['description'])       
    
import pandas   
pd1 = pandas.DataFrame()
pandas.concat([pd,pd1])

pd1.to_csv("retweets.csv")
 


for retweet in xx:
    retweets.append(retweet['user']['description'])  
    
retweets={}
for item in ids:    
    xx = t.get_retweets(id=item)
    for retweet in xx:
        retweets[item]=(retweet['user']['description'])
        
    
user_timeline = t.get_user_timeline(screen_name="DignityHealthAZ",include_rts=True)



entries = re.split("\n", str1)


import csv

idss = []
with open('id_samp.csv', 'rb') as csvfile:
   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
   for row in spamreader:
         idss.append(row)