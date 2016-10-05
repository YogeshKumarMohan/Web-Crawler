# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:36:15 2016

@author: Yogesh
"""


# Importing required packages
import pandas
import csv
from apiclient.discovery import build #pip install google-api-python-client
from apiclient.errors import HttpError #pip install google-api-python-client
from oauth2client.tools import argparser #pip install oauth2client

# Create Google Youtube Data API and get the Developer Key
DEVELOPER_KEY = "######################################" 
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Run this code on the python console. Running this ipython console would generate an error
# Create a object for search term, Enter the search term for which video data should be extracted
argparser.add_argument("--q",  help = "Search term", default="Popular Videos  - Cleveland Clinic")

argparser.add_argument("--max-results", help="Max results", default=50)
args = argparser.parse_args()
options = args

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

search_response = youtube.search().list(
 q=options.q,
 type="video",
 part="id,snippet",
 maxResults=options.max_results
).execute()

videos = {}


# populating the video details in a dictionary and storing it in a pandas DataFrame
for search_result in search_response.get("items", []):
    if search_result["id"]["kind"] == "youtube#video":
        videos[search_result["id"]["videoId"]] = search_result["snippet"]["title"]
        
s = ','.join(videos.keys())

videos_list_response = youtube.videos().list(
 id=s,
 part='id,statistics'
).execute()


res = []
for i in videos_list_response['items']:
    temp_res = dict(v_id = i['id'], v_title = videos[i['id']])
    temp_res.update(i['statistics'])
    res.append(temp_res)
    
  
pd_youtube = pandas.DataFrame(res)


# Code for extracting Video Duration, Video Description and Video Published data using Video IDs extracted in previous step

# importing packages
import requests
import lxml.html
from bs4 import BeautifulSoup
import pandas
import json
import urllib
import codecs
#from urllib import urlopenimport codecs
import csv


# Input your developer key
api_key = "####################" 


Videos_BannerHealth= pandas.read_csv("Youtube_CL.csv", encoding="ISO-8859-1")
Video_Ids= list(Videos_BannerHealth['v_id'])

VideoList = []
for vid in Video_Ids:
    
    videos={}    
    searchUrl="https://www.googleapis.com/youtube/v3/videos?id=" + str(vid) + "&key="+DEVELOPER_KEY+"=contentDetails"
    reader = codecs.getreader("utf-8")
    try:
        response= urllib.request.urlopen(searchUrl)
        data = json.load(reader(response))
        alldata=data['items']
        duration=alldata[0]['contentDetails']['duration']
    except:
        videos['duration'] =  " "
        
    try:
        url = "https://www.youtube.com/watch?v="+vid
        r = requests.get(url) # where url is the above url    
        bs = BeautifulSoup(r.text)
        tree1 = lxml.html.fromstring(r.content)
    except:
        videos['description'] = " "
        videos['published date'] = " "
    
    videos['duration'] = duration
    videos['description'] =[]
    videos['description'].append(tree1.xpath('//*[@id="eow-description"]/text()[1]'))
    videos['description'].append(tree1.xpath('//*[@id="eow-description"]/text()[2]'))
    videos['description'].append(tree1.xpath('//*[@id="eow-description"]/text()[3]'))

    videos['published date'] = tree1.xpath('////*[@id="watch-uploader-info"]/strong/text()')

    VideoList.append(videos)
    
    
pd_videolist = pandas.DataFrame(VideoList)
pd_youtube["duration"] = pd_videolist["duration"]
pd_youtube["description"] = pd_videolist["description"]
pd_youtube["published date"] = pd_videolist["published date"]


pd_youtube.to_csv("Youtube_Data.csv")

