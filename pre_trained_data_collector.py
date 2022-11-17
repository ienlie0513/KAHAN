import os
import requests
import gensim.downloader as api

# create a directory to store the data
dir_path = os.getcwd() + '/word2vec'
os.makedirs(dir_path, exist_ok=True)

# download the .plk file from the link (This takes a while to download!)
url = 'https://huggingface.co/jinmang2/dooly-hub/resolve/0fcf5b24ba3748253300579ecaac0de546aec668/word_embedding/en/wikipedia2vec.en/enwiki_20180420_100d.pkl'
r = requests.get(url, allow_redirects=True)
open(dir_path + '/enwiki_20180420_100d.pkl', 'wb').write(r.content)

twitter_data_url = api.load("glove-twitter-100", return_path=True)
# unzip the gz file
os.system('gunzip ' + twitter_data_url)
os.rename(twitter_data_url.split('.')[0], dir_path + '/glove-twitter-100')

wiki_data_url = api.load("glove-wiki-gigaword-100", return_path=True)
# unzip the gz file
os.system('gunzip ' + wiki_data_url)
os.rename(wiki_data_url.split('.')[0], dir_path + '/glove-wiki-gigaword-100')