#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sknetwork
from IPython.display import SVG
from sknetwork.visualization import svg_graph, svg_bigraph
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import string
import random

from sknetwork.utils import get_neighbors
from sknetwork.ranking import PageRank, top_k, HITS
from sknetwork.clustering import Louvain, get_modularity, PropagationClustering


# In[2]:


wikivitals = sknetwork.data.load_netset("wikivitals") # load dataset


# In[3]:


print(wikivitals.keys())


# In[4]:


adjacency = wikivitals.adjacency
names = wikivitals.names
labels = wikivitals.labels
names_labels = wikivitals.names_labels
biadjacency = wikivitals.biadjacency
print(wikivitals.keys())
label_to_id = {name: i for i, name in enumerate(names_labels)}
names[777]


# In[5]:


print(len(names))
print(np.unique(labels, return_counts=True))


# In[6]:


pagerank = PageRank()
n_selection = 50

selection = []
for label in np.arange(len(names_labels)): # get top page ranks for each given category
    ppr = pagerank.fit_predict(adjacency, seeds=(labels==label))
    scores = ppr * (labels==label)
    selection.append(top_k(scores, n_selection))
selection = np.array(selection)

for label, name_label in enumerate(names_labels): # print the top five page rank articles for each category
    print('---')
    print(label, name_label)
    print(names[selection[label, :5]])


# In[7]:


names = list(names)
type(names)


# In[8]:


scores = pagerank.fit_predict(adjacency)
scores.shape


# In[9]:


hits = HITS()
hubs = hits.fit_predict(biadjacency)


# In[10]:


scores = [scores]
scores.append(hubs)
scores = np.array(scores).T


# In[39]:


#### import requests # make wikipedia api request
import tabulate # display output in pretty table
from IPython.core.display import display, HTML
from bs4 import BeautifulSoup


def format_link(title):
    # Hipparcus -> https://en.wikipedia.org/wiki/Hipparchus
    title = title.replace(" ", "_")
    return 'https://en.wikipedia.org/wiki/' + title

def get_table(candidate_scores):
    
    # candidate_scores: {title: (pagerank, hubs, incoming, outgoing)}
    index_to_name = {-1: -1}
    headers = ["Index", "Title", "Link", "Incoming Links", "Outgoing Links","PageRank", "Hubs"]
    table = []
    for i,title in enumerate(candidate_scores):
        row = []
        row.append(i)
        row.append(title)
        row.append(format_link(title))
        row.append(candidate_scores[title][2])
        row.append(candidate_scores[title][3])
        row.append(candidate_scores[title][0])
        row.append(candidate_scores[title][1])
        table.append(row)
        index_to_name[i] = title
        
    return tabulate.tabulate(table, tablefmt='html', headers=headers , showindex=False), index_to_name
        

def search(query, scores, adjacency):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
                    'action':'query',
                    'format':'json',
                    'list':'search',
                    'utf8':1,
                    'srsearch':query,
                    'srlimit': 500,
                    'srqiprofile': 'empty' # dont want assistance from wikipedia search engine
                }
    params['srsearch'] = query
    data = requests.get(url, params=params).json()

    candidates = []
    for article in data['query']['search']:
        if article['title'] in names:
            candidates.append(article['title'])
    
    # row outgoing, col incoming
    candidate_metrics = {} # {name: (pagerank, hubs , incoming, outgooing)}
    for candidate in candidates:
        i = names.index(candidate)
        pagerank, hubs = scores[i][0], scores[i][1]
        outgoing = np.sum(adjacency[i]) # row
        incoming = np.sum(adjacency[:,i]) # col
        candidate_metrics[candidate] = (pagerank, hubs, incoming, outgoing) 
        
    candidate_metrics = dict(sorted(candidate_metrics.items(), key=lambda item: item[1][0], reverse=True)) # Put scores in descending order

    return get_table(candidate_metrics)

def query():
    query = input("Enter your Search Query:")
    table, index_to_name = search(query, scores, adjacency)
    display(table)
    option = int(input("Enter Index to get more information about an article or '-1' to enter another query:"))
    
    while option not in index_to_name.keys():
        option = int(input("Invalid Option: Enter Index to get more information about an article or '-1' to enter another query:"))
        
    return index_to_name[option]
        

def showArticle(name):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
                'action': 'parse',
                'page': name,
                'format': 'json',
                'prop':'text',
                'redirects':''
            }

    response = requests.get(url, params=params)
    data = response.json()

    raw_html = data['parse']['text']['*']
    soup = BeautifulSoup(raw_html,'html.parser')
    soup.find_all('p')
    text = ''

    for p in soup.find_all('p'):
        text += p.text

    print(text)
    
def menu():
    print("What would you like to do?")
    print("1. View Full Article")
    print("2. See Related Articles")
    print("3. See Popular Articles within the Same Category")
    print("4. See All Incoming Articles")
    print("5. See All Outgoing Articles")
    option = int(input("Choose Here"))
    return option

# outgoing = np.sum(adjacency[i]) # row
# incoming = np.sum(adjacency[:,i]) # col

def show_incoming(name):
    article_id = names.index(name)
    col = adjacency[:,article_id].toarray()[0]
    incoming_ids = np.where(col == True)[0]
    
        # row outgoing, col incoming
    metrics = {} # {name: (pagerank, hubs , incoming, outgooing)}
    for i in incoming_ids:
        name = names[i]
        pagerank, hubs = scores[i][0], scores[i][1]
        outgoing = np.sum(adjacency[i]) # row
        incoming = np.sum(adjacency[:,i]) # col
        metrics[name] = (pagerank, hubs, incoming, outgoing) 
        
    metrics = dict(sorted(metrics.items(), key=lambda item: item[1][0], reverse=True)) # Put scores in descending order
    table, index_to_name = get_table(metrics)
    display(table)
    return index_to_name
    
    
    



def show_outgoing(name):
    article_id = names.index(name)
    row = adjacency[article_id].toarray()[0]
    outgoing_ids = np.where(row == True)[0]
    metrics = {} # {name: (pagerank, hubs , incoming, outgooing)}
    for i in outgoing_ids:
        name = names[i]
        pagerank, hubs = scores[i][0], scores[i][1]
        outgoing = np.sum(adjacency[i]) # row
        incoming = np.sum(adjacency[:,i]) # col
        metrics[name] = (pagerank, hubs, incoming, outgoing) 
        
    metrics = dict(sorted(metrics.items(), key=lambda item: item[1][0], reverse=True)) # Put scores in descending order
    table, index_to_name = get_table(metrics)
    display(table)
    return index_to_name
    
    
def main():
    name = query()
    
    while True:
        if name == -1:
            query()
        else:
            menu_choice = menu()
            if menu_choice == 1:
                showArticle(name)
                
            elif menu_choice == 2:
                pass
            
            elif menu_choice == 3:
                pass
            
            elif menu_choice == 4:
                show_incoming(name)
                
            elif menu_choice == 5:
                show_outgoing(name)
        return
            
    
main()


# In[36]:


resolutions = np.linspace(0.1, 3, 30)

modularities = ['Dugue', 'Newman', 'Potts']
best = {"mod": None, "res": None, "score":0}

dugue = []
newman = []
potts = []

for mod in modularities:
    for res in resolutions:
        
        lv = Louvain(resolution=res, modularity=mod)
        lv.fit(adjacency)
        lv_labels = np.array(lv.labels_)
        lv_score = get_modularity(adjacency, lv_labels)
        
        if lv_score > best["score"]:
            best["mod"]=mod
            best["res"]=res
            best["score"]=lv_score
            
        if mod == "Dugue":
            dugue.append(lv_score)
        elif mod == "Newman":
            newman.append(lv_score)
        elif mod == "Potts":
            potts.append(lv_score)

            
            
best


# In[15]:


best_louv = Louvain(resolution=best['res'], modularity=best["mod"])
best_louv.fit(adjacency)
np.unique(best_louv.labels_)


# In[37]:


labels = best_louv.labels_
clusters = []
for label in np.unique(labels):
    cluster_indices = np.where(labels==label)[0]
    indice_to_score = {i: (scores[i][0], scores[i][1]) for i in cluster_indices}
    clusters.append(indice_to_score)
    
for cluster in clusters:
    
    sorted_cluster = dict(sorted(cluster.items(), key=lambda x: x[1][0], reverse=True))
    sorted_indices = list(sorted_cluster.keys())
    for i in sorted_indices[:10]:
        print(names[i] , scores[i])
    print("------------")
        
    


# In[19]:


import matplotlib.pyplot as plt

best={'mod': 'Dugue', 'res': 1.0999999999999999, 'score': 0.4721801809672379}

# plot the lines
plt.plot(resolutions, dugue, label='Dugue')
plt.plot(resolutions, potts, label='Potts')
plt.plot(resolutions, newman, label='Newman')


plt.legend()


plt.xlabel('Resolution')
plt.ylabel('Modularity Score')
plt.title('Parameter Tuning of Louvain Clustering')


plt.savefig('louvain.png')
plt.show()


# In[38]:


G = nx.DiGraph(adjacency)
G = G.to_undirected()


# In[ ]:




