
# coding: utf-8

# In[1]:


'''
Input: dataset in form of csv
Output: Two JSON files. One containing routes information 
        Second file contains the details of the asset
'''


# In[2]:


#imports
import datetime
import math
import json

import geopy.distance
import pandas as pd
import numpy as np


from sklearn.cluster import DBSCAN
from sklearn import metrics

import matplotlib.pyplot as plt

import matplotlib.cm as cmx
import matplotlib.colors as colors


# from rdp import rdp
from hausdorff import hausdorff 

import sys


# In[3]:


'''
TODO: 
    Need to find unique coordinates for the path.
    Handle stationary condition
    May be use RDP later on
'''

fileName="RouteDataset"


# Helpers for data clustering

# define a helper function to get the colors for different clusters

def get_cmap(N):
    '''
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.
    '''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


# In[5]:


debug_mode= False


# In[6]:


#Code to initialise the empty files
#Initialise empty json 
with open('route-data.json', mode='w') as f:
    json.dump({}, f)
    
with open('asset-data.json', mode='w') as f:
    json.dump({}, f)


# read input file 

asset_ids = pd.read_csv(fileName)["fk_asset_id"].unique()
        


# In[7]:


#  for every asset in the dataset loop over
for asset_id in asset_ids:# [177,    249]:#asset_ids:# [177,    249,    518,    142]: # asset_ids:#
    print("processing assetid {}".format(asset_id))
#     continue
#     continue
    df = pd.read_csv(fileName).query("fk_asset_id == {}".format(asset_id))        .sort_values('tis')
        
    # Get the latitude and logitude of the earthquakes
    coords = df.as_matrix(columns=['lat', 'lon'])
    time_array=df.as_matrix(columns=['tis'])
    
    # reset the index
    df=df.reset_index(drop=True)
    
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    minimum_samples = 1000

    # Run the DBSCAN in order to obtain the stopping points
    db = DBSCAN(eps=epsilon, min_samples=minimum_samples, algorithm='ball_tree',                 metric='haversine').fit((coords))

    cluster_labels = db.labels_
    n_clusters = len(set(cluster_labels))
    print n_clusters
    
    # get the cluster
    clusters =         pd.Series([coords[cluster_labels == n] for n in range(-1, n_clusters)])

    # cluster_labels = -1 means outliers
    print('Number of clusters: {}'.format(n_clusters - 1))
    
    
    if debug_mode:
        fig, ax = plt.subplots(figsize=[10, 6])

        unique_label = np.unique(cluster_labels)

        # get different color for different cluster
        cmaps = get_cmap(n_clusters)

        # plot different clusters on map, note that the black dots are 
        # outliers that not belone to any cluster. 
        for i, cluster in enumerate(clusters):
            lons_select = cluster[:, 1]
            lats_select = cluster[:, 0]
            x, y = (lons_select, lats_select)
            ax.scatter(x,y,5,marker='o',color=cmaps(i), zorder = 10)


        plt.show()
    
    #code to set the route number to coordinates reaching from start point to the destination.
    route_info=[]
    start_route_finding=False
    current_label=-1
    route_no=1
    
    for index, row in df.iterrows():
        if  cluster_labels[index] in range(0, n_clusters) and start_route_finding== False:
            start_route_finding=True
            current_label=cluster_labels[index] 
        elif start_route_finding== False:
            route_info.append(-1)

        if start_route_finding and cluster_labels[index] in [current_label,-1] :
            route_info.append(route_no)
        elif start_route_finding==True:
            current_label=cluster_labels[index] 
            route_no=route_no+1
            route_info.append(route_no)

    if debug_mode:
        print (route_info)
    
    df["route_info"]=route_info
    
    #code to get unique 
    parent_route=None
    route_name="route"
    is_parent_route_set=False
    
    for index,route_no in enumerate(df["route_info"].unique()):
        if route_no == -1:
            continue
        elif is_parent_route_set==False:
            parent_route= np.ascontiguousarray(df[df['route_info'] ==route_no].as_matrix(columns=['lat', 'lon']))
            is_parent_route_set=True
                  
        elif is_parent_route_set==True:
            current_route = np.ascontiguousarray(df[df['route_info'] == route_no].as_matrix(columns=['lat', 'lon']))
            if current_route.shape[0]==1:
                continue
            sim_val = hausdorff(parent_route,current_route)
#             print sim_val
            if  sim_val >0.01:
                if debug_mode:
       
                    print route_no,sim_val,                                df[df['route_info'] == route_no].iloc[0,1:3],                                sim_val,df[df['route_info'] == route_no].iloc[-1,1:3]
                    print current_route[0],current_route[-1]
                    print "\n"

                #perform calculation of distance travelled
                current_route_total_distance=0.0
                for first, second in zip(current_route, current_route[1:]):
                    current_route_total_distance+= geopy.distance.vincenty(first, second).meters
                #TODO THink some better name for the  routes
                route_data_point={
                    'r{}!{}-{}!{}-{}'.format(index,\
                                          current_route[0][0],\
                                          current_route[0][1],\
                                          current_route[-1][0],\
                                          current_route[-1][1]):{
                    "slat":current_route[0][0],
                    "slon":current_route[0][1],
                    "elat":current_route[-1][0],
                    "elon":current_route[-1][1],
                    "path":current_route.tolist(),
                    "total_distace":current_route_total_distance,
                    "is_route":current_route_total_distance > 5000 #need to make global for hardcoded values 
                    }
                }
    #             print route_data_point


                asset_id = df[df['route_info'] == route_no].iloc[0,0] #'fk_asset_id' take the asset_id
                #we need the time details
                min_time = df[df['route_info'] == route_no]['tis'].min()
                max_time = df[df['route_info'] == route_no]['tis'].max()

                avg_speed = df[df['route_info'] == route_no]['spd'].mean()
#                 print asset_id, max_time-min_time,avg_speed,"hi"

                #add the time take for the current track into the json file. 
                #this file contains for every asset the time taken by asset on a particular route.
                asset_data_point={
                    "{}!{}".format(asset_id,index):{
                        'r{}!{}-{}!{}-{}!{}'.format(index,\
                                          current_route[0][0],\
                                          current_route[0][1],\
                                          current_route[-1][0],\
                                          current_route[-1][1],\
                                          avg_speed):max_time-min_time
                    }
                }
                # Reading data back
                with open('route-data.json', 'r') as f:
                    data = json.load(f)
                    data.update(route_data_point)

                    # Writing JSON data
                    with open('route-data.json', 'w') as f1:
                        json.dump(data,f1)

                  # Reading data back
                with open('asset-data.json', 'r') as f:
                    data = json.load(f)
                    data.update(asset_data_point)

                    # Writing JSON data
                    with open('asset-data.json', 'w') as f1:
                        json.dump(data,f1)
                


# In[9]:


with open('asset-data.json', 'r') as f:
    data = json.load(f)

    asset_data={}
    for i in data.keys():
        asset_id= i.split("!")[0] 

        if asset_id in asset_data:
            asset_data[asset_id].append(data[i])
        else:
            asset_data[asset_id]=[data[i]]
            
     # Writing JSON data
    with open('asset-data-compiled.json', 'w') as f1:
        json.dump(asset_data,f1)
    


# In[ ]:




