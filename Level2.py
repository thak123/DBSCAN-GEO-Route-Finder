
# coding: utf-8

# In[1]:


import json
import networkx as nx
import numpy as np
from geopy.distance import vincenty
import googlemaps
import polyline
import sys

# In[2]:


debug_mode=False
gmaps = googlemaps.Client(key='AIzaSyDxh3HspbgJQ1ZBDfNMbJDnDo0PFymo228')


# In[3]:

u_source_lat=float(sys.argv[1]) #19.953286  
u_source_lon=float(sys.argv[2])#73.749121
u_dest_lat=float(sys.argv[3])#19.993201
u_dest_lon=float(sys.argv[4])#73.714526


#opena file to read the route-data

with open('route-data.json', 'r') as f:
    data = json.load(f)
    
with open("asset-data-compiled.json") as f:
    asset_data= json.load(f)


# In[4]:


#intialise the graph
G=nx.Graph()


# In[5]:


#create a repo of all the route start and end points
# a route is added to repo only if it is having total distance travelled greater than 5 km.
coord_list=set()
for i in data.iteritems():
    if i[1]["is_route"]:
        slat="{0:.6f}".format(i[1]["slat"])
        slon="{0:.6f}".format(i[1]["slon"])
        elat="{0:.6f}".format(i[1]["elat"])
        elon="{0:.6f}".format(i[1]["elon"])
        
        coord1=(str(slat),str(slon))
        coord2=(str(elat),str(elon))
        coord_list.add((slat,slon))
        coord_list.add((elat,elon))

        #Graph population code
        G.add_node(coord1)       
        G.add_node(coord2)
        G.add_edge(coord1,coord2,weight=i[1]["total_distace"],color='red')
        


# In[6]:


def return_closest_point(p_lat,p_lon):
    distance=0.0
    min_lat=None
    min_lon=None
    min_index=-1
    coord=(p_lat,p_lon)
    for index,i in enumerate(coord_list):
        if index == 0:
            distance =vincenty(coord,i).meters
            min_index=index
            min_lat=i[0]
            min_lon=i[1]

        else:
        
            new_distance =vincenty(coord,i).meters
            
            if new_distance < distance :
                distance= new_distance
                min_index=index
                min_lat=i[0]
                min_lon=i[1]
    if debug_mode: print distance,distance>1000
    return min_lat,min_lon, distance>1000
            


# In[7]:


def return_avg_time(first,second):
    travel_distance=[]
    #loop through the data keys
    for asset in asset_data.itervalues():
        for record in asset:
            value= record.keys()[0]
            if first[0].rstrip("0")+'-'+first[1].rstrip("0") in value.split("!")            and second[0].rstrip("0")+'-'+second[1].rstrip("0") in value.split("!"):
                travel_distance.append(record.values()[0])
        
    return np.mean(travel_distance)/3600

# return_avg_time(('11.144053','75.962166'),('11.266046','75.766821'))


# In[8]:


while(nx.is_connected(G)==False):
    cg =  list(nx.connected_components(G))

    for i_member in cg:
        best_score=99999999999
        best_neighbour=None

        best_host=None
        for j_member in  cg:
            for i in i_member:
                for j in j_member:
                    if i[0]!=j[0] and i[1]!=j[1] and j not in i_member:
                        if vincenty(i,j).meters < best_score:
                            best_neighbour=j
                            best_host=i
                            best_score=vincenty(i,j).meters
        if debug_mode:
            print best_neighbour,best_score,best_host,G.has_edge(best_neighbour,best_host)
            print G.has_node(best_neighbour),G.has_node(best_host),nx.has_path(G,best_neighbour,best_host)
        if G.has_edge(best_neighbour,best_host)==False:
            G.add_edge(best_neighbour,best_host,weight=best_score,pseudo=True,color='red')
        if debug_mode:
            print best_neighbour,best_score,best_host,G.has_edge(best_neighbour,best_host)
#     break


# In[9]:


#goa to kochi both nodes present as route
# u_source_lat=15.505601
# u_source_lon=73.833911

# u_dest_lat=11.265776
# u_dest_lon=75.766602




# In[10]:


#some where near kochi to kochi both present
# u_source_lat=11.265662
# u_source_lon=75.766618

# u_dest_lat=11.265776
# u_dest_lon=75.766602


# In[11]:


#manglore to kochi both present
# u_source_lat=12.868182
# u_source_lon=74.931608

# u_dest_lat=11.265776
# u_dest_lon=75.766602


# In[12]:


# u_source_lat=15.167324 
# u_source_lon= 74.010327
# u_dest_lat=19.993201
# u_dest_lon=73.714526
  
#  11.143636' '75.962817
 
# u_source_lat=   19.949103
# u_source_lon= 73.729574
# u_dest_lat=15.505514
# u_dest_lon=73.834993


    
    


# In[13]:


# take source/dest point and see if they exists
source_exists=False
dest_exists=False

#check for source
if G.has_node((str(u_source_lat),str(u_source_lon))):
    source_exists=True

#check for destination

if G.has_node((str(u_dest_lat),str(u_dest_lon))):
    dest_exists=True

print source_exists,dest_exists


# In[14]:


path=[]
path_distance=0.0

#array for joining two points
decoded_points=[]
decoded_duration=0
decoded_distance=0.0

new_u_dest_lat=0.0
new_u_dest_lon=0.0
new_u_source_lat=0.0
new_u_source_lon=0.0

is_processing_required=False

if source_exists and dest_exists: 
    path= (nx.dijkstra_path(G,source=(str(u_source_lat),str(u_source_lon)),                                     target=(str(u_dest_lat),str(u_dest_lon)),weight="weight"))
    if debug_mode: 
        print path
    path_distance= nx.dijkstra_path_length(G,source=(str(u_source_lat),str(u_source_lon))                                           ,target=(str(u_dest_lat),str(u_dest_lon)))
    
    if debug_mode :
        print [p for p in nx.all_shortest_paths(G,source=(str(u_source_lat),str(u_source_lon))                                            ,target=(str(u_dest_lat),str(u_dest_lon)),weight="weight")]  
        # for f,k in path[0].iteritems():
        print nx.is_connected(G)
    #     print  "{},{}".format(f,k)
        for f,k in path:
            print "{},{}".format(f,k)  
    #TODO
else:
 
    if source_exists==False and dest_exists==True:

        new_u_source_lat,new_u_source_lon,is_processing_required=return_closest_point(u_source_lat,u_source_lon)
        new_u_dest_lat,new_u_dest_lon=u_dest_lat,u_dest_lon
        
        if is_processing_required:
            #get the distance from user_source to the new found source
            
            directions_result = gmaps.directions((new_u_source_lat, new_u_source_lon),                                     (u_source_lat, u_source_lon),                                     mode="driving")
            directions_matrix_result = gmaps.distance_matrix([(new_u_source_lat, new_u_source_lon)],                                     [(u_source_lat, u_source_lon)],                                     mode="driving")
            
            if len(directions_result):
                decoded_points=polyline.decode(directions_result[0]['overview_polyline']['points'])
                #distance is meter
                #time in seconds
                decoded_distance =directions_matrix_result["rows"][0]["elements"][0]["distance"]["value"]
                decoded_duration= directions_matrix_result["rows"][0]["elements"][0]["duration"]["value"]
            else:
                print "google returned 0 result"
            

    elif source_exists==True and dest_exists==False :

        new_u_source_lat,new_u_source_lon=u_source_lat,u_source_lon
        new_u_dest_lat,new_u_dest_lon,is_processing_required=return_closest_point(u_dest_lat,u_dest_lon)
        
        if is_processing_required:
        #get the distance from user_source to the new found source

            directions_result = gmaps.directions((new_u_dest_lat, new_u_dest_lon),                                     (u_dest_lat, u_dest_lon),                                     mode="driving")
            directions_matrix_result = gmaps.distance_matrix([(new_u_dest_lat, new_u_dest_lon)],                                     [(u_dest_lat, u_dest_lon)],                                     mode="driving")

            if len(directions_result):
                decoded_points=polyline.decode(directions_result[0]['overview_polyline']['points'])
                #distance is meter
                #time in seconds
                decoded_distance =directions_matrix_result["rows"][0]["elements"][0]["distance"]["value"]
                decoded_duration= directions_matrix_result["rows"][0]["elements"][0]["duration"]["value"]

            else:
                print "google returned 0 result"
                
    elif source_exists==False and dest_exists==False:

        new_u_source_lat,new_u_source_lon,is_processing_required=return_closest_point(u_source_lat,u_source_lon)                            
        new_u_dest_lat,new_u_dest_lon,is_processing_required=return_closest_point(u_dest_lat,u_dest_lon)
        
        if is_processing_required:
            #source part
            directions_result = gmaps.directions((new_u_source_lat, new_u_source_lon),                                     (u_source_lat, u_source_lon),                                     mode="driving")
            directions_matrix_result = gmaps.distance_matrix([(new_u_source_lat, new_u_source_lon)],                                     [(u_source_lat, u_source_lon)],                                     mode="driving")

            if len(directions_result):
                decoded_points=polyline.decode(directions_result[0]['overview_polyline']['points'])
                #distance is meter
                #time in seconds
                decoded_distance =directions_matrix_result["rows"][0]["elements"][0]["distance"]["value"]
                decoded_duration= directions_matrix_result["rows"][0]["elements"][0]["duration"]["value"]

            else:
                print "google returned 0 result"
            #destination part
            directions_result = gmaps.directions((new_u_dest_lat, new_u_dest_lon),                                     (u_dest_lat, u_dest_lon),                                     mode="driving")
            directions_matrix_result = gmaps.distance_matrix([(new_u_dest_lat, new_u_dest_lon)],                                     [(u_dest_lat, u_dest_lon)],                                     mode="driving")

            if len(directions_result):
                decoded_points.extend(polyline.decode(directions_result[0]['overview_polyline']['points']))
                #distance is meter
                #time in seconds
                decoded_distance +=directions_matrix_result["rows"][0]["elements"][0]["distance"]["value"]
                decoded_duration += directions_matrix_result["rows"][0]["elements"][0]["duration"]["value"]

            else:
                print "google returned 0 result"
            
        
        
        
    
    path=nx.dijkstra_path(G,source=(str(new_u_source_lat),str(new_u_source_lon))                                         ,target=(str(new_u_dest_lat),str(new_u_dest_lon)),weight="weight")
    path_distance=nx.dijkstra_path_length(G,source=(str(new_u_source_lat),str(new_u_source_lon))                                                        ,target=(str(new_u_dest_lat),str(new_u_dest_lon)))
    


# In[15]:


#calculate time calculations
#calculate ex
# Route : 10% internal 90% external. 
# Total kms : xx 
# Total time for travel: xx
internal=0.0
external=0.0
total_time=0.0
for first,second in zip(path, path[1:]):
    if "pseudo" in G.get_edge_data(first,second):
        external+= G.get_edge_data(first,second)["weight"]
    else:
        internal+=G.get_edge_data(first,second)["weight"]
        if debug_mode: 
            print return_avg_time(first,second),first,second
        total_time +=return_avg_time(first,second)
        
if is_processing_required:
    external+=decode_distance
    path_distance+=decode_distance
    total_time+=(decoded_duration/3600)
    
if debug_mode: 
    print is_processing_required
print "Route : {} internal {} external".format(round((internal*100)/(path_distance),2),round((external*100)/path_distance,2))
print "Total kms :", path_distance/1000
print "time for travel in kms :", total_time


# In[16]:



#handle the looping through the 
map_shortest_path_nodes=[]

#check if new coordinates needs to be added to map points
if is_processing_required:
     map_shortest_path_nodes.extend(decoded_points)

for first, second in zip(path, path[1:]):
    #loop through the data keys
    for key in data.keys():
        if first[0]+'-'+first[1] in key.split("!") and second[0]+'-'+second[1] in key.split("!"):
            map_shortest_path_nodes.extend(list(set(tuple(round(a,6) for a in  p) for p in data[key]["path"])))
            


# In[17]:


#This is the last part
#code to add data points to the shortest_path_nodes
if source_exists==False and dest_exists==True:
    map_shortest_path_nodes.insert(0,(u_source_lat,u_source_lon))
    
elif source_exists==True and dest_exists==False :
    map_shortest_path_nodes.insert(len(map_shortest_path_nodes),(u_dest_lat,u_dest_lon))

elif source_exists==False and dest_exists==False:
    map_shortest_path_nodes.insert(0,(u_source_lat,u_source_lon))
    map_shortest_path_nodes.insert(len(map_shortest_path_nodes),(u_dest_lat,u_dest_lon))


# In[18]:


map_shortest_path_nodes=np.matrix(map_shortest_path_nodes)


# In[19]:


#code for mapping
import gmplot
from rdp import rdp

epsilon=0.15
algo='iter' 


if len(map_shortest_path_nodes) < 1:
    raise ValueError('A very specific bad thing happened')

new_locations = []


after_rdp = rdp(map_shortest_path_nodes, epsilon=epsilon, algo=algo, return_mask=True)

if debug_mode:
    print after_rdp
new_locations =map_shortest_path_nodes[after_rdp]



#todo add initial points if they were not present in the shortest path node
#display distance and time for the trip

x,y=zip(*np.array(new_locations).astype(np.float))
centerx,centery=float(max(list(x))+min(list(x)))/2.0, float(max(list(y))+min(list(y)))/2.0
gmap = gmplot.GoogleMapPlotter(centerx,centery, 13)

gmap.scatter(x, y, edge_color="cyan", edge_width=5, face_color="blue", face_alpha=0.1)

gmap.draw("mymap.html")

# print("Total Distance:{} ".format(shortest_path_distance))
    


# In[ ]:




