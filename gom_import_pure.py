import os
import numpy as np
from xml.dom import minidom
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

"""
Reads GOM data with reference points.

Input
- map_3d_dir (string): path to xml file containing the sample marker positions 

Output
- map_3d (np): numpy array containing the marker positions; shape=(*, 3) 

"""

# load xml file 
dirpath = '/home/cherb/LRZSync/Doktorarbeit/Anderes/Markos_Mirror_Profile/data/2p0mm'
#ap_3d_dir = os.path.join(dirpath, "Referenzpunkte.refxml")
data_loc = "Reference Points_2.0mm.refxml"
map_3d_dir = os.path.join(dirpath, data_loc)
with open(map_3d_dir,'r') as map_3d_xml_file: 
    map_3d_xml = minidom.parse(map_3d_xml_file)

    # check xml format 
    if(map_3d_dir[-6:] == "refxml"): 
        # refxml format 
        # get all points from xml  
        points = map_3d_xml.getElementsByTagName('point') 

        # initialize result array 
        map_3d = np.zeros(shape=(points.length, 3)) 

        # loop through all points and save their coordinates 
        for p in range(points.length): 
            map_3d[p,0] = float( points[p].getElementsByTagName('coordinates')[0].getElementsByTagName("x")[0].firstChild.nodeValue ) 
            map_3d[p,1] = float( points[p].getElementsByTagName('coordinates')[0].getElementsByTagName("y")[0].firstChild.nodeValue )
            map_3d[p,2] = float( points[p].getElementsByTagName('coordinates')[0].getElementsByTagName("z")[0].firstChild.nodeValue ) 
        # only get points with positive z-value
        map_3d = map_3d[ map_3d[:,2]>=5 ]
    else: 
        # xml format 
        # get all points from xml  
        points = map_3d_xml.getElementsByTagName('point') 

        # initialize result array 
        map_3d = np.zeros(shape=(points.length, 3)) 

        # loop through all points and save their coordinates 
        for p in range(points.length): 
            map_3d[p,0] = points[p].attributes["x"].value 
            map_3d[p,1] = points[p].attributes["y"].value
            map_3d[p,2] = points[p].attributes["z"].value 

    # # visualize points 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(map_3d[:,0], map_3d[:,1], map_3d[:,2]) 
    plt.show() 

    # # visualize points 
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(map_3d[:,1], map_3d[:,2]) 
    plt.show() 


    # close xml file 
    map_3d_xml_file.close() 
    # Save in npy file for further evaluation
    save_loc = os.path.join(dirpath, data_loc[:-6]+'npy')
    np.savetxt(save_loc, map_3d)
    #return map_3d
