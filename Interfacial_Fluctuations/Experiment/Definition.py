import numpy as np
import matplotlib.pyplot as plt
import freud
import pickle
from scipy.ndimage import uniform_filter1d
from scipy.stats import binned_statistic
from scipy import interpolate
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import pdist, cdist
import re

def pause():
    wait = input("PRESS ENTER TO CONTINUE")
    return wait

def get_frames_from_xyz(filename, ncols = 3, skipcol = 0):
    '''
    This is how you get the data from output xyz file.
    ncols = how many columns you would like to obtain.
    skipcol = how many columns you would like to skip.
    '''
    f = open(filename, 'r')
    frames = []
    for line in f:
        is_head = re.match(r'(\d+)\n', line)
        if is_head:
            frames.append([])
            particle_num = int(is_head.group(1))
            f.readline() # jump through comment line
            for j in range(particle_num):
                data = re.split(r'\s', f.readline())[skipcol:skipcol+ncols]
                frames[-1].append(list(map(float, data)))
    f.close()
    return np.array(frames)

def get_box_size(data):
    '''
    To get the box size of experimental data.
    data = the data obtained from xyz file (should be numpy array).
    '''
    single_frame = np.array(data[0])
    pos = single_frame[:, :2]
    box_size = [np.ceil(np.max(pos[:,0])), np.ceil(np.max(pos[:,1]))]
    return box_size

def pick_clusters(frame, box_size, cutoff_cluster_size):
    '''
    this is the function of how to get the clusters without 
    being cut by the edges and upper specific number of particles
    Args:
        frames : the single from xyz file
        box_size : the size of boundary 
        cutoff_cluster_size : the cutoff of number of particles (upper)
    Return:
        The data which are sorted by the conditions (np.array)
    '''
    delta = 5
    max_group = max(frame[:,3])
    data_group = []      
    for i in range (1,int(max_group)):
        group = []
        for j in range (len(frame)):
            if int(frame[j][3]) == i:
                group.append(frame[j])
        group = np.array(group)
        edges = [[0, 0], [box_size[0], box_size[1]]]
        near_edge = False
        for edge_xy in edges:
             for dim, edge in enumerate(edge_xy):
                near_edge = near_edge or (np.abs(group[:, dim] - edge).min() < delta)
        if len(group) >= cutoff_cluster_size and (not near_edge):
            data_group.append(np.array(group))
    return np.array(data_group)

def get_interface_from_voronoi(points, box, threshold):
    '''
    Use the voronoi volume to get the interface particles
    Args:
        points (np.ndarray): shape (n, 3), the position of z axis
            should all be zero for freud
        box (np.ndarray): the box size of x and y axis, shape (2,)
        threshold (double): if the volume is above threshold value
        the particle is the interface
    Return:
        np.ndarray: the points corresponding to the interface, may contain bubbles
    '''
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
    voro = freud.locality.Voronoi()
    box_obj = freud.box.Box.from_box(box)
    box_obj.periodic_x = True
    box_obj.periodic_y = True
    voro.compute((box_obj, points))
    vv = voro.volumes
    interface = points[vv > threshold, :2]
    return interface

def comDist(x,y):
    
    xarr=np.array(x)
    yarr=np.array(y)
    zarr=xarr-yarr
    return np.sqrt(sum(np.power(zarr,2)))

def orderPoints(all_points, cutoff):
     
    points_series_dict={}            
    start_point=all_points[0]       
    istep=0                          
    points_series_dict[istep]=[start_point,0]  
    current_index = 0                 

    while len(all_points)>1:    
        istep+=1
        current_point = all_points[current_index].copy()
        all_points = np.delete(all_points, current_index, axis=0)

        dist_list = cdist(current_point[np.newaxis, :], all_points).ravel()
        min_dist = dist_list.min()
        min_dist_idx = dist_list.argmin()     
        if min_dist > cutoff:
            break 

        next_point = all_points[min_dist_idx]
        points_series_dict[istep] = [next_point,min_dist] 

        if istep>5:
            currentPoint_startPoint_dist=comDist(current_point,start_point) 

            if min_dist>currentPoint_startPoint_dist:                         
                break
        current_index = min_dist_idx
    return points_series_dict,all_points

def sort_particles(interface, cutoff):
    '''cutoff = max distance to connect'''
    order = np.argsort(interface[:,1])
    interfaces = interface[order]
    #all_points = interfaces.tolist()
    all_points = interfaces.copy()
    points_series_dict, remain_points = orderPoints(all_points, cutoff)
    ordered_list = []
    for k,v in points_series_dict.items():
        ordered_list.append(v[0])
    ordered_array = np.array(ordered_list)
    return ordered_array

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    obtained from: https://stackoverflow.com/a/50974391
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def get_smoother_interfaces(interface, curvature_cutoff):
    '''
    To get the interface with lower averaged curvature.
    Args:
        interfaces: the interface sorted by the order
        curvature_cutoff: average curvature cutoff
    Return:
        smoother interface (np.array)
    '''
    curvature = []
    for i in range(1, len(interface)-1):
        p1 = interface[i-1]
        p2 = interface[i]
        p3 = interface[i+1]
        centre, radius = define_circle(p1, p2, p3)
        curvature.append(1/radius)
    if np.mean(curvature) <= curvature_cutoff:
        return interface
    else:
        return np.array([0, 0])