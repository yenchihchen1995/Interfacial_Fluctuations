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
    data_group = []      
    labels = set(frame[:, 3])
    for value in labels:
        group = frame[np.where(frame[:, 3] == value)]
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

def pick_smoother_area(group, curvature_cutoff):
    '''
    To get the smoother area individually from each cluster
    '''
    curvature = []
    for i in range (1, len(group)-1):
        p1 = group[i-1]
        p2 = group[i]
        p3 = group[i+1]
        centre, radius = define_circle(p1, p2, p3)
        curvature.append(1/radius)
    g = []
    result = []
    for c in range(len(curvature)):
        if curvature[c] <= curvature_cutoff:
            g.append(np.array(group[c]))
        else:
            result.append(np.array(g))
            g = []
    return np.array(result)

def get_mean_interpolated_interface(group, average_number):
    '''
    Get the mean interface by averging the actual interfacial particle posisions
    
    Args:
        group = np.array(n, 2)
        average_number = the number used to average the position --> actual averaging number = size(group) / average_number
        
    '''
    
    average_number_of_interface = int(len(group) / average_number)
    half_ani = int(average_number_of_interface / 2)
    group_cp = group.copy()
    group_cp = np.vstack((group_cp, group[:half_ani]))
    mean_pos = []
    for i in range (half_ani, len(group) + half_ani):
        mean_pos.append(np.average((group_cp[i-half_ani:i+half_ani, :2]), axis = 0))
    mean_pos = np.array(mean_pos)
    mean_pos = np.vstack((mean_pos, mean_pos[0,:]))
    x = mean_pos[:,0] ; y = mean_pos[:,1]
    tck, u = interpolate.splprep([x, y], s=0)
    xnew = np.linspace(0, 1, 1000, endpoint = True)
    out = interpolate.splev(xnew, tck)
    
    return np.array(out).T

def find_projection(point, profile):
    """
    Find the index of the projection of a point
        onto a discrete profile

    Args:
        point: shape (2, )
        profile: shape (n, 2)

    Return:
        the index of the projected point
    """
    distances = cdist(point[np.newaxis, :], profile)
    projection_idx = np.argmin(distances)
    return projection_idx

def get_projection_distance(p1, p2, profile):
    """
    Get the curvilinear distance between two points
        projected onto a discrete profile

    Args:
        p1: shape (2,)
        p2: shape (2,)
        profile: shape (n, 2), it should be ordered

    Return:
        the curvilinear distance
    """
    total_length = np.linalg.norm(profile[1:] - profile[:-1], axis=1).sum()
    p1_proj_idx = find_projection(p1, profile)
    p2_proj_idx = find_projection(p2, profile)
    start = min(p1_proj_idx, p2_proj_idx)
    end = max(p1_proj_idx, p2_proj_idx)
    if start == end:
        return 0
    # print(start, end)
    shifts = profile[start + 1 : end] - profile[start : end - 1]
    distance = np.linalg.norm(shifts, axis=1).sum()
    if distance < total_length / 2:
        return distance
    else:
        return total_length - distance

def hh_corr_cluster(interface, profile, height, bins):
    """
    Height-Height Correlation calculation
    Args:
        interface: (n, 2)
        profile: (n, 2)
    """
    curvilinear_distances = []
    hh_products = []
    for i, p1 in enumerate(interface[:-1]):
        for j, p2 in enumerate(interface[i + 1:]):
            j += i
            cd = get_projection_distance(p1, p2, profile)
            curvilinear_distances.append(cd)
            hh_products.append(height[i] * height[j])
    return binned_statistic(
        curvilinear_distances, hh_products, bins=bins, statistic='mean'
        )[0]