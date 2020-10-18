import numpy as np
import matplotlib.pyplot as plt
import freud
from variables import *
from Definition import *
from scipy.ndimage import uniform_filter1d
from scipy.stats import binned_statistic
from scipy import interpolate
from shapely.geometry import Point, Polygon
import pickle
from scipy.spatial.distance import pdist, cdist



def read_dump(filer, min_step=0, max_step=1e10):
    '''
    Reads dump files from LAMMPS and return data sorted by snap
    
    filer: Input file
    min_step: default 0
    max_step: maximum number of steps to read
    
    returns
        features: Dict sorting data by snap, e.g. features[snap_number] = N x 5 matrix
        
        Each column corresponds to:
        
        | x | y | particle_id | time_step | psi_6 |
    '''
    data=[]
    snapshot = {}
    features = {}
#     snapshot['traj'] = filer
    minstep_flag = False
    read_flag = False
    if (max_step<min_step) : min_step, max_step = max_step, min_step
    cnt = 0
    with open(filer,'r') as f:
        while True:
            line=f.readline().strip('\n')
            if not line: break
            items = line.split()
            if items[0] == 'ITEM:':
#                 try:
                if items[1] == 'TIMESTEP':
                    step = int(f.readline().split(' ')[0])
                    if step > max_step :
#                         print ('max_step reached (%d)' % max_step)
#                         print ('From %s, last TIMESTEP %d' % (filer, data[-1]['step']))
                        return features
                    if ( step >= min_step and minstep_flag == False) :
#                         print 'From %s, first TIMESTEP reached (%d)' % (filer, min_step)
                        minstep_flag = True
                    if (step >= min_step and step <= max_step):
                        read_flag = True
                        cnt += 1
#                         print '%d : reading TIMESTEP %d' % (cnt,step)
                    snapshot['step'] =  step
                        
                if items[1] == 'NUMBER':
                    N = int(f.readline().split()[0])
                    snapshot['N'] =  N
                if items[1] == 'BOX':
                    line = f.readline().split()
                    box_x = float(line[1]) - float(line[0])
                    line = f.readline().split()
                    box_y = float(line[1]) - float(line[0])
                    line = f.readline().split()
                    box_z = float(line[1]) - float(line[0])
                    snapshot['box'] = np.array([box_x, box_y]) #, box_z])
                    boxsize = [box_x, box_y]
                if items[1] == 'ATOMS':
                    header = items[2:]
#                     print header
                    x = np.zeros((N, 5))
                    for i in range(N):
                        line = f.readline().strip('\n').split()
                        for j in range(len(header)):
#                             x[int(line[0])-1, j] = float(line[j])
                            if j == 0:
                                x[int(line[0])-1, 0] = float(line[j])
                            if j == 15: #Psi6
                                x[int(line[0])-1, 1] = float(line[j])
                            if j == 2: #x
                                x[int(line[0])-1, 2] = float(line[j])
                            if j == 3: #y
                                x[int(line[0])-1, 3] = float(line[j])
                            if j == 19: #PE
                                x[int(line[0])-1, 4] = float(line[j])
#                     x[:,2] = step
                    features[cnt-1] = x
        return boxsize, features

def read_dump_angle(filer, min_step=0, max_step=1e10):
    '''
    Reads dump files from LAMMPS and return data sorted by snap
    
    filer: Input file
    min_step: default 0
    max_step: maximum number of steps to read
    
    returns
        features: Dict sorting data by snap, e.g. features[snap_number] = N x 5 matrix
        
        Each column corresponds to:
        
        | x | y | particle_id | time_step | psi_6 |
    '''
    data=[]
    snapshot = {}
    features = {}
#     snapshot['traj'] = filer
    minstep_flag = False
    read_flag = False
    if (max_step<min_step) : min_step, max_step = max_step, min_step
    cnt = 0
    with open(filer,'r') as f:
        while True:
            line=f.readline().strip('\n')
            if not line: break
            items = line.split()
            if items[0] == 'ITEM:':
#                 try:
                if items[1] == 'TIMESTEP':
                    step = int(f.readline().split(' ')[0])
                    if step > max_step :
#                         print ('max_step reached (%d)' % max_step)
#                         print ('From %s, last TIMESTEP %d' % (filer, data[-1]['step']))
                        return features
                    if ( step >= min_step and minstep_flag == False) :
#                         print 'From %s, first TIMESTEP reached (%d)' % (filer, min_step)
                        minstep_flag = True
                    if (step >= min_step and step <= max_step):
                        read_flag = True
                        cnt += 1
#                         print '%d : reading TIMESTEP %d' % (cnt,step)
                    snapshot['step'] =  step
                        
                if items[1] == 'NUMBER':
                    N = int(f.readline().split()[0])
                    snapshot['N'] =  N
                if items[1] == 'BOX':
                    line = f.readline().split()
                    box_x = float(line[1]) - float(line[0])
                    line = f.readline().split()
                    box_y = float(line[1]) - float(line[0])
                    line = f.readline().split()
                    box_z = float(line[1]) - float(line[0])
                    snapshot['box'] = np.array([box_x, box_y]) #, box_z])
                    boxsize = [box_x, box_y]
                if items[1] == 'ATOMS':
                    header = items[2:]
#                     print header
                    x = np.zeros((N, 6))
                    for i in range(N):
                        line = f.readline().strip('\n').split()
                        for j in range(len(header)):
#                             x[int(line[0])-1, j] = float(line[j])
                            if j == 0:
                                x[int(line[0])-1, 0] = float(line[j])
                            if j == 2: #x
                                x[int(line[0])-1, 1] = float(line[j])
                            if j == 3: #y
                                x[int(line[0])-1, 2] = float(line[j])
                            if j == 15: #psi6
                                x[int(line[0])-1, 3] = float(line[j])
                            if j == 16: #psi6_real
                                x[int(line[0])-1, 4] = float(line[j])
                            if j == 17: #psi6_imaginary
                                x[int(line[0])-1, 5] = float(line[j])
#                     x[:,2] = step
                    features[cnt-1] = x
        return boxsize, features

def read_dump_ovito(filer, min_step=0, max_step=1e10):
    '''
    Reads dump files from LAMMPS and return data sorted by snap
    
    filer: Input file
    min_step: default 0
    max_step: maximum number of steps to read
    
    returns
        features: Dict sorting data by snap, e.g. features[snap_number] = N x 5 matrix
        
        Each column corresponds to:
        
        | x | y | particle_id | time_step | psi_6 |
    '''
    data=[]
    snapshot = {}
    features = {}
#     snapshot['traj'] = filer
    minstep_flag = False
    read_flag = False
    if (max_step<min_step) : min_step, max_step = max_step, min_step
    cnt = 0
    with open(filer,'r') as f:
        while True:
            line=f.readline().strip('\n')
            if not line: break
            items = line.split()
            if items[0] == 'ITEM:':
#                 try:
                if items[1] == 'TIMESTEP':
                    step = int(f.readline().split(' ')[0])
                    if step > max_step :
#                         print ('max_step reached (%d)' % max_step)
#                         print ('From %s, last TIMESTEP %d' % (filer, data[-1]['step']))
                        return features
                    if ( step >= min_step and minstep_flag == False) :
#                         print 'From %s, first TIMESTEP reached (%d)' % (filer, min_step)
                        minstep_flag = True
                    if (step >= min_step and step <= max_step):
                        read_flag = True
                        cnt += 1
#                         print '%d : reading TIMESTEP %d' % (cnt,step)
                    snapshot['step'] =  step
                        
                if items[1] == 'NUMBER':
                    N = int(f.readline().split()[0])
                    snapshot['N'] =  N
                if items[1] == 'BOX':
                    line = f.readline().split()
                    box_x = float(line[1]) - float(line[0])
                    line = f.readline().split()
                    box_y = float(line[1]) - float(line[0])
                    line = f.readline().split()
                    box_z = float(line[1]) - float(line[0])
                    snapshot['box'] = np.array([box_x, box_y]) #, box_z])
                    boxsize = [box_x, box_y]
                if items[1] == 'ATOMS':
                    header = items[2:]
#                     print header
                    x = np.zeros((N, 5))
                    for i in range(N):
                        line = f.readline().strip('\n').split()
                        for j in range(len(header)):
#                             x[int(line[0])-1, j] = float(line[j])
                            if j == 0: 
                                x[int(line[0])-1, 0] = float(line[j])
                            if j == 1: #x
                                x[int(line[0])-1, 1] = float(line[j])
                            if j == 2: #y
                                x[int(line[0])-1, 2] = float(line[j])
                            if j == 3: #av
                                x[int(line[0])-1, 3] = float(line[j])
                            if j == 4: #clu
                                x[int(line[0])-1, 4] = float(line[j])
#                     x[:,2] = step
                    features[cnt-1] = x
        return boxsize, features

def sort_group(frames, cutoff, group_col):
    frames_net = np.array([frames[f]for f in range(len(frames))])
    result = []
    for i in range (len(frames_net)):
        single_frame = frames_net[i]
        group_number = int(np.max(single_frame[:,group_col]))
        # print(group_number)
        groups = []
        for j in range (1, group_number+1):
            groups.append(single_frame[single_frame[:,group_col] == j])
        group = []
        for k in range (len(groups)):
            if len(groups[k]) > int(cutoff) :
                group.append(groups[k])
        result.append(group)
    return np.array(result)


def get_frames_from_xyz(filename, ncols=3, skipcol=0):
    f = open(filename, 'r')
    frames = []
    for line in f:
        is_head = re.match(r'(\d+)\n', line)
        if is_head:
            frames.append([])
            particle_num = int(is_head.group(1))
            f.readline()  # jump through comment line
            for j in range(particle_num):
                data = re.split(r'\s', f.readline())[skipcol: skipcol + ncols]
                frames[-1].append(list(map(float, data)))
    f.close()
    return np.array(frames)


import pysftp
def get_data_from_BC (host, username, password, remote_file_path, local_file_path):
    with pysftp.Connection(host = host, username = username, password = password) as sftp:
        print ("connection succesfully stablished ... ")
        remote_file_path = remote_file_path # example: '/newhome/le19932/LouisEP/steady_state_check/dump_steady_state_5000_region_1e9_logfreq.lammpstrj'
        local_file_path = local_file_path
        sftp.get(remote_file_path, local_file_path)

import math

# Finds the area of the intersection between a circle and a rectangle
# http://stackoverflow.com/questions/622287/area-of-intersection-between-circle-and-rectangle
def circleRectangleIntersectionArea(r, xcenter, ycenter, xleft, xright, ybottom, ytop):         
#find the signed (negative out) normalized distance from the circle center to each of the infinitely extended rectangle edge lines,
    d = [0, 0, 0, 0]
    d[0]=(xcenter-xleft)/r
    d[1]=(ycenter-ybottom)/r
    d[2]=(xright-xcenter)/r
    d[3]=(ytop-ycenter)/r
    #for convenience order 0,1,2,3 around the edge.

    # To begin, area is full circle
    area = math.pi*r*r

    # Check if circle is completely outside rectangle, or a full circle
    full = True
    for d_i in d:
        if d_i <= -1:   #Corresponds to a circle completely out of bounds
            return 0
        if d_i < 1:     #Corresponds to the circular segment out of bounds
            full = False

    if full:
        return area

    # this leave only one remaining fully outside case: circle center in an external quadrant, and distance to corner greater than circle radius:
    #for each adjacent i,j
    adj_quads = [1,2,3,0]
    for i in [0,1,2,3]:
        j=adj_quads[i]
        if d[i] <= 0 and d[j] <= 0 and d[i]*d[i]+d[j]*d[j] > 1:
            return 0

    # now begin with full circle area  and subtract any areas in the four external half planes
    a = [0, 0, 0, 0]
    for d_i in d:
        if d_i > -1 and d_i < 1:    
            a[i] = math.asin( d_i )  #save a_i for next step
            area -= 0.5*r*r*(math.pi - 2*a[i] - math.sin(2*a[i])) 

    # At this point note we have double counted areas in the four external quadrants, so add back in:
    #for each adjacent i,j
    
    for i in [0,1,2,3]:
        j=adj_quads[i] 
        if  d[i] < 1 and d[j] < 1 and d[i]*d[i]+d[j]*d[j] < 1 :
            # The formula for the area of a circle contained in a plane quadrant is readily derived as the sum of a circular segment, two right triangles and a rectangle.
            area += 0.25*r*r*(math.pi - 2*a[i] - 2*a[j] - math.sin(2*a[i]) - math.sin(2*a[j]) + 4*math.sin(a[i])*math.sin(a[j]))
    
    return area

def get_h_corr(heights, x_vals, bins, pbc_box_x):
    dx = pdist(x_vals[:, np.newaxis])  # pair wise x-distance
    dx[dx > pbc_box_x / 2] = pbc_box_x - dx[dx > pbc_box_x / 2] # pbc
    # get pairwise heights product
    h_flctn = heights - heights.mean()
    h_product = []
    for i in range(len(h_flctn)):
        for j in range(i+1, len(h_flctn)):
            h_product.append(h_flctn[i] * h_flctn[j])
    dx = np.concatenate([dx, np.zeros(len(h_flctn))])
    h_product = np.concatenate((h_product, h_flctn * h_flctn))
    return binned_statistic(dx, h_product, bins=bins, statistic='mean')[0]

# example: C_x_t = get_h_corr(top[:, 2], top[:, 1], bins=np.linspace(0, 50, 50), pbc_box_x=box_size[0])

def intersection_area(d, R, r):
    """Return the area of intersection of two circles.

    The circles have radii R and r, and their centres are separated by d.

    """

    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    return ( r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta)))

def func_density_profile(x,mean_fracation,liquid_area_fraction,gas_area_fraction,interfacial_position,interfacial_width):
    return mean_fracation+ (liquid_area_fraction+gas_area_fraction)/2*np.tanh((x-interfacial_position)/interfacial_width) 

def func_HH_fit(x,surface_tension,interfacial_width):
    return 1 / (2*np.pi*surface_tension) * kn(0,x/interfacial_width)

def get_drifting_coor(initial_frame_number, frames, everyframe, box_size):
    dim = len(box_size)
    frames_copy = np.array([frames[f]for f in range(len(frames))])
    frame_net = []
    for i in range (initial_frame_number, len(frames)-everyframe, everyframe):
        distance = frames[i+everyframe][:,2:4] - frames[i][:,2:4]
        for j in range(dim): 
            for k in range (len(distance)):
                if abs(distance[k][j]) > box_size[j]/2:
                    if distance[k][j] > 0:
                        distance[k][j] = distance[k][j] - box_size[j]
                    else:
                        distance[k][j] = box_size[j] + distance[k][j]
                else:
                    distance[k][j]= distance[k][j]
        real_dis = frames_copy[i][:,2:4] + np.array(distance)
        frames_copy[i+everyframe][:,2:4] = real_dis
        frame_net.append(real_dis)
    frames_net = np.array(frame_net)
    return frames_net

def get_drift_coor_y (initial_frame_number, frames, everyframe, y_box):
    frames_copy = np.array([frames[f]for f in range(len(frames))])
    frame_net = []
    for i in range (initial_frame_number, len(frames)-everyframe, everyframe):
        distance = frames[i+everyframe][:,3] - frames[i][:,3]
        for j in range (len(distance)):
            if abs(distance[j]) > y_box/2:
                if distance[j] > 0:
                    distance[j] = distance[j] - y_box
                else:
                    distance[j] = y_box + distance[j] 
            else:
                distance[j] = distance[j]
        real_dis = frames_copy[i][:,3] + np.array(distance)
        frames_copy[i+everyframe][:,3] = real_dis
        frame_net.append(frames_copy[i+everyframe][:,2:4])
    frames_net = np.array(frame_net)
    return frames_net




def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return np.array((theta, rho))

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array((x, y))

def get_circular_distance(sorted_particles):
    '''
    formate of sorted_particles: np.array(n,2)
    '''
    total_distance = 0
    each_distance = [0]
    for i in range (len(sorted_particles[:,0])-1):
        j = i + 1
        total_distance += ((sorted_particles[:,0][i]-sorted_particles[:,0][j])**2 + (sorted_particles[:,1][i]-sorted_particles[:,1][j])**2)**(1/2)
        each_distance.append(total_distance)
    return total_distance, np.array(each_distance)

def get_PBCs_extension(sframe, box_size):
    '''
    sigle_frame with np.array(2,n)
    box_size = (2,)
    '''
    top = np.vstack((sframe[0], sframe[1] + box_size[1])).T
    bottom = np.vstack((sframe[0], sframe[1] - box_size[1])).T
    right = np.vstack((sframe[0] + box_size[0], sframe[1])).T
    left = np.vstack((sframe[0] - box_size[0], sframe[1])).T
    topleft = np.vstack((sframe[0] - box_size[0], sframe[1] + box_size[1])).T
    topright = np.vstack((sframe[0] + box_size[0], sframe[1] + box_size[1])).T
    bottomleft = np.vstack((sframe[0] - box_size[0], sframe[1] - box_size[1])).T
    bottomright = np.vstack((sframe[0] + box_size[0], sframe[1] - box_size[1])).T
    mix = np.concatenate((sframe.T,top,bottom,right,left,topleft,topright,bottomleft,bottomright),axis = 0)
    return mix

def should_join(p1, p2):
    return 0 in pdist(np.concatenate((p1, p2))[:, None])


def join_pairs(pairs, copy=True):
    """
    Args:
        pairs (list): a list of tuples
        copy (bool): very difficult
    Example:
        >>> pairs = [(2, 3), (3, 5), (2, 6), (8, 9), (9, 10)]
        >>> join_pairs(pairs)
        [(2, 3, 5, 6), (8, 9, 10)]
    """
    pairs_joined = pairs.copy() if copy else pairs
    for p1 in pairs_joined:
        for p2 in pairs_joined:
            if (p1 is not p2) and should_join(p1, p2):
                pairs_joined.append(tuple(set(p1 + p2)))
                pairs_joined.remove(p1)
                pairs_joined.remove(p2)
                pairs_joined = join_pairs(pairs_joined, copy=False)
                return pairs_joined  # join once per recursion
    return pairs_joined

def load_str_from_file(f, rows):
    """
    load data from a file handler to numpy array
    Args:
        f (TextIOWrapper): a file handler obtained from `open`
        rows (int): the number of rows to load into the numpy array
    """
    data_str = ""
    for _ in range(rows):
        data_str += f.readline()
    io_stream = StringIO(data_str)
    return np.loadtxt(io_stream)


def parse_dump(filename):
    """
    Args:
        filename (str): path to the .dump file
    Return:
        dict: the configuration and metadata in different frames
    """
    f = open(filename, 'r')
    frames = []
    for line in f:
        if re.match(r'ITEM: TIMESTEP\n', line):
            time_step = int(re.search('\d+', f.readline()).group(0))
            f.readline()  # jump through one line
            particle_num = int(re.search('\d+', f.readline()).group(0))
            f.readline()  # jump through one line
            box = load_str_from_file(f, 3)
            col_names = re.split('\s+', f.readline())[1:]
            configuration = load_str_from_file(f, particle_num)
            frames.append({
                "conf": configuration,
                "col_names": col_names,
                "box": box,
                "particle_num": particle_num,
                "time_step": time_step,
            })
    f.close()
    return frames


def pdist_pbc(positions, box):
    """
    Get the pair-wise distances of particles in a priodic boundary box
    Args:
        positiosn (:obj:`numpy.ndarray`): coordinate of particles, shape (N, dim)
        box (:obj:`numpy.ndarray`): the length of the priodic bondary, shape (dim,)
    Return:
        :obj:`numpy.ndarray`: the pairwise distance, shape ( (N * N - N) / 2, ),
            use :obj:`scipy.spatial.distance.squareform` to recover the matrix form.
    """
    n, dim = positions.shape
    result = np.zeros(int((n * n - n) / 2), dtype=np.float64)
    for d in range(dim):
        dist_1d = pdist(positions[:, d][:, np.newaxis])
        dist_1d[dist_1d > box[d] / 2] -= box[d]
        result += np.power(dist_1d, 2)
    return np.sqrt(result)


def get_interface_from_voro(points, box, threshold):
    """
    Use the voronoi volume to get the interface particles
    Args:
        points (np.ndarray): shape (n, 3), the position of z axis
            should all be zero for freud
        box (np.ndarray): the box size of x and y axis, shape (2,)
        threshold (double): if the volume is above threshold value
        the particle is the interface
    Return:
        np.ndarray: the points corresponding to the interface, may contain bubbles
    """
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


def get_percolating_cycles(
    points, box, neighbour_distance, percolate_cutoff=0.9, connection_cutoff=4
):
    """
    Generate a graph based on distance threshold and find cycles in the graph
        that is percolating in the X-axis
    Args:
        points (np.ndarray): shape (n, 2), the position of the interface
            that may contain bubboles
        box (np.ndarray): the pbc box size of x and y axis, shape (2,)
        neighbour_distance (float): an edge is if two particles have distance
            smaller than this value
        percolate_cutoff (float): a percolate circle shoud span this_value * box_length
        connection_cutoff (double): a percolate circle should be continues, the minimum
            distance between two particles should be smaller than this_value * nn_distance
    """
    distances = squareform(pdist_pbc(points, box))  # adjacancy matrix
    adjacency = distances < neighbour_distance
    graph = nx.from_numpy_matrix(adjacency)

    cycles = nx.cycle_basis(graph)
    cycles_percolate = []

    num_cutoff = (box[0] * percolate_cutoff) / neighbour_distance

    for c in cycles:
        if len(c) > num_cutoff:
            indices = np.array(c)
            x_vals = points[indices, 0]
            x_sorted = np.sort(x_vals)
            is_percolate = (x_vals.max() - x_vals.min()) > box[0] * percolate_cutoff
            no_gap = (x_sorted[1:] - x_sorted[:-1]).max() < neighbour_distance * connection_cutoff
            if is_percolate and no_gap:
                cycles_percolate.append(indices)
    for i, cp in enumerate(cycles_percolate):
        neighbours = np.concatenate([
            np.fromiter(graph.neighbors(node), dtype=int) for node in cp
        ])
        cycles_percolate[i] = np.concatenate((cp, neighbours), axis=0)
    return cycles_percolate

def get_cycles(
    points, box, neighbour_distance, percolate_cutoff=0.9, connection_cutoff=4
):
    """
    Generate a graph based on distance threshold and find cycles in the graph
        that is percolating in the X-axis
    Args:
        points (np.ndarray): shape (n, 2), the position of the interface
            that may contain bubboles
        box (np.ndarray): the pbc box size of x and y axis, shape (2,)
        neighbour_distance (float): an edge is if two particles have distance
            smaller than this value
        percolate_cutoff (float): a percolate circle shoud span this_value * box_length
        connection_cutoff (double): a percolate circle should be continues, the minimum
            distance between two particles should be smaller than this_value * nn_distance
    """
    distances = squareform(pdist_pbc(points, box))  # adjacancy matrix
    adjacency = distances < neighbour_distance
    graph = nx.from_numpy_matrix(adjacency)

    cycles = nx.cycle_basis(graph)

    num_cutoff = (box[0] * percolate_cutoff) / neighbour_distance

    for i, cp in enumerate(cycles):
        if len(cp) < 10:
            continue
        neighbours = np.concatenate([
            np.fromiter(graph.neighbors(node), dtype=int) for node in cp
        ])
        cycles[i] = np.concatenate((cp, neighbours), axis=0)
    return cycles


def merge_overlap_interface_indices(interface_indices):
    """
    Merge the interface indices if they have a common element/index
    """
    n = len(interface_indices)
    if n <= 1:
        return interface_indices
    else:
        should_merge = []
        for i in range(n):
            for j in range(i + 1, n):
                if len(np.intersect1d(
                    interface_indices[i], interface_indices[j]
                )) > 0:
                    should_merge.append((i, j))
        if len(should_merge) == 0:
            return interface_indices
        else:
            to_merge = join_pairs(should_merge)
            result = [interface_indices[x] for x in range(n) if x not in np.concatenate(to_merge)]
            for indices in to_merge:
                merged = np.unique(np.concatenate([interface_indices[x] for x in indices]))
                result.append(merged)
            return result


def get_interface_refined(
    points, box, neighbour_distance, percolate_cutoff=0.9, connection_cutoff=4
):
    """
    Remove deceptive interface points that were actually internal bubbles
        by finding circles in a Graph representation of the data
    Args:
        points (np.ndarray): shape (n, 2), the position of the interface
            that may contain bubboles
        box (np.ndarray): the pbc box size of x and y axis, shape (2,)
    Return:
        np.ndarray: the points corresponding to the true interfaces
    """
#     interface_indices = get_percolating_cycles(
#         points, box, neighbour_distance, percolate_cutoff, connection_cutoff
#     )
    interface_indices = get_cycles(
        points, box, neighbour_distance, percolate_cutoff, connection_cutoff
    )
    merged_indices = merge_overlap_interface_indices(interface_indices)
    return [points[mi] for mi in merged_indices]

def get_dnn_mean(points, box):
    """
    Get the average nearest neighbour distances in a PBC box
    Args:
        points (np.ndarray): shape (n, 2), the position of the interface
            that may contain bubboles
        box (np.ndarray): the pbc box size of x and y axis, shape (2,)
    Return:
        np.ndarray: the points corresponding to the true interfaces
    """
    distances = squareform(pdist_pbc(points, box))
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=0).mean()


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
