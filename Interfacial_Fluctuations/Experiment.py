#%% Definition
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
%load_ext autoreload
%autoreload 2

#%% Collect data and determine the box size--------------

with open(data_file,'rb') as file:
    data = pickle.load(file)
box_size = get_box_size(data)
print ('box size = ', box_size)

#%% obtain the specific data
frames = []
for f in range(initial_frame, len(data)-everyframe, everyframe):
    frames.append(np.array(data[f]))
frames = np.array(frames)
print('number of frames = ', len(frames))
#%% get clusters
cluster_frames = []
for frame in frames:
    cluster_frames.append(pick_clusters(frame,box_size,150))
cluster_frames = np.array(cluster_frames)

#%% get interfacial particles (slow)
interfacial_particles_frames = []
for i, frame in enumerate(cluster_frames):
    interfacial_particles_groups = []
    for group in frame:
        pos = group[:,1:3]
        interfacial_particles = get_interface_from_voronoi(pos, np.array(box_size)+200, 43)
        interfacial_particles_groups.append(np.array(interfacial_particles))
    print('No.%s frame has '%i, len(interfacial_particles_groups), 'groups')
    interfacial_particles_frames.append(np.array(interfacial_particles_groups))
interfacial_particles_frames = np.array(interfacial_particles_frames)

#%% sort interfacial particle
sorted_particles_frames = []
for frame in interfacial_particles_frames:
    sorted_particles_group = []
    for group in frame:
        sorted_particles = sort_particles(group, 95)
        sorted_particles_group.append(np.array(sorted_particles))
    sorted_particles_frames.append(np.array(sorted_particles_group))
sorted_particles_frames = np.array(sorted_particles_frames)
#%% get smoother interfaces
smoother_interface_frame = []
for i, frame in enumerate(sorted_particles_frames):
    smoother_interface_group = []
    for group in frame:
        smoother_interface = get_smoother_interfaces(group, curvature_cutoff)
        if smoother_interface.all() == 0:
            break
        else:
            smoother_interface_group.append(smoother_interface)
    print('No.%s frame has '%i, len(smoother_interface_group), 'smoother group')
    smoother_interface_frame.append(np.array(smoother_interface_group))
smoother_interface_frame = np.array(smoother_interface_frame)
#%% interface check
for frame in smoother_interface_frame:
    for group in frame:
        plt.plot(*group.T, 'o', ms =1)
    plt.show()
#%%


