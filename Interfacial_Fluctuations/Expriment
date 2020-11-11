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
# %load_ext autoreload
# %autoreload 2

#%% Collect data and determine the box size--------------
'''
The data should be the format with 4 columns (x y voronoi_area cluster_number)
Welcome to use another file 'voronoi.py' to transfer your data.
'''
with open('Exp_voronoi.pickle','rb') as file:
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
    cluster_frames.append(pick_clusters(frame,box_size, minmum_particle_number_of_cluster))
cluster_frames = np.array(cluster_frames)

#%% get interfacial particles
interfacial_particles_frames = []
for i, frame in enumerate(cluster_frames):
    interfacial_particles_groups = []
    for group in frame:
        interfacial_particles = group[group[:,2] > voronoi_cutoff]
        interfacial_particles_groups.append(np.array(interfacial_particles))
    print('No.%s frame has '%i, len(interfacial_particles_groups), 'groups')
    interfacial_particles_frames.append(np.array(interfacial_particles_groups))
interfacial_particles_frames = np.array(interfacial_particles_frames)

#%% sort interfacial particle
sorted_particles_frames = []
for frame in interfacial_particles_frames:
    sorted_particles_group = []
    for group in frame:
        sorted_particles = sort_particles(group[:,:2], 95)
        sorted_particles_group.append(np.array(sorted_particles))
    sorted_particles_frames.append(np.array(sorted_particles_group))
sorted_particles_frames = np.array(sorted_particles_frames)
#%%
# for frame in sorted_particles_frames:
#     for group in frame:
#         plt.plot(*group.T[:2],'-o', ms = 1)
    
#     plt.show()

#%% get smoother regions
smoother_area = []
for frame in sorted_particles_frames:
    for group in frame:
        smoother_area_group = pick_smoother_area(group, 0.09)
        plt.plot(*group.T[:2], 'k')
        for area in smoother_area_group:
            if len(area) > 10:
                smoother_area.append(area)
                plt.plot(*area.T)
    plt.title('smoother areas')            
    plt.show()
smoother_area = np.array(smoother_area)
# %%
