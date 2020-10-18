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

# Collect data and determine the box size--------------

with open(data_file,'rb') as file:
    data = pickle.load(file)
box_size_check_frame = np.array(data[0])
pos = box_size_check_frame[:,1:3]
box_size = [np.ceil(np.max(pos[:,0])), np.ceil(np.max(pos[:,1]))]
print ('box size = ', box_size)

# Main work-------------------------
Total_H = []
kappa = []
HH_total = []
for f in range (initial_frame, len(data)-everyframe, everyframe):
    # sort group
    single_frame = np.array(data[f])
    max_group = int(np.max(single_frame[:,5]))
    sorted_data = []
    for i in range (1,max_group):
        group = []
        for j in range (len(single_frame)):
            if int(single_frame[j][5]) == i:
                group.append(single_frame[j])
        group = np.array(group)
        edges = [[0, 0], [box_size[0], box_size[1]]]
        near_edge = False
        delta = 5
        for edge_xy in edges:
            for dim, edge in enumerate(edge_xy):
                near_edge = near_edge or (np.abs(group[:, dim+1] - edge).min() < delta)
        if len(group) >= 150 and (not near_edge):
            sorted_data.append(group)
    kappa_group = []
    HH_group = []
    # each group
    for g in range(len(sorted_data)):
        single_cluster = sorted_data[g] 
        interfacial_particles = get_interface_from_voro(np.array(single_cluster)[:,1:3],box_size,50)
        interface = interfacial_particles
        sort_interface = sort_particles(interface, 50)
        ave_number = int(len(interface)/average_interface_number)
        sort_interface_average = np.vstack((
                sort_interface[-ave_number:],
                sort_interface,
                sort_interface[:ave_number]
                ))
        mean_irf = np.array((
            uniform_filter1d(sort_interface_average[:, 0], size=ave_number//2),
            uniform_filter1d(sort_interface_average[:, 1], size=ave_number//2)
        )).T[ave_number : -ave_number]
        x = mean_irf[:,0]
        y = mean_irf[:,1]
        tck, u = interpolate.splprep([x, y], s=0)
        unew = np.arange(0, 1.00, 0.001)
        out = np.array(interpolate.splev(unew, tck))
        # define negtive or positive
        polygon = Polygon(np.array(out).T) 
        distance  = cdist(sort_interface, np.array(out).T)
        height = distance.min(axis=1)
        height_sign = np.ones(len(height), dtype = int)
        for i in range (len(sort_interface)):
            point = Point(sort_interface[i,0], sort_interface[i,1])
            if polygon.contains(point) == True:
                height_sign[i] *= -1
        # out = np.array(out)
#        bins = np.linspace(0,40,15)
        bin_centres = (bins[1:]+bins[:-1]) / 2
        HH = hh_corr_cluster(sort_interface , out.T, height * height_sign, bins)
        HH_group.append(HH)
        HH_total.append(HH)
        #collect data      
        Total_H.append(height) # for height PDF
        girth = 0
        for d in range (len(out[0])-1):
            girth = girth + np.sqrt((out[0][d]-out[0][d+1])**2+(out[1][d]-out[1][d+1])**2)
        kappa_group.append(girth/12/np.mean(height)**2)
    kappa.append(np.mean(kappa_group))
#    print ('κ=',kappa)

# Plot and save the result

a = np.nanmean(HH_total, axis = 0)
plt.plot(bins[:-1],a,'-o')
plt.xlabel('x',fontsize = 16)
plt.ylabel('$g_h(x)$', fontsize = 16)
plt.show()

result = np.vstack((bins[:-1],a)).T
with open('HH_%s.pickle'%(average_interface_number),'wb') as file:
    pickle.dump(result,file)



