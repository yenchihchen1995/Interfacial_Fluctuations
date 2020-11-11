#amoeba.py

#%%
from Definition import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
%load_ext autoreload
%autoreload 2

# %%
data = read_dump_ovito("Path") #id x y AtomicVolume Cluster

# %%
box_size = data[0]; frames = data[1]
print ('Box size = ', box_size, '\nParticle numbers = ',len(frames[0]), '\nFrame number = ', len(frames) )

#%%
frame_number = 1500
which_group = 0
frame = frames[frame_number]
points = frame[:,1:3]
particle_sorted = sort_group(frames, 4)
print ('Numbers of groups: ', len(particle_sorted[frame_number]))
single_group = particle_sorted[frame_number][which_group]
plt.plot(*points.T[:2],'o',ms = 2)
plt.plot(single_group[:,1], single_group[:,2], 'o',ms = 2)
# %%
interface  = single_group[single_group[:,3] > 0.3][:,1:3]
# plt.plot(*interface.T[:2],'o', c = 'darkorange')
plt.xlabel('x')
plt.ylabel('y')
sort_interface = sort_particles(interface, 5)
plt.plot(*sort_interface.T[:2],'-o')

# %%
mean_irf = []
for i in range (0, len(sort_interface),4):
    mean_irf.append(np.average(sort_interface[i:i+3:,:2],axis=0))
mean_irf = np.array(mean_irf)
mean_irf = np.vstack((mean_irf, mean_irf[0]))
print (mean_irf.shape)
# plt.plot(*mean_irf.T[:2], '-o', )

from scipy import interpolate
t = np.arange(0, 1, .01)
x = mean_irf[:,0]
y = mean_irf[:,1]
tck, u = interpolate.splprep([x, y], s=0)
unew = np.arange(0, 1.00, 0.002)
out = interpolate.splev(unew, tck)
plt.plot(*sort_interface.T[:2],'o', label = 'interfacial particles')
plt.plot(*out[:2],'-o',ms=1, label= 'mean interface')
plt.legend()



#%%
height = []
for i in range (len(interface)):
    everyheight = []
    for j in range (len(out[0])):
        distance = ((interface[i][0] - out[0][j])**2 + (interface[i][1] - out[1][j])**2)** (1/2)
        everyheight.append(distance)
    height.append(np.min(everyheight))
plt.show()
plt.xlabel('$h/\u03c3$')
plt.ylabel('$P(h)$')
plt.hist(height, bins = 15)

print (len(height))
plt.show()

#%%
dis_interface = get_circular_distance(sort_interface)
dx = dis_interface[1]
print ('total distance:', dis_interface[0])
dx[dx > dis_interface[0] / 2] = dis_interface[0] - dx[dx > dis_interface[0] / 2] # pbc
bins = np.linspace(0, dis_interface[0]/2, 20)
H_x = []
for i in range (len(bins)):
    h_x = []
    for j in range (len(dx)):
        if bins[i] <= dx[j]:
            h_x.append(height[i]*height[j])
    H_x.append(np.nanmean(h_x))
# print (H_x)
plt.xlabel('$x/\u03c3$')
plt.ylabel('$g_h(x)$')
plt.plot(bins,H_x,'-o')
#%% Main
from scipy import interpolate
initial_frame = 1000
everyframe = 500
particle_sorted = sort_group(frames, 10)
Total_HH = []
for i in range (initial_frame,len(frames)-everyframe, everyframe):
    frame_number = i
    frame = frames[frame_number]
    points = frame[:,1:3]
    for g in range(len(particle_sorted[i])):
        which_group = g
        single_group = particle_sorted[frame_number][which_group]
        interface  = single_group[single_group[:,3] > 0.3][:,1:3]
        sort_interface = sort_particles(interface, 5)
        mean_irf = []
        for j in range (0, len(sort_interface),3):
            mean_irf.append(np.average(sort_interface[j:j+3:,:2],axis=0))
        mean_irf = np.array(mean_irf)
        mean_irf = np.vstack((mean_irf, mean_irf[0]))
        t = np.arange(0, 1, .01)
        x = mean_irf[:,0]
        y = mean_irf[:,1]
        tck, u = interpolate.splprep([x, y], k=2)
        unew = np.arange(0, 1.00, 0.001)
        out = interpolate.splev(unew, tck)
        # plt.plot(*sort_interface.T[:2],'o')
        # plt.plot(*out[:2],'o',ms=1)
        height = []
        for a in range (len(interface)):
            everyheight = []
            for b in range (len(out[0])):
                distance = ((interface[a][0] - out[0][b])**2 + (interface[a][1] - out[1][b])**2)** (1/2)
                everyheight.append(distance)
            height.append(np.min(everyheight))
    Total_HH.append(height)
    # break
        # dis_interface = get_circular_distance(sort_interface)
        # dx = dis_interface[1]
        # # print ('total distance:', dis_interface[0])
        # dx[dx > dis_interface[0] / 2] = dis_interface[0] - dx[dx > dis_interface[0] / 2] # pbc
        # bins = np.linspace(0, dis_interface[0]/2, 20)
        
        # H_x = []
        # for c in range (len(bins)):
        #     h_x = []
        #     for d in range (len(dx)):
        #         if bins[c] <= dx[d]:
        #             h_x.append(height[c]*height[d])
        #         else:
        #             break
        #     H_x.append(np.nanmean(h_x))
        # Total_H_x.append(np.nanmean(H_x))

#%%
flattened_list = [y for x in Total_HH for y in x]
# print (flattened_list[0])
# plt.xlim((-0.2,3.5))
counts, bins = np.histogram(flattened_list, bins = np.linspace(0,6,20))
# print(bins)
bins = bins[:-1]+(bins[1]-bins[0])/2
counts = counts / counts.sum()
plt.xlabel('$h/\u03c3$')
plt.ylabel('$P(h)$')
plt.plot(bins, counts, 'o')

# print(len(counts))
#%%
from scipy.optimize import curve_fit
def func(x,a):
    return 1/((2*np.pi*a**2)**0.5) * np.exp(-x**2/(2*a**2))
popt, pcov = curve_fit(func, bins ,counts)
print(popt)
plt.plot(bins, counts, 'o')
plt.plot(bins, func(bins, *popt), label = 'fitting')
# plt.xlim((-0.2,5))
plt.legend()
# print (func(bins, *popt))
#%%
result = np.vstack((bins,counts)).T
print(result)


#%%
with open ('data/amoeba_0.pickle','wb') as file:
    pickle.dump(result,file)

#%%

with open ('data/amoeba_10.pickle','rb') as read:
    test_0 = pickle.load(read)
with open ('data/amoeba_20.pickle','rb') as read:
    test_1 = pickle.load(read)
def func(x,a):
    return 1/((2*np.pi*a**2)**0.5) * np.exp(-(x**2/(2*a**2)))
params, pp = curve_fit(func, bins ,counts)
params1, pp = curve_fit(func, bins ,counts)
plt.plot(*test_0.T[:2],'o', label = 'Pe=10',markerfacecolor = 'None')
plt.plot(*test_1.T[:2],'^', label = 'Pe=20',markerfacecolor = 'None')
# params, params_covariance = curve_fit(func,test_0[:,0],test_0[:,1],maxfev = 1000000)
# params1, params_covariance1 = curve_fit(func,test_1[:,0],test_1[:,1],maxfev = 1000000)
plt.legend()
plt.xlabel('$h/\u03c3$')
plt.ylabel('$P(h)$')
# plt.xlim((-0.1,3.5))
# plt.plot(test_0[:,0],func(test_0[:,0], *params) ,color='purple')
# plt.plot(test_1[:,0],func(test_1[:,0], *params1) ,color='orange')
plt.savefig('amoeba_ph.png',dpi = 300)








%%
from sklearn.neighbors import NearestNeighbors
points = interface
clf = NearestNeighbors(2).fit(points)
G = clf.kneighbors_graph()
import networkx as nx
T = nx.from_scipy_sparse_matrix(G)
order = list(nx.dfs_preorder_nodes(T, 0))

interface_order = interface[order]

edge_dists = np.linalg.norm(
    interface_order[1:] - interface_order[:-1], axis=1
)
to_remain = edge_dists < 10
to_remain = np.insert(to_remain, 0, True)

interface_order = interface_order[to_remain]
ß
plt.plot(*interface_order.T)
plt.show()

# %%
