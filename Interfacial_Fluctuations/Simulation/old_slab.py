#%%
from Definition import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
#%% Data
import pickle

load_pickle = True

Pe = 0
if load_pickle:
    with open('slab_%i.pickle'%(Pe), 'rb') as file:
        data = pickle.load(file)
else:
    data = read_dump("/Volumes/E/OneDrive - University of Bristol/LouisEP/Data_collection/epsilon_0.25/5040/5040_%i.lammpstrj" %(Pe)) #id x y AtomicVolume Cluster
    with open('slab_%i.pickle'%(Pe), 'wb') as file:
        pickle.dump(data, file)
#%% Data Info
box_size = data[0]; frames = data[1]
print ('Box size = ', box_size, '\nParticle numbers = ',len(frames[0]), '\nFrame number = ', len(frames) )

#%% Choose the initial frame
frame_number = 3000
initial_frame = frames[frame_number]
plt.scatter(initial_frame[:,2],initial_frame[:,3])
neighbour_dis = get_dnn_mean(initial_frame[:,2:4],box_size)
print(neighbour_dis)
#%% HH-correlation
everyframe = 500
frames_net = np.array([frames[f]for f in range(len(frames))])   # < 我修改了这里
# print (frames_net.shape)
bins = np.linspace(0,box_size[0]/2,40)
HH_corr = []
HF = []
for i in range (frame_number, len(frames)-everyframe, everyframe):
    distance = []
    j = i+everyframe
    particle_pos0 = frames[i][:,3]
    particle_pos1 = frames[j][:,3]
    y_dis = np.array(particle_pos1-particle_pos0)
    # plt.scatter (frames[i][:,2],frames[i][:,3])
    for k in range (len(y_dis)):
        if abs(y_dis[k]) > box_size[1]/2:
            if y_dis[k] > 0:
                distance.append(y_dis[k]-box_size[1])
            else:
                distance.append(box_size[1]+y_dis[k])
        else:
            distance.append(y_dis[k])
    real_y = frames_net[i][:,3] + np.array(distance)
    frames_net[j][:,3] = real_y  
    # frames_net[j][:, 3] -= real_y.mean()
    plt.scatter(frames_net[j][:,2], frames_net[j][:,3])   # < 原始圖形
    points = frames_net[j][:,2:4]
    interfaces  = get_interface_from_voro(points, [box_size[0],300], 1.0)
    real_interfaces = get_interface_refined(interfaces, box_size, neighbour_dis)
    h_x = []
    print (len(real_interfaces))

#     for interface in real_interfaces:
#         plt.scatter (interface[:,0], interface[:,1],s=30) # 界面
#         dx = pdist(interface[:,0][:, np.newaxis])  # pair wise x-distance
#         dx[dx > box_size[0] / 2] = box_size[0] - dx[dx > box_size[0] / 2] # pbc
#         h_flctn = interface[:,1]-interface[:,1].mean()
#         HF.append(h_flctn)
#         h_product = []
#         for i in range(len(h_flctn)):
#             for j in range(i+1, len(h_flctn)):
#                 h_product.append(h_flctn[i] * h_flctn[j])
#         dx = np.concatenate([dx, np.zeros(len(h_flctn))])
#         h_product = np.concatenate((h_product, h_flctn * h_flctn))
#         Total = []
#         for k in range (len(bins)):
#             total = []
#             for n in range(len(h_product)):
#                 if dx[n] <= bins[k]:
#                     total.append(h_product[n])
#             Total.append(np.nanmean(total))
#         h_x.append(Total)
#         # plt.yscale('log')
#         # plt.plot(bins,Total,'-o')
# plt.show()
# H_x = np.mean(np.array(h_x),axis=0)
# plt.plot(bins,H_x,'-o')
# # plt.yscale('log')
# plt.show()

# HF = np.concatenate(HF)
# plt.hist(HF, bins=30)
# plt.show()

#%% Data saving
bins, H_x = np.array(bins), np.array(H_x)
result = np.vstack((bins,H_x)).T
with open ('data/1000_data_2.pickle','wb') as file:
    pickle.dump(result,file)


# %% Data ploting
from scipy.special import k0
from scipy.optimize import curve_fit
with open ('data/1000_data_0.pickle','rb') as read_file:
    test_0 = pickle.load(read_file)
with open ('data/1000_data_1.pickle','rb') as read_file:
    test_1 = pickle.load(read_file)
with open ('data/1000_data_2.pickle','rb') as read_file:
    test_2 = pickle.load(read_file)
test_0 = test_0[~np.isnan(test_0).any(axis=1)]
test_1 = test_1[~np.isnan(test_1).any(axis=1)]
test_2 = test_2[~np.isnan(test_2).any(axis=1)]
# test_0[:,1] /= test_0[0,1]
# test_1[:,1] /= test_1[0,1]
# test_2[:,1] /= test_2[0,1]
# print(test_0)
def func_(x,a,b):
    return 1 / (2*np.pi*a) * k0(x/b)
def func(x,a,b):
    return a*x+b
# params, params_covariance = curve_fit(func,test_0[:,1],test_0[:,0],maxfev = 1000000)
# params1, params1_covariance = curve_fit(func,test_1[:,1],test_1[:,0],maxfev = 1000000)
# params2, params2_covariance = curve_fit(func,test_2[:,1],test_2[:,0],maxfev = 1000000)
# plt.plot(func(test_0[:,1], *params), test_0[:,1] ,color='purple')
# plt.plot(func(test_1[:,1], *params1), test_1[:,1] ,color='orange')
# plt.plot(func(test_2[:,1], *params2), test_2[:,1] ,color='blue')
plt.plot(test_0[:,0],test_0[:,1],'^',label = 'Pe=0',markerfacecolor = 'None')
plt.plot(test_1[:,0],test_1[:,1],'o',label = 'Pe=1',markerfacecolor = 'None')
plt.plot(test_2[:,0],test_2[:,1],'s',label = 'Pe=2',markerfacecolor = 'None')
plt.legend()
plt.yscale('log')
# plt.title('1000 particles')
plt.xlabel('$x/\u03C3$')
plt.ylabel('$g_h(x/\u03C3)$')
# plt.savefig('2400_HH_begining_same.png',dpi=300)
# plt.savefig('1000_HH.png',dpi=300)
# plt.savefig('1000_HH_begin.png',dpi=300)
# plt.savefig('2400_HH_log_begining_same.png',dpi=300)
# %%
