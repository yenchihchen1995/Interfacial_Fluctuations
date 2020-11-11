#%% Droplet.py
from Definition import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
import freud
%load_ext autoreload
%autoreload 2
#%%
data = read_dump('/Volumes/E/OneDrive - University of Bristol/LouisEP/Simulation/200_drop.lammpstrj')

# %%
box_size, frames = data[0], data[1]
print ('Box size = ', box_size, '\nParticle numbers = ',len(frames[0]), '\nFrame number = ', len(frames) )


#%% drifting
frames = np.array([frames[f]for f in range(len(frames))])
particle_pos = get_drifting_coor(219, frames, 50, box_size)
print(particle_pos.shape)
for particle_poss in particle_pos:
    plt.plot(*particle_poss.T[:2],'o')

#%% Circle Boundary
bins = np.linspace(0,2*np.pi/2,30)
mean_height = []
for i in range (len(particle_pos)):
    # plt.plot(*a[i].T[:2],'o')
    CM = np.average(particle_pos[i][:,:2],axis = 0)
    mod_pos = particle_pos[i] - CM
    interface = get_interface_from_voro(mod_pos, [100,100], 0.9)
    # plt.plot(*interface.T[:2],'o')
    Polar_interfaces = cart2pol(interface[:,0],interface[:,1])
    # print (sp_interface[0])
    plt.axes(polar = True)
    plt.plot(Polar_interfaces[0],Polar_interfaces[1],'o')
    dx = pdist(Polar_interfaces[0][:,np.newaxis])  # pair wise x-distance
    dx[dx > 2*np.pi / 2] = 2*np.pi - dx[dx > 2*np.pi / 2] # pbc
    mean_height.append(np.mean(Polar_interfaces[1]))
    h_flctn = Polar_interfaces[1] - np.mean(Polar_interfaces[1])
    h_product = []
    h_x = []
    for i in range(len(h_flctn)):
        for j in range(i+1, len(h_flctn)):
            h_product.append(h_flctn[i] * h_flctn[j])
    Total = []
    dx = np.concatenate([dx, np.zeros(len(h_flctn))])
    h_product = np.concatenate((h_product, h_flctn * h_flctn))
    for k in range (len(bins)):
        total = []
        for n in range(len(h_product)):
            if dx[n] < bins[k]:
                total.append(h_product[n])
        Total.append(np.mean(total))    
    h_x.append(Total)

#%%
H_x = np.array(h_x).reshape(30,)
plt.plot(bins,H_x,'-o')
print (np.mean(mean_height))


#%% plot
plt.title('400 Droplet with Pe=0')
plt.xlabel('radian')
plt.ylabel('$g_h(x)$')
plt.plot(bins, H_x,'o-',markerfacecolor= (1,1,1),alpha=0.7)
# plt.savefig('Pictures/400droplet_0.png',dpi = 300)

# %% Interface determination
single_frame = frames[len(frames)-1]
pos = single_frame[:,2:4]
CM = np.average(pos[:,:2],axis=0)
polar_pos = cart2pol((pos-CM)[:,0],(pos-CM)[:,1])
plt.axes(polar=True)
plt.plot(*polar_pos[:2],'o',fillstyle=None)
polar_pos = np.array(polar_pos).T
bins = np.linspace(0,box_size[0]-10,20)

area_frac = []
for j in range(1,len(bins),1):
    area = []
    for i in range (len(polar_pos)):
        a = intersection_area(polar_pos[i][1], bins[j],0.45) - intersection_area(polar_pos[i][1], bins[j-1],0.45)
        area.append(a)
        # area.append(count)
    area_frac.append(sum(area)/(bins[j]**2*np.pi - bins[j-1]**2*np.pi))
plt.show()
plt.axes(polar=None)

plt.plot(np.delete(bins,0),area_frac,'o')
# plt.plot(np.delete(bins,0),area_frac)

# %%
y = area_frac
x = np.delete(bins,0)
plt.plot(x,y,'o')
params, params_covariance = curve_fit(func_density_profile,x,y,maxfev = 1000000)
plt.plot(x, func_density_profile(x, *params) ,color='purple',label='Fitted function (Top)')
print("Top:\nMean\u03C6:",params[0], "\n\u03B4\u03C6:",params[2]+params[1] , "\nh:", params[3], "\n\u03BE:",params[4])
# %%

with open ('data/drop_HH_0.pickle','wb') as file:
    pickle.dump(result,file)

# %%
result = np.vstack((bins,H_x)).T
print (result)

# %%
with open ('data/drop_HH_0.pickle','rb') as read_file:
    pe0 = pickle.load(read_file)
with open ('data/drop_HH_1.pickle','rb') as read_file:
    pe1 = pickle.load(read_file)
with open ('data/drop_HH_2.pickle','rb') as read_file:
    pe2 = pickle.load(read_file)
pe0[:,0],pe1[:,0] = pe0[:,0]* np.mean(mean_height), pe1[:,0]*np.mean(mean_height); pe2[:,0] *= np.mean(mean_height)
# pe0[:,1] /= pe0[1,1]
# pe1[:,1] /= pe1[1,1]
# pe2[:,1] /= pe2[1,1]
# print (pe0)
# plt.title('400 droplet')
plt.plot(*pe0.T[:2],'o',label='Pe=0',markerfacecolor = 'None', ms = 8)
plt.plot(*pe1.T[:2],'^',label='Pe=1',markerfacecolor = 'None', markersize = 8)
plt.plot(*pe2.T[:2],'s',label='Pe=2',markerfacecolor = 'None', markersize = 8)

plt.xlabel('x')
plt.ylabel('$g_h(x)$')
plt.legend()
# plt.yscale('log')
# plt.savefig('400 droplet_begining.png',dpi = 300)
# plt.savefig('400 droplet.png',dpi = 300)

%%

# %% Blurring

number = 1000
position = frames[number][:,2:4]
CM = np.average(position[:,:2],axis=0)
position = position - CM
interface = get_interface_from_voro(position, box_size, 0.8)
# plt.plot(*interface.T[:2],'o')

interface_polar = cart2pol(interface[:,0],interface[:,1]).T
# print (interface_polar.shape)

order = np.argsort(interface_polar[:,0])
interface_sort_polar = interface_polar[order]
# print (interface_sort_polar)
# plt.axes(polar = True)
interface_sort = pol2cart(interface_sort_polar[:,1], interface_sort_polar[:,0]).T

plt.plot(*interface_sort.T[:2],'o')
# print (interface_sort)
mean_irf = []
for i in range (0, len(interface_sort),6):
    mean_irf.append(np.average(interface_sort[i:i+6:,:2],axis=0))
mean_irf = np.array(mean_irf)
mean_irf = np.vstack((mean_irf, mean_irf[0]))
print (mean_irf.shape)
plt.plot(*mean_irf.T[:2], '-o')


# %%
from scipy import interpolate
t = np.arange(0, 1.1, .1)
x = mean_irf[:,0]
y = mean_irf[:,1]
tck, u = interpolate.splprep([x, y], s=0)
unew = np.arange(0, 1.00, 0.002)
out = interpolate.splev(unew, tck)
plt.plot(*interface_sort.T[:2],'o')
plt.plot(*out[:2],'o',ms=1)
# print (len(out[0]))

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
dis_interface = get_circular_distance(interface_sort)
dx = dis_interface[1]
print ('total distance:', dis_interface[0])
dx[dx > dis_interface[0] / 2] = dis_interface[0] - dx[dx > dis_interface[0] / 2] # pbc
bins = np.linspace(0, dis_interface[0]/2, 10)
H_x = []
for i in range (len(bins)):
    h_x = []
    for j in range (len(dx)):
        if bins[i] <= dx[j]:
            h_x.append(height[i]*height[j])
    H_x.append(np.nanmean(h_x))
print (H_x)
plt.xlabel('$x/\u03c3$')
plt.ylabel('$g_h(x)$')
plt.plot(bins,H_x,'-o')
# %%
plt.xlabel('$x/\u03C3$')
plt.ylabel('$g_h(x/\u03C3$')
plt.plot(bins,H_x,'-o')







# %% least-squares
from scipy.optimize import curve_fit
def func(x,a,b):
    return a*x + b
for i in range (0, len(interface_sort[:,0]), 6):
    xdata = interface_sort[i:i+6][:,0]
    ydata = interface_sort[i:i+6][:,1]
    popt, pcov = curve_fit(func, xdata, ydata)
    # plt.plot(xdata, ydata,'o')
    plt.plot(xdata, func(xdata, *popt) , '-o', ms = 3);
plt.plot(*interface_sort.T[:2],'o')
# %%
