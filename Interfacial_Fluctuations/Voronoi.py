#%%
from Definition import get_frames_from_xyz
import numpy as np
import freud
import pickle
# %%
data = get_frames_from_xyz('file_directory', ncols = 3, skipcol = 0)
pos = np.array(data[0])[:,1:3] # depends on where your x and y columns are
box_size = [np.ceil(np.max(pos[:,0])), np.ceil(np.max(pos[:,1]))]
print('box size = ', box_size)
# %%
new_data = []
for frame in data:
    pos = np.array(frame)[:,1:3]
    points = np.hstack((pos, np.zeros((pos.shape[0],1))))
    voro = freud.locality.Voronoi()
    box_obj = freud.box.Box.from_box(np.array(box_size)+200)
    voro.compute((box_obj, points))
    vv = voro.volumes
    result = np.hstack((pos, np.array(vv).reshape(len(vv),1)))
    new_data.append(result)

#%% add the information you like to append behind x y and voronoi_area
new_result = []
for i, frame in enumerate(data):
    clusters = np.array(frame)[:,5] # which column(s)
    result = np.hstack((new_data[i], clusters.reshape(len(clusters), 1))) 
    new_result.append(result)
new_result = np.array(new_result)


#%% save pickle file
with open('Exp_voronoi.pickle', 'wb') as file:
    pickle.dump(np.array(new_data), file)
