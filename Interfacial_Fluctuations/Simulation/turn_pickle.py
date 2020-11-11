#%%
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

#%%
import pickle

pe = [0,1,2,3,4,5]

for i in range (len(pe)):
        data = read_dump("/Users/mac/Downloads/1200_%s.lammpstrj"%(pe[i]))
        with open ('/Users/mac/Downloads/pickle/1200_%s.pickle'%(pe[i]), 'wb') as file:
            pickle.dump(data, file)
# %%
import numpy as np
import re
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
# %%
data = get_frames_from_xyz('/Users/mac/Downloads/experiment', 6, 0)
# %%
import pickle
with open('/Users/mac/Downloads/experiment.pickle','wb') as file:
    pickle.dump(data,file)
# %%
