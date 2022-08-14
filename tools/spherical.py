import numpy as np
from numpy import linalg as LA

def normalization(data):
    _max = np.max(data)
    _min = np.min(data)
    return (data - _min)/(_max - _min)

def get_Spherical_coordinate(batch_data,normalized = False):
    R = LA.norm(batch_data,ord=2,axis=1)
    # print(batch_data)

    x = batch_data[:,0]
    y = batch_data[:,1]
    z = batch_data[:,2]
    # num = batch_data[:,3]
    Theta = np.arccos(z/R)      # [0,pi]
    Phi = np.arctan2(y,x)       # [-pi,pi]
    if normalized:
        R1 = normalization(R)
        Theta = Theta / np.pi
        Phi = (Phi + np.pi) / (2*np.pi)
    s_coor = np.stack([R1,Theta,Phi],axis=1)
    # print(R.shape)
    return s_coor,R