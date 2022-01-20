import numpy as np
import mdtraj as md
from sklearn import preprocessing



def read_traj(file_nc,file_top):
    
    traj = md.load(file_nc,top=file_top)[:100]
    
    return traj


def dis_bias(traj1, traj2):
    
    traj_after = md.Trajectory.superpose(traj1+traj2, traj1+traj2, frame=0)
    traj_after1 = traj_after[:len(traj1)]
    traj_after2 = traj_after[len(traj1):]
    
    return traj_after1, traj_after2



trs = [[2.0413690, -0.5649464, -0.3446944], 
       [-0.9692660, 1.8760108, 0.0415560],
       [0.0134474, -0.1183897, 1.0154096]]

# trs = [[2.493478, -0.931556, -0.402658], 
#        [-0.829621, -0.829621, 0.023600],
#        [0.035842, -0.076161, 0.956927]]

def xyz_to_rgb(xyz, trs=trs):

    # rgb = trs * xyz
    rgb = []
    for i in range(3):
        tmp = xyz[0] * trs[i][0] + xyz[1] * trs[i][1] + xyz[2] * trs[i][2]
        rgb.append(tmp)
        
    return rgb


def sca_xyz(xyz, min=0, max=255):

    x = np.array(xyz)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(min,max))
    x_minmax = min_max_scaler.fit_transform(x)
    x_minmax2 = []
    for item in x_minmax:
        x_minmax2.append([int(item[0]),int(item[1]),int(item[2])])
        
    return x_minmax2


def flt_to_int(tup):
    for i in range(len(tup)):
        tup[i] = int(tup[i])*10
    
    return tup


def rgb_to_hex(rgb):
    string = ''
    digit = list(map(str, range(10))) + list("ABCDEF")
    for i in rgb:
        a1 = i // 16
        a2 = i % 16
        string += digit[a1] + digit[a2]
    return string


def traj_to_hex(traj):

    traj = traj.xyz
    
    traj1 = []
    # xyz->rgb）
    for item in traj:
        item1 = []
        for atom in item:
            atom1 = xyz_to_rgb(atom)
            item1.append(atom1)
        traj1.append(item1)
    # print('traj ready')
    
    traj2 = []
    #（0,255）
    for item in traj1:
        item2 = sca_xyz(item, min=0, max=255)
        traj2.append(item2)
    # print(2)
    pixel_map = traj2
    
    traj_hex = []
    for item in traj2:
        tep = []
        for atom in item:
            atom1 = rgb_to_hex(tuple(atom))
            rgb_dec = int(atom1.upper(), 16)
            tep.append([rgb_dec])
        traj_hex.append(tep)
        
        
    return traj_hex, pixel_map


def load_traj(file0_1, file1_1, file0_2, file1_2):
    # =============================================================================
    # traj0
    file0_traj = read_traj(file0_1, file0_2)
    # traj1
    file1_traj = read_traj(file1_1, file1_2)

    print('Info of traj_anta:')
    print(file0_traj)
    print('Info of traj_actv:')
    print(file1_traj)
    print("Pixel-representation Start.")
    # =============================================================================

    # =============================================================================
    file0_traj, file1_traj = dis_bias(file0_traj, file1_traj)
    # print(len(file0_traj.xyz),len(file0_traj.xyz[0]))
    # =============================================================================

    # ============================================================================
    import time
    start = time.time()

    traj0_hex, pixel_map0 = traj_to_hex(file0_traj)
    traj1_hex, pixel_map1 = traj_to_hex(file1_traj)

    end = time.time()
    print('time:', end-start,'s')
    # =============================================================================

    return traj0_hex, traj1_hex, pixel_map0, pixel_map1


def traj_to_pic(traj0_hex, traj1_hex, pixel0, pixel1):
    # size*size
    atom_n = len(traj0_hex[0])
    import math
    size = math.ceil(atom_n**0.5)
    traj0_pic = []
    for item in range(len(traj0_hex)):
        for ti in range(size*size-atom_n):
            traj0_hex[item].append([0])
        pic = []
        for i in range(size):
            line = []
            for j in range(size):
                line.append(traj0_hex[item][i*size+j])
            pic.append(line)
        traj0_pic.append(pic)
        
    traj1_pic = []
    for item in range(len(traj1_hex)):
        for ti in range(size*size-atom_n):
            traj1_hex[item].append([0])
        pic = []
        for i in range(size):
            line = []
            for j in range(size):
                line.append(traj1_hex[item][i*size+j])
            pic.append(line)
        traj1_pic.append(pic)
    
    pixel_map0 = []
    for item in range(len(pixel0)):
        for ti in range(size*size-atom_n):
            pixel0[item].append([0,0,0])
        pic = []
        for i in range(size):
            line = []
            for j in range(size):
                line.append(pixel0[item][i*size+j])
            pic.append(line)
        pixel_map0.append(pic)
        
    pixel_map1 = []
    for item in range(len(pixel1)):
        for ti in range(size*size-atom_n):
            pixel1[item].append([0,0,0])
        pic = []
        for i in range(size):
            line = []
            for j in range(size):
                line.append(pixel1[item][i*size+j])
            pic.append(line)
        pixel_map1.append(pic)
    
    
    return traj0_pic, traj1_pic, pixel_map0, pixel_map1


def data_split(traj0_pic, traj1_pic, pixel_map0, pixel_map1):
    from sklearn.model_selection import train_test_split
    X_all = []
    y_all = []
    pixel_map_all = []
    for i in range(len(traj0_pic)):
        X_all.append(traj0_pic[i])
        pixel_map_all.append(pixel_map0[i])
        y_all.append(0)
    for i in range(len(traj1_pic)):
        X_all.append(traj1_pic[i])
        pixel_map_all.append(pixel_map1[i])
        y_all.append(1)

#     X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.8)

#     print('set1:', len(X_train), ', set2:', len(X_test))
#     print('ok')
    
#     return X_train, y_train, X_test, y_test
    return X_all, y_all, pixel_map_all

def data_split_dnn(traj0_pic, traj1_pic):
    from sklearn.model_selection import train_test_split

    # traj0-0, traj1-1
    X_all = []
    y_all = []
    for i in range(len(traj0_pic)):
        X_all.append(traj0_pic[i])
        y_all.append(0)
    for i in range(len(traj1_pic)):
        X_all.append(traj1_pic[i])
        y_all.append(1)

#     X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.8)

#     print('set1:', len(X_train), ', set2:', len(X_test))
#     print('ok')
    
#     return X_train, y_train, X_test, y_test
    return X_all, y_all, pixel_map_all


def traj_pre(file0_1, file1_1, file0_2, file1_2):
    import numpy as np

    traj0_hex, traj1_hex, pixel0, pixel1 = load_traj(file0_1, file1_1, file0_2, file1_2)
    traj0_pic, traj1_pic, pixel_map0, pixel_map1 = traj_to_pic(traj0_hex, traj1_hex, pixel0, pixel1)
#     X_train, y_train, X_test, y_test = data_split(traj0_pic, traj1_pic)
    X_all, y_all, pixel_map_all = data_split(traj0_pic, traj1_pic, pixel_map0, pixel_map1)
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)
    X_all = np.array(X_all)
    y_all = np.array(y_all)
#     return X_train, y_train, X_test, y_test
    return X_all, y_all, np.array(pixel_map_all)
