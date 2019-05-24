import numpy as np
import glob
import os
import time

seqence_length = 100
train_num = 1000
def findFiles(path): return glob.glob(path)

def flag(i,x,p):
    data[i][x][p].append(0) if p == 0 or p == (len(data[i][x]) - 1) else data[i][x][p].append(1)
    return data[i][x][p]

def trans(d1,d2):
    return [d1[0]-d2[0], d1[1]-d2[1], d1[2]*d2[2]]


#print(findFiles('Data10/Train/*.npy'))
'''
path = findFiles('Data10/Train/*.npy')

for file in range(10):
    data = np.load(path[file])
    ## min 位置的长度 22385
    #t1 = time.time()
    cnt = [len(data[i][x]) for i in range(data.shape[0]) for x in range(len(data[i])) ]
    #t2 = time.time()-t1
    #print(t2)
    #print(cnt)
    print(np.sum(np.array(cnt)))


    new_data = []
    [new_data.append(flag(i,x,p)) for i in range(data.shape[0]) for x in range(len(data[i])) for p in range(len(data[i][x])) ]


    RHS = [trans(new_data[i],new_data[i+1]) for i in range(len(new_data)-1)]
    RHS = np.array(RHS)

    for i in range(train_num):
        index = np.random.permutation(len(RHS))
        save_data = RHS[index[0:seqence_length]]
        file_name = './Task1/Train10/' + str(file)
        name = file_name + '/' + str(i).zfill(6) + '.npy'

        if os.path.exists(file_name):
            pass
        else:
            os.mkdir(file_name)
        np.save(name, save_data)
'''

seqence_length = 100
train_num = 600
path = findFiles('Data10/Validation/*.npy')
print(path)
for file in range(10):
    data = np.load(path[file])
    ## min 位置的长度 7469
    #t1 = time.time()
    cnt = [len(data[i][x]) for i in range(data.shape[0]) for x in range(len(data[i])) ]
    #t2 = time.time()-t1
    #print(t2)
    #print(cnt)
    print(np.sum(np.array(cnt)))


    new_data = []
    [new_data.append(flag(i,x,p)) for i in range(data.shape[0]) for x in range(len(data[i])) for p in range(len(data[i][x])) ]


    RHS = [trans(new_data[i],new_data[i+1]) for i in range(len(new_data)-1)]
    RHS = np.array(RHS)

    for i in range(train_num):
        index = np.random.permutation(len(RHS))
        save_data = RHS[index[0:seqence_length]]
        file_name = './Task1/Test10/' + str(file)
        name = file_name + '/' + str(i).zfill(6) + '.npy'

        if os.path.exists(file_name):
            pass
        else:
            os.mkdir(file_name)
        np.save(name, save_data)

print(1)

