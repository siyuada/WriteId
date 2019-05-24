import numpy as np
#str1 = 'Data\Train10' + '\' + str(2015011414) + '.npy'
a = np.load("Data10/Train10/2016310874.npy")
print(a.shape)
print(len(a[9])) #第一个字里包含的笔画数
for x in a[9]:
    print(np.array(x).shape)#每个笔画的位置数

## 每个包含300个list
# 每个list结构：n个笔画数，每个笔画中间经过的xy坐标(提笔落笔)

#测试
a = np.load("Data10/Validation/2016310874.npy")
print(a.shape)
print(len(a[0])) #第一个字里包含的笔画数
for x in a[0]:
    print(np.array(x).shape)#每个笔画的位置数

# Validation Characters10 - 字的照片

# Validation with labels 名称为类别
a = np.load("Data10/Validation_with_labels/true_ids.npy")
print(a)



