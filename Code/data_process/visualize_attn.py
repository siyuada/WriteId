import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re


def flag(i, x, p):
    t = data[i][x][p][:]
    t.append(0) if p == 0 else t.append(1)
    return t


def trans(d1, d2):
    return [d2[0] - d1[0], d2[1] - d1[1], d2[2] * d1[2]]


def get_color():
    c = ['orangered', 'plum', 'cyan', 'steelblue', 'peru',
         'brown', 'teal', 'pink', 'lightsalmon', 'orange',
         'greenyellow', 'gold', 'yellowgreen', 'tomato', 'lemonchiffon',
         'orangered', 'plum', 'cyan', 'steelblue', 'peru',
         'brown', 'teal', 'pink', 'lightsalmon', 'orange',
         'greenyellow', 'gold', 'yellowgreen', 'tomato', 'lemonchiffon']
    return c


color = get_color()

file = 'Data10\Validation_with_labels/005.npy'
data = np.load(file)
file_lst = re.split(r'[/\\]\s*', file)

attn_path = 'RHS_for_attention\Validation_with_labels/005/RHS_len=50'
index = np.load(attn_path + '/index.npy')

# calculate the cumulative sum of points for each character
points_each_cha = []
for character in data:
    points_num = 0
    for stroke in character:
        points_num = points_num + len(stroke)
    points_each_cha.append(points_num)
points_cumsum = np.cumsum(points_each_cha)

# get S
S = []
[S.append(flag(i, x, p)) for i in range(len(data)) for x in range(len(data[i])) for p in range(len(data[i][x]))]

# calculate delta_S
interval = 1
delta_S = [trans(S[i], S[i + interval]) for i in range(len(S) - interval)]
delta_S_len = len(delta_S)

# sample RHS
RHS_num = 30
RHS_len = 50

# plot characters
fig = plt.figure()
cha_num = len(data)
count = 0
x_base = 0
y_base = 9 * 550
for character in data:
    if count == 10:
        count = 0
        y_base = y_base - 550
    x_base = count * 550

    for i in range(len(character)):
        stroke = character[i]
        st = np.array(stroke)
        st = st.T
        st[0] = st[0] + x_base
        st[1] = 500 - st[1] + y_base
        st = st.tolist()
        plt.plot(st[0], st[1], linewidth=1.0, color='k')
    count = count + 1

# plot RHS
thres = 0.00045
for i in range(RHS_num):
    attn_file = attn_path + '/attn_' + str(i).zfill(6) + '.npy'
    attn = np.load(attn_file)

    temp = [attn[0],]
    for j in range(len(attn) - 1):
        temp.append(attn[j + 1] - attn[j])
    temp = np.array(temp)
    attn = temp

    p = len(np.where(attn >= thres)[0])/100
    print('max = %.4f, min = %.4f, thres = %.4f, p = %.2f'%(max(attn), min(attn), thres, p))

    start = index[i]
    end = index[i] + RHS_len
    sample_points = S[start: end + 1]

    # find which character the RHS belongs to
    cha_index = []
    for j in range(RHS_len + 1):
        cha_index.append(min(np.where(points_cumsum > start + j)[0]))

    for j in range(RHS_len):

        x1_base = (cha_index[j] - 10 * int(cha_index[j] / 10)) * 550
        y1_base = (9 - int(cha_index[j] / 10)) * 550

        x2_base = (cha_index[j + 1] - 10 * int(cha_index[j + 1] / 10)) * 550
        y2_base = (9 - int(cha_index[j + 1] / 10)) * 550

        x1 = sample_points[j][0] + x1_base
        y1 = 500 - sample_points[j][1] + y1_base

        x2 = sample_points[j + 1][0] + x2_base
        y2 = 500 - sample_points[j + 1][1] + y2_base

        if attn[j] >= thres:
            if delta_S[start + j][2] == 1:
                line = 'r-'
            else:
                line = 'r--'
        else:
            if delta_S[start + j][2] == 1:
                line = 'b-'
            else:
                line = 'b--'

        plt.plot([x1, x2], [y1, y2], line, linewidth=1.0)
plt.axis('off')
plt.show()

# save RHS plot
RHS_plot_path = 'attn_plot/' + file_lst[0] + '/' + file_lst[1]
if os.path.isdir(RHS_plot_path):
    pass
else:
    os.makedirs(RHS_plot_path)
name = '/' + file_lst[2][: -4] + '_RHS_len=' + str(RHS_len)
fig.savefig(RHS_plot_path + name, dpi=600, bbox_inches='tight')
