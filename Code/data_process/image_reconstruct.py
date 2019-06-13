import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# generate 100/300 characters simultaneously in a plot for each student

path = glob.glob('Data/Validation/*.npy')

for fd in path:
    data = np.load(fd)
    # fig = plt.figure(figsize=[8, 13])  # 对于每位同学生成一张图片
    fig = plt.figure()
    count = 0
    x_base = 0
    y_base = 0
    for character in data:
        if count == 10:
            count = 0
            y_base = y_base + 550
        x_base = count * 550
        for stroke in character:
            st = np.array(stroke)
            st = st.T
            st[0] = st[0] + x_base
            st[1] = 500 - st[1] + y_base
            st = st.tolist()
            plt.plot(st[0], st[1], linewidth=1.0)
        # plt.show()
        count = count + 1
    plt.axis('off')
    plt.show()

    # 保存图片
    pic_path = 'figure/' + fd[: -15]
    if os.path.isdir(pic_path):
        pass
    else:
        os.makedirs(pic_path)
    pic_name = '\\' + fd[-14: -4] + '.png'
    fig.savefig(pic_path + pic_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
