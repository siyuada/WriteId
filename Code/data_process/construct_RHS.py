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


file_paths = ['Data/Validation/*.npy', 'Data/Train/*.npy',
              'Data/Validation_with_labels/*.npy', 'Data10/Validation/*.npy',
              'Data10/Train/*.npy', 'Data10/Validation_with_labels/*.npy']
color = get_color()

for p in file_paths:
    print(p)
    path = glob.glob(p)
    path = sorted(path)

    for file in path:
        if file == 'true_ids.npy':
            continue
        data = np.load(file)
        file_lst = re.split(r'[/\\]\s*', file)
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

        # # copy the last point for each stroke
        # for i in range(len(data)):
        #     character = data[i]
        #     for j in range(len(character)):
        #         stroke = character[j]
        #         last_point = stroke[-1][:]
        #         stroke.append(last_point)
        #
        # # S with the last point of each stroke repeated
        # S_extend = []
        # [S_extend.append(flag(i, x, p)) for i in range(len(data)) for x in range(len(data[i])) for p in
        #  range(len(data[i][x]))]

        # calculate delta_S
        interval = 1
        delta_S = [trans(S[i], S[i + interval]) for i in range(len(S) - interval)]
        delta_S_len = len(delta_S)

        # sample RHS
        RHS_num = 1
        for RHS_len in [50, 80, 100]:
            # index = np.random.randint(0, delta_S_len - RHS_len, RHS_num)
            index = [150, ]

            # # generate RHS
            # RHS = []
            # for i in range(RHS_num):
            #     start = index[i]
            #     end = index[i] + RHS_len
            #     RHS.append(delta_S[start: end] + [[0, 0, 0]] * (100 - RHS_len))

            # # save RHS
            # RHS_path = 'RHS_for_attention/' + file_lst[0] + '/' + file_lst[1] + '/' + file_lst[2][: -4] \
            #            + '/RHS_len=' + str(RHS_len)
            # if os.path.isdir(RHS_path):
            #     pass
            # else:
            #     os.makedirs(RHS_path)
            #
            # for i in range(RHS_num):
            #     file_name = RHS_path + '/' + str(i).zfill(6) + '.npy'
            #     np.save(file_name, RHS[i])
            # np.save(RHS_path + '/index.npy', index)

            # plot characters
            fig = plt.figure(figsize=[8, 13])
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
                    plt.plot(st[0], st[1], linewidth=1.0, color=color[i])
                count = count + 1

            # plot RHS
            for i in range(RHS_num):
                start = index[i]
                end = index[i] + RHS_len
                sample_points = S[start: end + 1]

                # find which character the RHS belongs to
                cha_index = []
                for j in range(len(sample_points)):
                    cha_index.append(min(np.where(points_cumsum > start + j)[0]))

                for j in range(len(sample_points) - 1):

                    x1_base = (cha_index[j] - 10 * int(cha_index[j] / 10)) * 550
                    y1_base = (9 - int(cha_index[j] / 10)) * 550

                    x2_base = (cha_index[j + 1] - 10 * int(cha_index[j + 1] / 10)) * 550
                    y2_base = (9 - int(cha_index[j + 1] / 10)) * 550

                    x1 = sample_points[j][0] + x1_base
                    y1 = 500 - sample_points[j][1] + y1_base

                    x2 = sample_points[j + 1][0] + x2_base
                    y2 = 500 - sample_points[j + 1][1] + y2_base

                    if delta_S[start + j][2] == 1:
                        line = 'k-'
                    else:
                        line = 'b--'

                    plt.plot([x1, x2], [y1, y2], line, linewidth=1.0)
            plt.axis('off')

            # save RHS plot
            RHS_plot_path = 'RHS_plot/' + file_lst[0] + '/' + file_lst[1]
            if os.path.isdir(RHS_plot_path):
                pass
            else:
                os.makedirs(RHS_plot_path)
            name = '/' + file_lst[2][: -4] + '_RHS_len=' + str(RHS_len)
            fig.savefig(RHS_plot_path + name, dpi=600, bbox_inches='tight')
