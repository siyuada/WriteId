import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# generate all characters separately for each student

file_paths = ['Data/Validation/*.npy', 'Data/Train/*.npy',
              'Data/Validation_with_labels/*.npy', 'Data10/Validation/*.npy',
              'Data10/Train/*.npy', 'Data10/Validation_with_labels/*.npy']

for p in file_paths:
    path = glob.glob(p)

    for fd in path:
        data = np.load(fd)

        pic_path = 'figure/' + fd[: -4]
        if os.path.isdir(pic_path):
            pass
        else:
            os.makedirs(pic_path)
        count = 1

        for character in data:
            fig = plt.figure()
            for stroke in character:
                st = np.array(stroke)
                st = st.T
                st[1] = 500 - st[1]
                st = st.tolist()
                plt.plot(st[0], st[1], linewidth=20.0)
            plt.axis('off')
            # plt.show()

            pic_name = '\\' + str(count) + '.png'
            fig.savefig(pic_path + pic_name, dpi=50, bbox_inches='tight')
            plt.close(fig)
            count = count + 1
