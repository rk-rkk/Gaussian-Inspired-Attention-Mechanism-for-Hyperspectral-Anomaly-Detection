import os
import time

import numpy as np
from scipy.io import savemat

from GCL_fun import GCL_main


if __name__ == '__main__':
    data_dir = './data/'
    save_dir = './result/'

    directions = ['AVIRIS-2-fog']
    # directions = ['AVIRIS-1-sunlight', 'abu-urban-3-shadow']
    # directions = ['AVIRIS-1', 'AVIRIS-2', 'abu-urban-3']
    # dirlist = os.listdir(data_dir)
    # directions = [file for file in dirlist if file.endswith('.mat')]
    num_iter = 31
    for direction in directions:
        name = direction.split('.')[0]
        print(name)

        AUCS = np.zeros(num_iter)
        LOSSES = np.zeros(num_iter)
        end_iters = np.zeros(num_iter) != 0
        for i in range(num_iter):
            start = time.perf_counter()
            auc, loss, end_iter, detection = GCL_main(data_dir + name)  # end_iter: True: mean_loss < thres
            end = time.perf_counter()

            loss = loss.detach().cpu().squeeze().numpy()
            AUCS[i] = auc
            LOSSES[i] = loss
            end_iters[i] = end_iter
            print(i + 1, '\tauc: ', auc, '\tloss: ',
                  loss, "\truntime: ", end - start, "\ttype: ", end_iter)
            savemat(save_dir + name + '_GCL_' + str(i+1) + '.mat',
                    {'auc': auc, 'detection': detection, 'loss': loss, 'type': end_iters})
            
        savemat(save_dir + name + '_GCL_all.mat',
                {'AUCS': AUCS, 'LOSSES': LOSSES, 'type': end_iters})
