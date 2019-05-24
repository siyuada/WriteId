'''
Read the notes carefully
1. Note you should only replace 'somewhere' and 'YourFunction' with your own model but WITHOUT any other modifications when you submit this file
2. Run this file on validation set by '' python test.py --testfolder ../Validation_with_labels --num_class 10'', where '../Validation_with_labels' is the path of validation set, 10/107 is number of classes
'''

## import your model: replace 'somewhere' and 'YourFunction'
from somewhere import YourFunction

import numpy as np
import os
import argparse

if __name__ == "__main__":
    
    # read the test folder
    parser = argparse.ArgumentParser(description='parameters setting')
    parser.add_argument('--testfolder', type=str, default=None)
    parser.add_argument('--num_class', type=int, default=10)
    args = parser.parse_args()
    testfolder = args.testfolder
    print(args.testfolder)
    
    # read true_ids
    true_ids = np.load(os.path.join(testfolder, 'true_ids.npy'))
    
    # read files
    files = os.listdir(testfolder)
    files.sort()
    
    # predict the ids
    predict_ids = []
    for filename in files:
        
        # ignore true_ids.npy
        if filename == 'true_ids.npy':
            continue
        filedata = np.load(os.path.join(testfolder, filename))

        ## predict_ids must be a list with student ids of int format
        ## e.g. predict_ids = [2015011414, 2015011431, ..., 2018312484]
        ## replace 'YourFunction' with your own model
        predict_id = YourFunction(filedata, args.num_class)
        predict_ids.append(predict_id)
        
    # compute the test accuracy
    test_accuracy = np.mean(np.array(predict_ids) == np.array(true_ids))
    print 'Test Accuracy: {:.2f}'.format(test_accuracy)
