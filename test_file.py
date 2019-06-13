'''
Read the notes carefully
1. Note you should only replace 'somewhere' and 'YourFunction' with your own model but WITHOUT any other modifications when you submit this file
2. Run this file on validation set by '' python test.py --testfolder ../Validation_with_labels --num_class 10'', where '../Validation_with_labels' is the path of validation set, 10/107 is number of classes
'''

## import your model: replace 'somewhere' and 'YourFunction'
import torch
from upload.Model import load_model, data_process
import numpy as np
import os
import argparse
import torch.nn.functional as F
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(rnn, classifier, test_data, true_ids, class_num):
    rnn.eval()
    with torch.no_grad():
        test_data = torch.FloatTensor(test_data).to(device)
        encoder_outputs = rnn(test_data)
        out, attn = classifier(encoder_outputs)
        out_softmax = F.softmax(out, dim=1)
        out_softmax_sum = out_softmax.sum(0)
        t = [round(out_softmax_sum.cpu().numpy()[i], 4) for i in range(class_num)]

        id = t.index(max(t))
        return true_ids[id], attn


if __name__ == "__main__":
    # read the test folder
    parser = argparse.ArgumentParser(description='parameters setting')
    parser.add_argument('--testfolder', type=str, default='/dataset/Data10/Validation_with_labels')
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
    time_use = []

    print('Model Loading ...')
    rnn, classifier, couple_length = load_model(args.num_class)

    for filename in files:
        # ignore true_ids.npy
        if filename == 'true_ids.npy':
            continue

        data = np.load(os.path.join(testfolder, filename))

        # data process
        t1 = time.time()
        test_data = data_process(data, couple_length)
        tmp, attn = predict(rnn, classifier, test_data, true_ids, args.num_class)
        time_use.append(time.time()-t1)
        predict_ids.append(tmp)

    # compute the test accuracy
    test_accuracy = np.mean(np.array(predict_ids) == np.array(true_ids))
    print('Test Accuracy: {:.2f}'.format(test_accuracy))
    print('Mean Predict time: {:.3f}'.format(np.array(time_use).sum()/args.num_class))
