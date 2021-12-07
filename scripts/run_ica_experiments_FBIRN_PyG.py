import time
from collections import deque
from itertools import chain
import numpy as np
import torch
import os
from scipy import stats

from src.utils import get_argparser
from src.encoders_fMRI import NatureOneCNN

import pandas as pd
import datetime
from src.All_Architecture import combinedModel

from src.pyg_class import Net

from src.graph_the_works_fMRI import the_works_trainer


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):
    start_time=time.time()

    ID = args.script_ID - 1
    JobID = args.job_ID
    print("Job Id = ", str(JobID))

    ID = 4
    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = str(JobID) + '_' + str(ID)

    Name = args.exp + '_FBIRN_' + args.pre_training + 'Graph_NewArch_NoLSTM_WeightedAvg_116_0.25'
    dir = 'run-' + d1 + d2 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    wdb1 = 'wandb_new'
    wpath1 = os.path.join(os.getcwd(), wdb1)


    p = 'UF'
    dir = 'run-2019-09-1223:36:31' + '-' + str(ID) + 'FPT_ICA_COBRE'
    p_path = os.path.join(os.getcwd(), p)
    p_path = os.path.join(p_path, dir) 
    args.p_path = p_path
    # os.mkdir(fig_path)
    # hf = h5py.File('../FBIRN_AllData.h5', 'w')
    tfilename = str(JobID) + 'outputFILENEWONE' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)
    # output_text_file = open(output_path, "w+")
    # writer = SummaryWriter('exp-1')
    ntrials = args.ntrials
    ngtrials = 10
    best_auc = 0.
    best_gain = 0
    current_gain=0
    tr_sub = [15, 25, 50, 75, 100]


    gain = [0.15, 0.25, 0.5, 0.9, 0.9]  # NPT

    sub_per_class = tr_sub[ID]
    current_gain = gain[ID]
    args.gain = current_gain
    sample_x = 100
    sample_y = 160
    subjects = 311
    tc = 160

    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 1 # not dividing time series into windows, using complete window as input
    # samples_per_subject = int((tc - sample_y)+1)
    ntest_samples_perclass = 8
    nval_samples_perclass = 40
    test_start_index = 0
    test_end_index = test_start_index + ntest_samples_perclass
    window_shift = 160

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print('device = ', device)

    n_regions = 116 # number of regions acquired by using the atlas


    with open('../inputs/FBIRN/data.npz', 'rb') as file: ## input data, should be (subjects,n_regions,tc)
        data = np.load(file)


    data[data != data] = 0
    for t in range(311):
        for r in range(116):
            data[t, r, :] = stats.zscore(data[t, r, :])
    finalData = np.zeros((subjects, samples_per_subject, n_regions, sample_y))
    for i in range(subjects):
        for j in range(samples_per_subject):
            finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]



    finalData2 = torch.from_numpy(finalData).float()
    finalData2[finalData2 != finalData2] = 0








    filename = '../inputs/index_array_labelled_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).long()
    index_array = index_array.view(subjects)

    filename = '../inputs/labels_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1


    finalData2 = finalData2[index_array, :, :, :]
    all_labels = all_labels[index_array]

    test_indices = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 142]


    number_of_cv_sets = args.cv_Set
    HC_index, SZ_index = find_indices_of_each_class(all_labels)
    results = torch.zeros(ntrials * number_of_cv_sets, 5)
    adjacency_matrices_learned = torch.zeros(ntrials * number_of_cv_sets, ntest_samples_perclass*2, n_regions * n_regions)
    test_targets = torch.zeros(ntrials * number_of_cv_sets, ntest_samples_perclass * 2)
    test_pred = torch.zeros(ntrials * number_of_cv_sets, ntest_samples_perclass * 2)
    regions_selected = torch.zeros(ntrials * number_of_cv_sets, ntest_samples_perclass * 2 * 23) # 23 is the number of regions left after last pooling layer
    result_counter = 0

    for test_ID in range(number_of_cv_sets):

        test_ID = test_ID + args.start_CV

        print('test Id =', test_ID)

        test_start_index = test_indices[test_ID]
        test_end_index = test_start_index + ntest_samples_perclass
        total_HC_index_tr_val = torch.cat([HC_index[:test_start_index], HC_index[test_end_index:]])
        total_SZ_index_tr_val = torch.cat([SZ_index[:test_start_index], SZ_index[test_end_index:]])

        HC_index_test = HC_index[test_start_index:test_end_index]
        SZ_index_test = SZ_index[test_start_index:test_end_index]

        total_HC_index_tr = total_HC_index_tr_val[:(total_HC_index_tr_val.shape[0] - nval_samples_perclass)]
        total_SZ_index_tr = total_SZ_index_tr_val[:(total_SZ_index_tr_val.shape[0] - nval_samples_perclass)]

        HC_index_val = total_HC_index_tr_val[(total_HC_index_tr_val.shape[0] - nval_samples_perclass):]
        SZ_index_val = total_SZ_index_tr_val[(total_SZ_index_tr_val.shape[0] - nval_samples_perclass):]


        auc_arr = torch.zeros(ngtrials, 1)
        avg_auc = 0.
        for trial in range(ntrials):
                print ('trial = ', trial)

                g_trial=1
                output_text_file = open(output_path, "a+")
                output_text_file.write("CV = %d Trial = %d\r\n" % (test_ID,trial))
                output_text_file.close()
                # Get subject_per_class number of random values
                HC_random = torch.randperm(total_HC_index_tr.shape[0])
                SZ_random = torch.randperm(total_SZ_index_tr.shape[0])
                HC_random = HC_random[:sub_per_class]
                SZ_random = SZ_random[:sub_per_class]


                # Choose the subject_per_class indices from HC_index_val and SZ_index_val using random numbers

                HC_index_tr = total_HC_index_tr[HC_random]
                SZ_index_tr = total_SZ_index_tr[SZ_random]



                tr_index = torch.cat((HC_index_tr, SZ_index_tr))
                val_index = torch.cat((HC_index_val, SZ_index_val))
                test_index = torch.cat((HC_index_test, SZ_index_test))

                tr_index = tr_index.view(tr_index.size(0))
                val_index = val_index.view(val_index.size(0))
                test_index = test_index.view(test_index.size(0))



                tr_eps = finalData2[tr_index.long(), :, :, :]
                val_eps = finalData2[val_index.long(), :, :, :]
                test_eps = finalData2[test_index.long(), :, :, :]

                tr_labels = all_labels[tr_index.long()]
                val_labels = all_labels[val_index.long()]
                test_labels = all_labels[test_index.long()]



                tr_labels = tr_labels.to(device)
                val_labels = val_labels.to(device)
                test_labels = test_labels.to(device)
                tr_eps = tr_eps.to(device)
                val_eps = val_eps.to(device)
                test_eps = test_eps.to(device)





                observation_shape = finalData2.shape
                L=""
                lmax=""

                if args.model_type == "graph_the_works":
                    encoder = NatureOneCNN(observation_shape[2], args)
                    graph_model = Net(n_regions=n_regions)
                    dir = ""


                complete_model = combinedModel(encoder, graph_model, samples_per_subject, gain=current_gain, PT=args.pre_training, exp=args.exp, device=device, oldpath=args.oldpath )
                complete_model.to(device)

                config = {}
                config.update(vars(args))
                config['obs_space'] = observation_shape  # weird hack
                if args.method == "graph_the_works":
                    trainer = the_works_trainer(complete_model, config, device=device, device_encoder=device,
                                                tr_labels=tr_labels,
                                          val_labels=val_labels, test_labels=test_labels, trial=str(trial),
                                                crossv=str(test_ID),gtrial=str(g_trial))
                    results[result_counter][0], results[result_counter][1], results[result_counter][2], \
                    results[result_counter][3], results[result_counter][4], test_targets[result_counter, :], test_pred[result_counter, :] = trainer.train(tr_eps, val_eps, test_eps)
                    result_counter = result_counter + 1
                    # np_fpr = fpr.numpy()
                    # np_tpr = tpr.numpy()
                    # np_threshold = threshold.numpy()
                    # with open('../fMRI/FBIRN/AdjacencyMatrices/fpr' + str(test_ID) + '.npz', 'wb') as filesim:
                    #     np.save(filesim, np_fpr)
                    # with open('../fMRI/FBIRN/AdjacencyMatrices/tpr' + str(test_ID) + '.npz', 'wb') as filesim:
                    #     np.save(filesim, np_tpr)
                    # with open('../fMRI/FBIRN/AdjacencyMatrices/threshold' + str(test_ID) + '.npz', 'wb') as filesim:
                    #     np.save(filesim, np_threshold)


                else:
                    assert False, "method {} has no trainer".format(args.method)

    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")
    # np_adjacency_matrices = adjacency_matrices_learned.numpy()
    np_test_targets = test_targets.numpy()
    np_test_pred = test_pred.numpy()
    # with open('../fMRI/FBIRN/AdjacencyMatrices/adjacencymatrix'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_adjacency_matrices)
    with open('../fMRI/FBIRN/AdjacencyMatrices/testtargets'+str(JobID)+'.npz', 'wb') as filesim:
        np.save(filesim, np_test_targets)
    with open('../fMRI/FBIRN/AdjacencyMatrices/testpred'+str(JobID)+'.npz', 'wb') as filesim:
        np.save(filesim, np_test_pred)

    elapsed = time.time() - start_time
    print('total time = ', elapsed);


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
