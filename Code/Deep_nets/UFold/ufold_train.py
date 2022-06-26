import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data
import time
import numpy as np


import pdb
import subprocess

# import sys
# sys.path.append('./..')


# from FCN import FCNNet
from Network import U_Net as FCNNet
#from Network3 import U_Net_FP as FCNNet

from ufold.utils import *
from ufold.config import process_config
from ufold.postprocess import postprocess_new as postprocess

from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_merge as Dataset_FCN_merge
from ufold.data_generator import Dataset_Cut_concat_new_merge_multi as Dataset_FCN_merge
import collections

# randomly select one sample from the test set and perform the evaluation
def model_eval(contact_net,test_generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_train = list()
    pos_weight = torch.Tensor([300]).to(device)
    loss_list = []

    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)

    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
        if seq_lens.item() > 1500:
            continue
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        with torch.no_grad():

            pred_contacts = contact_net(seq_embedding_batch)
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
        loss_list.append(float(loss_u))

        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
        map_no_train = (u_no_train > 0.5).float()
        
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        

    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    return(np.average(loss_list), np.average(nt_exact_f1))

def train(contact_net,train_merge_generator,epoches_first,start_time, test_gen):
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    folder = path + "/UFold/models"
    epoch_start = start_time
    epoch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    u_optimizer = optim.Adam(contact_net.parameters())
    steps_done = 0
    for epoch in range(epoches_first):
        contact_net.train()
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            
            pred_contacts = contact_net(seq_embedding_batch)
    
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done=steps_done+1

        val_loss, val_f1 = model_eval(contact_net, test_gen)
        current_time = time.time()
        print('{}, {}, {}, {}, {}'.format(epoch, loss_u, val_loss, val_f1, current_time - epoch_start ))
        epoch_start = current_time
        torch.save(contact_net.state_dict(),  (folder+'/ufold_train_{}.pt').format(epoch))


def main():
    train_time =  time.time()
    BATCH_SIZE = 1
    epoches_first = 70
    print(epoches_first)
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch()
    train_data_list = []
    test_data_list = []
    train_data_list.append(RNASSDataGenerator("data/reduced_saved_data","my_train_set.cPickle"))
    test_data_list.append(RNASSDataGenerator("data/reduced_saved_data","my_test_set.cPickle"))
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    train_merge = Dataset_FCN_merge(train_data_list)
    test_merge = Dataset_FCN_merge(test_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
    test_merge_generator = data.DataLoader(test_merge, **params)
    contact_net = FCNNet(img_ch=17)
    contact_net.to(device)
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    folder = path + "/UFold/models/ufold_train_13.pt"
    contact_net.load_state_dict(torch.load(folder,map_location='cuda'))
    train(contact_net,train_merge_generator,epoches_first,train_time, test_merge_generator)
    train_time_end = time.time()
    print(train_time_end - train_time)

        

RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
main()



