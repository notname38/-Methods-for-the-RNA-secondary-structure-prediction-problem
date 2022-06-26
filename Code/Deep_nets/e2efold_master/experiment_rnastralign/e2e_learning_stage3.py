import torch.optim as optim
from torch.utils import data

import time
import os

from e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.models import Lag_PP_NN, RNA_SS_e2e, Lag_PP_zero, Lag_PP_perturb
from e2efold.models import Lag_PP_mixed, ContactAttention_simple
from e2efold.common.utils import *
from e2efold.common.config import process_config
from e2efold.evaluation import all_test_only_e2e

args = get_args()

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
folder = path + "/models_ckpt"

config_file = args.config

config = process_config(config_file)

os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu

d = config.u_net_d
BATCH_SIZE = config.BATCH_SIZE
OUT_STEP = config.OUT_STEP
LOAD_MODEL = config.LOAD_MODEL
pp_steps = config.pp_steps
pp_loss = config.pp_loss
data_type = config.data_type
model_type = config.model_type
pp_type = '{}_s{}'.format(config.pp_model, pp_steps)
rho_per_position = config.rho_per_position
model_path = '../models_ckpt/e2efold_50_epoch_1.pt'
pp_model_path = '../models_ckpt/lag_pp_{}_{}_{}_position_{}.pt'.format(
pp_type, data_type, pp_loss,rho_per_position)
# The unrolled steps for the upsampling model is 10
# e2e_model_path = '../models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}_upsampling.pt'.format(model_type,
#     pp_type,d, data_type, pp_loss,rho_per_position)
e2e_model_path = '../models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(model_type,
    pp_type,d, data_type, pp_loss,rho_per_position)
epoches_third = config.epoches_third
evaluate_epi = config.evaluate_epi
step_gamma = config.step_gamma
k = config.k

train_time =  time.time()
steps_done = 0
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seed everything for reproduction
seed_torch(0)


# for loading data
# loading the rna ss data, the data has been preprocessed
# 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
from e2efold.data_generator import RNASSDataGenerator, Dataset
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')

train_data = RNASSDataGenerator('../data/reduced_saved_data/', 'train')
val_data = RNASSDataGenerator('../data/reduced_saved_data/', 'val')
# test_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_no_redundant')
test_data = RNASSDataGenerator('../data/reduced_saved_data/', 'test')


seq_len = train_data.data_y.shape[-2]
#print('Max seq length ', seq_len)

# using the pytorch interface to parallel the data generation and model training
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)

# only for save the final results
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 6,
          'drop_last': False}
test_set = Dataset(test_data)
test_generator = data.DataLoader(test_set, **params)


if model_type =='test_lc':
    contact_net = ContactNetwork_test(d=d, L=seq_len).to(device)
if model_type == 'att6':
    contact_net = ContactAttention(d=d, L=seq_len).to(device)
if model_type == 'att_simple':
    contact_net = ContactAttention_simple(d=d, L=seq_len).to(device)    
if model_type == 'att_simple_fix':
    contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len, 
        device=device).to(device)
if model_type == 'fc':
    contact_net = ContactNetwork_fc(d=d, L=seq_len).to(device)
if model_type == 'conv2d_fc':
    contact_net = ContactNetwork(d=d, L=seq_len).to(device)

# need to write the class for the computational graph of lang pp
if pp_type=='nn':
    lag_pp_net = Lag_PP_NN(pp_steps, k).to(device)
if 'zero' in pp_type:
    lag_pp_net = Lag_PP_zero(pp_steps, k).to(device)
if 'perturb' in pp_type:
    lag_pp_net = Lag_PP_perturb(pp_steps, k).to(device)
if 'mixed'in pp_type:
    lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position).to(device)

if LOAD_MODEL and os.path.isfile(model_path):
    contact_net.load_state_dict(torch.load(model_path))
if LOAD_MODEL and os.path.isfile(pp_model_path):
    lag_pp_net.load_state_dict(torch.load(pp_model_path))


rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

if LOAD_MODEL and os.path.isfile(e2e_model_path):
    rna_ss_e2e.load_state_dict(torch.load(e2e_model_path))

        
all_optimizer = optim.Adam(rna_ss_e2e.parameters())

# for 5s
# pos_weight = torch.Tensor([100]).to(device)
# for length as 600
pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight = pos_weight)
criterion_mse = torch.nn.MSELoss(reduction='sum')


def per_family_evaluation(rna_ss_e2e):
    contact_net.eval()
    lag_pp_net.eval()
    result_pp = list()
    loss_list = []
    f1_pp = list()
    seq_lens_list = list()

    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens in val_generator:
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        state_pad = torch.zeros(contacts.shape).to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            PE_batch = get_pe(seq_lens, seq_len).float().to(device)
            # the end to end model
            pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, 
                seq_embedding_batch, state_pad)

            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

            # Compute loss, consider the intermidate output
            if pp_loss == "l2":
                loss_a = criterion_mse(
                    a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*criterion_mse(
                        a_pred_list[i]*contact_masks, contacts_batch)
                mse_coeff = 1.0/(seq_len*pp_steps)

            if pp_loss == 'f1':
                loss_a = f1_loss(a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*f1_loss(
                        a_pred_list[i]*contact_masks, contacts_batch)            
                mse_coeff = 1.0/pp_steps

            loss_a = mse_coeff*loss_a
            loss = loss_u + loss_a

        loss_list.append(float(loss))

        # the learning pp result
        final_pred = (a_pred_list[-1].cpu()>0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp += result_tmp


        f1_tmp = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp += f1_tmp
        seq_lens_list += list(seq_lens)

    pp_exact_p,pp_exact_r,pp_exact_f1 = zip(*result_pp)

    return(np.average(loss_list),np.sum(np.array(pp_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list)))
    

# There are three steps of training
# Last, joint fine tune
# final steps
if not args.test:
    all_optimizer.zero_grad()
    for epoch in range(epoches_third):
        epoch_start = train_time
        rna_ss_e2e.train()
        for contacts, seq_embeddings, matrix_reps, seq_lens in train_generator:
        # for train_step  in range(1000): 
        #     contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(train_generator))

            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            matrix_reps_batch = torch.unsqueeze(
                torch.Tensor(matrix_reps.float()).to(device), -1)
            
            contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
            # padding the states for supervised training with all 0s
            state_pad = torch.zeros([matrix_reps_batch.shape[0], 
                seq_len, seq_len]).to(device)

            PE_batch = get_pe(seq_lens, seq_len).float().to(device)
            # the end to end model
            pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, 
                seq_embedding_batch, state_pad)

            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

            # Compute loss, consider the intermidate output
            if pp_loss == "l2":
                loss_a = criterion_mse(
                    a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*criterion_mse(
                        a_pred_list[i]*contact_masks, contacts_batch)
                mse_coeff = 1.0/(seq_len*pp_steps)

            if pp_loss == 'f1':
                loss_a = f1_loss(a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*f1_loss(
                        a_pred_list[i]*contact_masks, contacts_batch)            
                mse_coeff = 1.0/pp_steps

            loss_a = mse_coeff*loss_a
            loss = loss_u + loss_a

            # Optimize the model, we increase the batch size by 100 times
            loss.backward()
            if steps_done % 30 ==0:
                all_optimizer.step()
                all_optimizer.zero_grad()
            steps_done=steps_done+1

        val_loss, val_f1 = per_family_evaluation(rna_ss_e2e)
        current_time = time.time()
        print('{}, {}, {}, {}, {}'.format(epoch, loss_u, val_loss, val_f1, current_time - epoch_start ))
        epoch_start = current_time
        torch.save(rna_ss_e2e.state_dict(), (folder+'e2efold_full_train_{}.pt').format(epoch))

train_time_end = time.time()
print("Elapsed time: ", train_time_end - train_time)


#all_test_only_e2e(test_generator, contact_net, lag_pp_net, device, test_data)







