import _pickle as pickle

import torch.optim as optim
from torch.utils import data
import time
import os

from e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.models import ContactAttention_simple
from e2efold.common.utils import *
from e2efold.common.config import process_config
from e2efold.postprocess import postprocess

args = get_args()

config_file = args.config

config = process_config(config_file)


os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu

d = config.u_net_d
BATCH_SIZE = config.batch_size_stage_1
#OUT_STEP = config.OUT_STEP
LOAD_MODEL = config.LOAD_MODEL
pp_steps = config.pp_steps
data_type = config.data_type
model_type = config.model_type
model_path = '../models_ckpt/mymodel.pt'
epoches_first = config.epoches_first
evaluate_epi = config.evaluate_epi_stage_1

train_loss = []
f1_score_validation = []
train_time =  time.time()


steps_done = 0
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch()

# for loading data
# loading the rna ss data, the data has been preprocessed
# 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
from e2efold.data_generator import RNASSDataGenerator, Dataset
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
folder = path + "/models_ckpt"

train_data = RNASSDataGenerator('../data/reduced_saved_data/', 'train')
val_data = RNASSDataGenerator('../data/reduced_saved_data/', 'val')
# test_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_no_redundant')
test_data = RNASSDataGenerator('../data/reduced_saved_data/', 'test')

seq_len = train_data.data_y.shape[-2]

# using the pytorch interface to parallel the data generation and model training
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)

test_set = Dataset(test_data)
test_generator = data.DataLoader(test_set, **params)

# seq_len =500

# store the intermidiate activation

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

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

# contact_net.conv1d2.register_forward_hook(get_activation('conv1d2'))

#if LOAD_MODEL and os.path.isfile(model_path):
#    contact_net.load_state_dict(torch.load(model_path))


u_optimizer = optim.Adam(contact_net.parameters())

# for 5s
# pos_weight = torch.Tensor([100]).to(device)
# for length as 600
pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight = pos_weight)


# randomly select one sample from the test set and perform the evaluation
def model_eval():
    """ device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.eval()
    contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(val_generator))
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
    matrix_reps_batch = torch.unsqueeze(
        torch.Tensor(matrix_reps.float()).to(device), -1)

    state_pad = torch.zeros([matrix_reps_batch.shape[0], 
        seq_len, seq_len]).to(device)
    PE_batch = get_pe(seq_lens, seq_len).float().to(device)

    with torch.no_grad():
        contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
        pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
        loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

    return(loss_u) """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.eval()
    result_no_train = list()
    seq_lens_list = list()
    loss_list=[]
    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens in val_generator:
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        matrix_reps_batch = torch.unsqueeze(
            torch.Tensor(matrix_reps.float()).to(device), -1)

        state_pad = torch.zeros([matrix_reps_batch.shape[0], 
            seq_len, seq_len]).to(device)

        PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, 
                seq_embedding_batch, state_pad)
            contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
        loss_list.append(float(loss_u))

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
        map_no_train = (u_no_train > 0.5).float()
        result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        seq_lens_list += list(seq_lens)

    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    return(np.average(loss_list),np.sum(np.array(nt_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list)))

# There are three steps of training
# step one: train the u net
print(epoches_first)
epoch_start = train_time
for epoch in range(epoches_first):
    contact_net.train()
    # num_batches = int(np.ceil(train_data.len / BATCH_SIZE))
    # for i in range(num_batches):

    for contacts, seq_embeddings, matrix_reps, seq_lens in train_generator:
        # contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(train_generator))

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        matrix_reps_batch = torch.unsqueeze(
            torch.Tensor(matrix_reps.float()).to(device), -1)

        # padding the states for supervised training with all 0s
        state_pad = torch.zeros([matrix_reps_batch.shape[0], 
            seq_len, seq_len]).to(device)


        PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
        pred_contacts = contact_net(PE_batch, 
            seq_embedding_batch, state_pad)

        # Compute loss
        loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
       
        
            #train_loss.append(loss_u)
            #model_eval()


        # Optimize the model
        u_optimizer.zero_grad()
        loss_u.backward()
        u_optimizer.step()
        steps_done=steps_done+1

    val_loss, val_f1 = model_eval()
    current_time = time.time()
    print('{}, {}, {}, {}, {}'.format(epoch, loss_u, val_loss, val_f1, current_time - epoch_start ))
    epoch_start = current_time
    torch.save(contact_net.state_dict(),  (folder+'/e2efold_train_{}.pt').format(epoch))
        
    
train_time_end = time.time()
print("Elapsed time: ", train_time_end - train_time)

