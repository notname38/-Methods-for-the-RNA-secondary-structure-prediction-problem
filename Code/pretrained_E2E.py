
import torch.optim as optim
from torch.utils import data

from Deep_nets.e2efold_master.e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from Deep_nets.e2efold_master.e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from Deep_nets.e2efold_master.e2efold.models import Lag_PP_NN, RNA_SS_e2e, Lag_PP_zero, Lag_PP_perturb
from Deep_nets.e2efold_master.e2efold.models import Lag_PP_mixed, ContactAttention_simple
from Deep_nets.e2efold_master.e2efold.utils import *

    # for saving the results

def save_file(folder, file, ct_contact):
    file_path = os.path.join(folder, file)
    first_line = str(len(ct_contact)) + '\t' + file + '\n'
    content = ct_contact.to_csv(header=None, index=None, sep='\t')
    with open(file_path, 'w') as f:
        f.write(first_line+content)

def pretrainE2E(path,path_to_save,data):
    d = 10
    BATCH_SIZE = 1
    OUT_STEP = 100
    LOAD_MODEL = True
    pp_steps = 20
    pp_loss = "f1"
    data_type = "rnastralign_all_600"
    model_type = "att_simple_fix"
    pp_type = '{}_s{}'.format("mixed", pp_steps)
    rho_per_position = "matrix"
    e2e_model_path = path + '/saved_models/e2e/our_e2e_model.pt'
    epoches_third = 10
    evaluate_epi = 1
    step_gamma = 1
    k = 1
    error_num = 0
    # elements_to_save = []
    aux_cont = -1

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed everything for reproduction
    #seed_torch(0)

    seq_len = 600

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

    rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

    if LOAD_MODEL and os.path.isfile(e2e_model_path):
        #print('Loading e2e model...')
        rna_ss_e2e.load_state_dict(torch.load(e2e_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))


    # load test sequences from he custom database

    sequences = list()
    names = list()
    for elem in data:
        if len(elem.sequence) < seq_len:
            sequences.append(elem.sequence)
            names.append(elem.name)

    # make the predictions
    contact_net.eval()
    lag_pp_net.eval()
    final_result_dict = dict()
    seq_batch = np.array_split(np.array(sequences), 
        math.ceil(len(sequences)/BATCH_SIZE))

    ct_list = list()
    for seqs in seq_batch:
        aux_cont = aux_cont + 1
        #print("(",aux_cont,"/",len(names),") E2E Evaluations. Sequence: ", names[aux_cont])
        try:
            seq_embeddings =  list(map(seq_encoding, seqs))
            seq_embeddings = list(map(lambda x: padding(x, seq_len), seq_embeddings))
        except KeyError:
            error_num = error_num + 1
            # print("               ")
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("Error number: ", error_num)
            # print("Error info : Key error in seq_batch maping")
            # print("Seqs: ", seqs)
            # print("out of: ", aux_cont)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("               ")
            continue
        
        # elements_to_save.append(aux_cont)
        seq_embeddings = np.array(seq_embeddings)
        seq_lens = torch.Tensor(np.array(list(map(len, seqs)))).int()
        seq_embedding_batch = torch.Tensor(seq_embeddings).float().to(device)
        state_pad = torch.zeros(1,2,2).to(device)

        PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, 
                seq_embedding_batch, state_pad)
            a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

        # the learning pp result
        final_pred = (a_pred_list[-1].cpu()>0.5).float()

        for i in range(final_pred.shape[0]):
            ct_tmp = contact2ct(final_pred[i].cpu().numpy(), 
                seq_embeddings[i], seq_lens.numpy()[i])
            ct_list.append(ct_tmp)

    for i in range(len(names)):
        save_file(path_to_save, names[i]+'.ct', ct_list[i])

#    for i in range(len(ct_list)):
#        save_file(path_to_save, names[elements_to_save[i]]+'.ct', ct_list[i])