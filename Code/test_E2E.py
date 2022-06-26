import load_dataset as ld
#from algorithms import RNAFold
import pretrained_E2E as e2e 
import os
import csv
import time

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
database = ld.load_sequences(path + "/Databases/archiveII")

test_list = database

time_1 =  time.time()
e2e.pretrainE2E(path+"/Code", path+"/Code/results_e2e", test_list)
time_2 =  time.time()
print(time_2 - time_1)
results = ld.load_ct_only(path+"/Code/results_e2e")

#check = True
for prediction in results:
    for elem in test_list:
        if prediction.name == elem.name:
            if("N" not in elem.sequence):
                #print("Len predicted: ", len(prediction.structure["base_pair"].tolist()), ", should be: ", len(elem.structure["base_pair"].tolist()))
                #check = check and (len(prediction.structure["base_pair"].tolist()) == len(elem.structure["base_pair"].tolist()))
                bin_acc, bin_recall, bin_f, bin_prec, ex_acc, ex_recall, ex_f, ex_prec, ex_amm = elem.comp_evaluate(prediction.structure["base_pair"].tolist())
                # elem.evaluate(prediction.structure["base_pair"].tolist())
                # list_bin_acc.append(bin_acc)
                # list_bin_recall.append(bin_recall)
                # list_bin_f.append(bin_f)
                # list_bin_prec.append(bin_prec)
                # list_ex_acc.append(ex_acc)
                # list_ex_recall.append(ex_recall)
                # list_ex_f.append(ex_f)
                # list_ex_prec.append(ex_prec)
                # list_ex_amm.append(ex_amm)
                print('{}, {}, {}, {}, {}, {}, {}, {}, {}. {}'.format(      elem.group, bin_acc,    bin_recall,
                                                                            bin_f,      bin_prec,   ex_acc,
                                                                            ex_recall,  ex_f,       ex_prec,
                                                                            ex_amm))

            break

#print("End check: ", check)


# with open('saved_stats/e2e_RNA_Groups.csv', mode='w') as results_file:
#     writer = csv.writer(results_file, delimiter='\t')
#     #writer.writerow(["Metric"] + list(range(len(list_bin_acc))))
#     # writer.writerow(list_bin_acc)
#     # writer.writerow(list_bin_recall)
#     # writer.writerow(list_bin_f)
#     # writer.writerow(list_bin_prec)
#     # writer.writerow(list_ex_acc)
#     # writer.writerow(list_ex_recall)
#     # writer.writerow(list_ex_f)
#     # writer.writerow(list_ex_prec)
#     # writer.writerow(list_ex_amm)
#     writer.writerow(list_RNA_Group)















