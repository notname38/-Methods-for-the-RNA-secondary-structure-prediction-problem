from typing import no_type_check
import load_dataset as ld
#from algorithms import RNAFold
import os
import csv
import numpy as np

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
database = ld.load_sequences(path + "/Databases/archiveII", mode = "ufold")

list_bin_acc = ["Bin Accuracy"]
list_bin_recall = ["Bin Recall"]
list_bin_f = ["Bin F1"]
list_bin_prec = ["Bin Precision"]
list_ex_acc = ["Elem Accuracy"]
list_ex_recall = ["Elem Accuracy"]
list_ex_f = ["Elem Accuracy"]
list_ex_prec = ["Elem Accuracy"]
list_ex_amm = ["Exact match rate"]

list_RNA_Group = ["RNA Group"]
list_lens = ["Length"]

test_list = database


#test_list = test_set

def format_result(list_1, list2, nElements):
    result_list = np.zeros(nElements, dtype=np.int32)
    for i in range(nElements):
        if i in list(range(len(list_1))):
            try:
                #print("aux ", list_1[i], " ", list2[i])
                result_list[list_1[i]] = list2[i] + 1
            except Exception:
                print(i)
                print(list_1)
                for i in range(len(result_list)):
                    print(i, " ", result_list[i])
                break
        
    return list(result_list) 

results = ld.load_bpseq_only(path+"/Code/results_ufold")


check = True
aux = 1
for prediction in results:
    #print("(",aux,"/",len(results),")  File: "+ prediction.name)
    aux = aux + 1
    formatted_prediction = format_result(prediction.structure["base"].tolist(), prediction.structure["base_pair"].tolist(),int(prediction.sequence)) 
    for elem in test_list:
        if prediction.name == elem.name:

            if("N" not in elem.sequence):
                print(prediction.name, elem.name)
                print("Len predicted: ", len(formatted_prediction), ", should be: ", len(elem.structure["base_pair"].tolist()))
                check = check and (len(formatted_prediction) == len(elem.structure["base_pair"].tolist()))
                bin_acc, bin_recall, bin_f, bin_prec, ex_acc, ex_recall, ex_f, ex_prec, ex_amm = elem.comp_evaluate(formatted_prediction)
                #elem.evaluate(prediction.structure["base_pair"].tolist())
                list_bin_acc.append(bin_acc)
                list_bin_recall.append(bin_recall)
                list_bin_f.append(bin_f)
                list_bin_prec.append(bin_prec)
                list_ex_acc.append(ex_acc)
                list_ex_recall.append(ex_recall)
                list_ex_f.append(ex_f)
                list_ex_prec.append(ex_prec)
                list_ex_amm.append(ex_amm)
                list_RNA_Group.append(elem.group)
                list_lens.append(len(elem.sequence))
            break

print("End check: ", check)


with open('saved_stats/my_ufold.csv', mode='w') as results_file:
    writer = csv.writer(results_file, delimiter='\t')
    writer.writerow(["Metric"] + list(range(len(list_bin_acc))))
    writer.writerow(list_bin_acc)
    writer.writerow(list_bin_recall)
    writer.writerow(list_bin_f)
    writer.writerow(list_bin_prec)
    writer.writerow(list_ex_acc)
    writer.writerow(list_ex_recall)
    writer.writerow(list_ex_f)
    writer.writerow(list_ex_prec)
    writer.writerow(list_ex_amm)

results_file.close()

with open('saved_stats/ufold_RNA_Groups.csv', mode='w') as results_file:
    writer = csv.writer(results_file, delimiter='\t')
    writer.writerow(list_RNA_Group)
results_file.close()

with open('saved_stats/ufold_lengths.csv', mode='w') as results_file:
    writer = csv.writer(results_file, delimiter='\t')
    writer.writerow(list_lens)
results_file.close()











