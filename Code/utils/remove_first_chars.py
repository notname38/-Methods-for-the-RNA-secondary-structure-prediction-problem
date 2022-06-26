import os
import random

def rewrite_file(file, name):
    namefile = file
    aux = 0
    f = open(namefile)
    lines = f.readlines()
    f.close()

    new = "fixed_files/" + name + ".csv"
    f = open(new, "w")
    for line in lines:
        if aux == 0:
            aux = aux + 1
            continue
        if aux <= 9:
            end_line = line[4:]
        else:
            end_line = line[3:]
        aux = aux + 1
        #f.write(end_line.replace('\t', ','))
        f.write(end_line)
    f.close()



path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
folder = path + "/Code/ctFiles"
aux_cont = 0
for filename in sorted(os.listdir(folder), reverse=True):
    if filename.endswith(".ct"):
        name = filename[:-3]
        file = folder + "/" + filename
        aux_cont += 1
        print(aux_cont ,", Working on ", filename)
        rewrite_file(file, name)
    
    if aux_cont == 40000:
        break