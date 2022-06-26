import os
import pandas as pd
from pandas.errors import ParserError

def write_on_file(file, name, debug = False):
    ind = 1
    namefile = path + "/data/bpseq_data/" + name + ".bpseq"
    f = open(namefile, "w")
    file1 = open(file, "r")
    lines = file1.readlines()
    lines = lines[1:]

    for line in lines:
        splitted_line = line.split()
        newline = str(ind) + " " + splitted_line[1] + " " + splitted_line[4]
        if debug:
            print(newline)
            debug = False
        f.write(newline)
        f.write("\n")
        ind = ind + 1
    f.close()

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
folder = path + "/data/RNAStrAlign-Reduced_NoFolders"
aux_cont = 0
skipped_files = []
for filename in sorted(os.listdir(folder), reverse=True):
    if filename.endswith(".ct"):
        name = filename[:-3]
        file = folder + "/" + filename
        aux_cont += 1
        print(aux_cont ,", Converting ", name, " to .bpseq.")
        try:
            dataframe = write_on_file(file, name, False)
        except Exception:
            skipped_files.append(name)
            

print("Files that did not meet the requirement: ")
print(skipped_files)

