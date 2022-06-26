import seqfold as sF
import operations as op

class algorithm:

    def __init__(self, method = "None"):
        self.method = method
        
    def calculate_structure(seq):
        print(" Calculates the optimal secondary structure for the sequence: " + seq)

class RNAFold(algorithm):

    def calculate_structure(self, seq):
        pair_list = []
        structure = sF.fold(seq)
        for pair in structure:
            pair_list.append([pair.ij[0][0],pair.ij[0][1]])
        pair_list = op.format_result(pair_list, len(seq))

        return list(pair_list)


""" print("Algorithm Test:")
seq = "GGAUACGGCCAUACUGCGCAGAAAGCACCGCUUCCCAUCCGAACAGCGAAGUUAAGCUGCGCCAGGCGGUGUUAGUACUGGGGUGGGCGACCACCCGGGAAUCCACCGUGCCGUAUCCU"
print("Sequence:", seq)
print("Sequence lenght: ", len(seq))
rnaFold = RNAFold("RNAFOLD")
result = rnaFold.calculate_structure(seq)
print(" ")
print("Method: ", rnaFold.method)
for pair in result:
    print("I: ", pair[0], ",        J: ", pair[1]) """