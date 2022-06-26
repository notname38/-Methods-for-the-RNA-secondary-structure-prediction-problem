import numpy as np

def format_result(pair_list, nElements):
    # TODO:
    # take a good look at this method. Might find a better way.
    result_list = np.zeros(nElements, dtype=np.int32)

    for pair in pair_list:
        result_list[pair[0]] = pair[1] + 1
        result_list[pair[1]] = pair[0] + 1
    return list(result_list) 


##########################################################################
##########################################################################

# binary_lists serves to help evaluate structures 
    # Structure_1 is a pandas dataset.
    # Structure_2 is a list of pairs.

    # We have 2 options:
        # Not paired.
        # Paired.

    # Encoding of the options:
        # Not paired. = 0
        # Paired. = 1
    
    # Returns two lists:
        # y_pred where y_pred(i) is if element i is paired in the prediction.
        # y_true where y_true(i) is if element i is paired in the ground truth.

def binary_list(structure_1, structure_2):
    true_pairs = structure_1["base_pair"].tolist()
    nElements = len(true_pairs)
    y_pred = []
    y_true = []

    for i in range(nElements):
        aux = structure_2[i]
        aux_2 = true_pairs[i]
        x = 1
        y = 1

        if aux == 0:
            x = 0
        y_pred.append(x)

        if aux_2 == 0:
            y = 0
        y_true.append(y)
        
    return list(y_pred), list(y_true)

##########################################################################
##########################################################################

##########################################################################
##########################################################################

# multilabel_list serves to help evaluate structures 
    # Structure_1 is a pandas dataset.
    # Structure_2 is a list of pairs.
    # Seq is the original sequence.

    # We have five options:
        # Not paired.
        # Paired with C
        # Paired with G
        # Paired with A
        # Paired with U
    
    # Encoding of the options:
        # Not paired. = 0
        # C = 1
        # G = 2
        # A = 3
        # U = 4
    
    # Returns two lists:
        # y_pred where y_pred(i) is to which element the i element is paired to in the prediction.
        # y_true where y_true(i) is to which element the i element is paired to in the ground truth.
    

def encode(element):
    if element == "C":
        return 1
    elif element == "G":
        return 2
    elif element == "A":
        return 3
    elif element == "U":
        return 4

def multilabel_list(structure_1, structure_2, seq):
    nElements = len(seq)
    true_pairs = structure_1["base_pair"].tolist()
    y_pred = []
    y_true = []

    for i in range(nElements):

        aux = structure_2[i]
        aux_2 = true_pairs[i]
        if aux == 0:
            x = 0
        else:
            x = encode(seq[aux - 1])
        y_pred.append(x)

        if aux_2 == 0:
            y = 0
        else:
            y = encode(seq[aux_2 - 1])

        y_true.append(y)

    return list(y_pred), list(y_true)

##########################################################################
##########################################################################

def exact_matches(structure_1, structure_2):
    nElements = len(structure_1)
    matches = []

    for i in range(nElements):
        if structure_1[i] == structure_2[i]:
            matches.append(1)
        else:
            matches.append(0)
    
    return list(matches)





