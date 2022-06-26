import operations as op
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 
from sklearn.metrics import confusion_matrix


class Rna:

    def __init__(self, pName, pSequence, pStructure, pGroup):
        self.name = pName
        self.group = pGroup
        self.sequence = pSequence
        self.structure = pStructure
    
    # Update methods
    def update_structure(self, pStructure):
        self.structure = pStructure
    
    def update_name(self, pName):
        self.name = pName
    
    def update_sequence(self, pSequence):
        self.sequence = pSequence

    ##########################################################
    ##########################################################

    # Evaluation methods:

        # Binary valuation methods:
            # Binary evaluation methods calculate the metrics 
            # where we dont care with wich nucleotide one is paired.
            # We only care if they are paired or not
            # True: X has a pair. False: X doesnt have a pair.
            # True positive: X is paired in both predicted sequence and ground truth...
    
        # Multiclassevaluation methods:
            # Multiclass evaluation methods calculate the metrics 
            # where we do care with wich nucleotide one is paired.

    def fscore(self, pStructure, mode = "multiclass"):
        # Calculates the F1 score 
        if mode == "binary":
            y_pred, y_true = op.binary_list(self.structure, pStructure)
            score = f1_score(y_true,y_pred)
        else:
            y_pred, y_true = op.multilabel_list(self.structure, pStructure, self.sequence)
            score = f1_score(y_true,y_pred, average="micro")
        return score
    
    def precision(self, pStructure, mode = "multiclass"):
        # Calculates the Precision score 
        if mode == "binary":
            y_pred, y_true = op.binary_list(self.structure, pStructure)
            score = precision_score(y_true,y_pred)
        else:
            y_pred, y_true = op.multilabel_list(self.structure, pStructure, self.sequence)
            score = precision_score(y_true,y_pred, average="micro")
        return score
    
    def recall(self, pStructure, mode = "multiclass"):
        # Calculates the recall score
        if mode == "binary":
            y_pred, y_true = op.binary_list(self.structure, pStructure)
            score = recall_score(y_true,y_pred)
        else:
            y_pred, y_true = op.multilabel_list(self.structure, pStructure, self.sequence)
            score = recall_score(y_true,y_pred, average="micro")
        return score

    def accuracy(self, pStructure, mode = "multiclass"):
        # Calculates the recall score
        if mode == "binary":
            y_pred, y_true = op.binary_list(self.structure, pStructure)
        else:
            y_pred, y_true = op.multilabel_list(self.structure, pStructure, self.sequence)

        score = accuracy_score(y_true,y_pred)
        return score
    
    def cnf_matrix(self, pStructure, mode = "multiclass"):
        # Calculates de confusion matrix
        if mode == "binary":
            y_pred, y_true = op.binary_list(self.structure, pStructure)
        else:
            y_pred, y_true = op.multilabel_list(self.structure, pStructure, self.sequence)

        mtrix = confusion_matrix(y_true,y_pred)
        
        return mtrix

##########################################################
##########################################################

# Debug method
    
    def evaluate(self, pStructure):
        print("Name: ", self.name)
        print("Sequence lenght: ", len(self.sequence))
        print(" ")

        print("Sequence: ", self.sequence)
        # cont = 0
        # for let in self.sequence:
        #     print(cont,let, " ", end="")
        #     cont +=1
            
        # print(" ")
        # print(" ")

        # print("Truth:")
        # aux_list = self.structure["base_pair"].tolist()
        # cont = 0
        # for let in aux_list:
        #     print(cont,let, " ", end="")
        #     cont +=1

        # print(" ")
        # print(" ")

        # print("Predicted structure: ")
        # cont = 0
        # for let in pStructure:
        #     print(cont,let, " ", end="")
        #     cont +=1

        print(" ")
        print(" ")

        print("Results: Binary. (Paired or not paired)")
        print("Accuracy: ",     self.accuracy(pStructure, mode="binary"))
        print("Precision: ",    self.precision(pStructure, mode="binary"))
        print("Recall: ",       self.recall(pStructure, mode="binary"))
        print("F1 score: ",     self.fscore(pStructure, mode="binary"))
        print("Confusion matrix:")
        print(self.cnf_matrix(pStructure, mode="binary"))

        print(" ")
        print(" ")

        print("Results: Multiclass. (Paired with same element)")
        print("Accuracy: ",     self.accuracy(pStructure))
        print("Precision: ",    self.precision(pStructure))
        print("Recall: ",       self.recall(pStructure))
        print("F1 score: ",     self.fscore(pStructure))
        print("Confusion matrix:")
        print(self.cnf_matrix(pStructure))

        print(" ")
        print(" ")
        
        print("Exact matches (paired with the same index, thus with same element):")
        matches = op.exact_matches(self.structure["base_pair"].tolist(), pStructure)
        print(matches.count(1), " out of ", len(self.sequence))
        print("Exact match rate ",(matches.count(1)*100)/len(self.sequence),"%" )

        print(" ")
        print(" ")

    def comp_evaluate(self, pStructure):
        bin_acc = self.accuracy(pStructure, mode="binary")
        bin_recall = self.recall(pStructure, mode="binary")
        bin_f = self.fscore(pStructure, mode="binary")
        bin_prec = self.precision(pStructure, mode="binary")

        ex_acc = self.accuracy(pStructure)
        ex_recall = self.recall(pStructure)
        ex_f = self.fscore(pStructure)
        ex_prec = self.precision(pStructure)
        matches = op.exact_matches(self.structure["base_pair"].tolist(), pStructure)
        ex_amm = ((matches.count(1)*100)/len(self.sequence))

        return bin_acc, bin_recall, bin_f, bin_prec, ex_acc, ex_recall, ex_f, ex_prec, ex_amm
