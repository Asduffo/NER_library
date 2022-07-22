# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:41:35 2022

@author: devenv
"""

import torch

class MaskedJaccard():
    def __init__(self,
                 pad_id = 14,                   #id used by the model to represent the PAD tokens
                 from_logits = False,           #whether the output is from logits or not. If not, perform arg_max
                 device = 'cpu',                #cpu or cuda
                 verbose = 2,                   #0: don't print anything. 1: print the prediction, 2: print the target as well.
                 verbose_display_tags = 32):    #if verbose > 0, print the tokens from 0 to verbose_display_tags - 1
        self.pad_id = pad_id
        self.from_logits = from_logits
        
        self.current_jaccard   = 0
        self.examined_elements = 0
        
        self.device = device
        self.verbose = verbose
        self.verbose_display_tags = verbose_display_tags
        
    def update_state(self, d, o):
        y_t = torch.clone(d).to(self.device).detach()
        y_p = torch.clone(o).to(self.device).detach()
        
        if not self.from_logits:
            y_p = torch.argmax(y_p, dim=2).to(self.device).detach()
        
        if(self.verbose >= 2):
            print("y_t[0] = ", y_t[0][:self.verbose_display_tags])
        
        if(self.verbose >= 1):
            print("y_p[0] = ", y_p[0][:self.verbose_display_tags])
            
        
        
        #y_p.size(dim=0) = #elementi nel batch (ci serve in quanto poi facciamo la media
        #di tutti i jaccard di tutti gli elementi in self.result()))
        self.examined_elements += y_p.size(dim=0)
            
        valid_elements_matrix = torch.where(y_t != self.pad_id, 1, 0).to(self.device).detach()
        
        #numero elementi validi per ogni elemento nel batch
        valid_elements_sum = torch.sum(valid_elements_matrix, dim = 1).to(self.device).detach()
        
        #matrice di elementi ad 1 dove abbiamo predetto correttamente il token
        correctly_predicted = torch.where(y_t == y_p, 1, 0).to(self.device).detach()
        
        #elimina gli elementi non validi dai correttamente predetti
        correctly_predicted*=valid_elements_matrix
        correctly_predicted = torch.sum(correctly_predicted, dim = 1).to(self.device).detach()
        
        #calcola jaccard come (#elem. predetti correttamente)/(2*#tokens - #elem. predetti correttamente)
        #NB: è calcolato per ogni singolo elemento nel batch
        jaccards = torch.where((2*valid_elements_sum - correctly_predicted) > 0.0, 
                               correctly_predicted/(2*valid_elements_sum - correctly_predicted), 
                               torch.tensor(1.)).to(self.device).detach()
        
        #somma i vari jaccard nella somma totale. in result() faremo la media.
        self.current_jaccard +=  torch.sum(jaccards).to(self.device).detach()
        
        #safety check
        del y_t, y_p, valid_elements_matrix, valid_elements_sum, correctly_predicted, jaccards
        
    def result(self):
        if(self.examined_elements == 0):
            return 1
        else:
            return (self.current_jaccard/self.examined_elements).detach().numpy()
    
    def reset_state(self):
        self.current_jaccard   = 0
        self.examined_elements = 0
        