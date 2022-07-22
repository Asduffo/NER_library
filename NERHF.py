# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:43:00 2022

@author: devenv
"""

import torch
import pandas as pd
import numpy as np

from NERBase import NERBase, OptimizerData, SchedulerData
from CustomAccuracy import MaskedJaccard

from transformers import AutoModel, AutoTokenizer

#NERBase specialized in hugging face's pretrained models
class NERHF(NERBase):
    def __init__(self,
                 model_name : str = 'bert-base-uncased',    #name of the model (according to huggingface's database)
                 
                 #for more informations, see NERBase
                 max_epochs : int = 10,
                 batch_size : int = 1,
                 accumulate_batch_size : int = None,
                 loss : torch.nn = torch.nn.CrossEntropyLoss(),
                 accuracy = MaskedJaccard(pad_id = 14),
                 optimizer_data : OptimizerData = OptimizerData(torch.optim.SGD, lr = .001),
                 scheduler_data : SchedulerData = SchedulerData(torch.optim.lr_scheduler.ExponentialLR, gamma = .9),
                 random_state : int = 0,
                 verbose : int = 0,
                 device = 'cpu',
                 use_softmax_output = False,
                 apply_get_entities_predict_postprocessing = False):
        
        super().__init__(max_epochs,
                         batch_size,
                         accumulate_batch_size,
                         loss,
                         accuracy,
                         optimizer_data,
                         scheduler_data,
                         random_state,
                         verbose,
                         device,
                         use_softmax_output)
        self.model_name = model_name
        self.apply_get_entities_predict_postprocessing = apply_get_entities_predict_postprocessing
        
    #creates the huggingface transformer and its tokenizer
    def create_encoder(self):
        self.encoder = AutoModel.from_pretrained(self.model_name, 
                                                 add_pooling_layer=False).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast = True)
        
    
    #feed forward method
    def forward(self, X, y = None):
        #each encoder might process the input in a different way. as a result,
        #we proceed to have a customized method
        o = self.encoder_forward(X).to(self.device)
        
        #feed forwards through the other layers
        for layer in self.layers:
            o = layer(o).to(self.device)
        
        # print("o.shape = ", o.shape)    
        
        loss = None #batch loss
        # acc = None  #batch accuracy
        if(y != None):
            d = torch.clone(y).to(self.device)
            
            active_loss = (X[1].view(-1) == 1).to(self.device)
            
            #flattens curr_out
            active_logits = o.view(-1, self.output_layer_size).to(self.device)
            
            #ignores the padding tokens
            active_labels = torch.where(
                active_loss, 
                d.view(-1), 
                torch.tensor(self.loss.ignore_index).type_as(d).to(self.device)
            )
            
            #calculates the loss
            loss = self.loss(active_logits, active_labels).to(self.device)
            
            del active_labels
            del active_logits
            del active_loss
            del d
        
        return o, loss
    
    #transformer's feed forward
    def encoder_forward(self, X, y = None):
        encoded = self.encoder(X[0], attention_mask = X[1])[0].to(self.device)
        return encoded
    
    #same thing of NERBase
    def preprocess_dataset(self, X_raw, y_raw = None):
        #dataframes requires moooore work
        if(type(X_raw) == pd.DataFrame):
            X, y = self.preprocess_dataset_dataframe_format(X_raw, y_raw)
            return X, y
        
        X = []
        y = None
        
        #turns X_raw into an array containing two tensors (data and attention mask)
        if(type(X_raw) == list):
            if(len(X_raw) != 2):
                raise Exception("NERBert.py - preprocess_dataset: accepted X_raw length is 2."
                                "Actual length: ", len(X_raw))
            
            for i in range(2):
                if(type(X_raw[i]) == np.ndarray):
                    X.append(torch.from_numpy(X_raw[i]).to(self.device))
                elif(type(X_raw[i]) == torch.tensor):
                    X.append(X_raw[i].clone().to(self.device))
                else:
                    raise Exception("NERBert.py - preprocess_dataset: X[", i, "]",
                                    " type ", type(X_raw[i]), " not recognized.")
        else:
            raise Exception("NERBert.py - preprocess_dataset: X_raw unrecognized ",
                            "data format: ", type(X_raw))
        
        #turns y_raw into a tensor
        if(type(y_raw) == np.ndarray):
            y = torch.from_numpy(y_raw).to(self.device)
        elif(type(y_raw) == torch.tensor):
            y = y_raw.clone().to(self.device)
        elif(y_raw is None):
            y = None
        else:
            raise Exception("NERBert.py - preprocess_dataset: y type ", 
                            type(y_raw[i]), " not recognized.")
        
        return X, y