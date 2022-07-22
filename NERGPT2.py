# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:08:14 2022

@author: devenv
"""

import copy as cp

import torch
import numpy as np
import pandas as pd
import copy as cp

from NERBert import NERBert
from NERBase import OptimizerData, LayerData, SchedulerData
from CustomAccuracy import MaskedJaccard

from transformers import GPT2Config, AutoModel, AutoTokenizer, GPT2TokenizerFast

#NERBert variant which uses GPT-2 instead
class NERGPT2(NERBert):
    def __init__(self,
                 #same parameters as NERBert
                 model_name : str = 'gpt2', #gpt2-large
                 max_epochs : int = 10,
                 batch_size : int = 1,
                 accumulate_batch_size : int = None,
                 loss : torch.nn = torch.nn.CrossEntropyLoss(),
                 accuracy = MaskedJaccard(pad_id = 14),
                 optimizer_data : OptimizerData = OptimizerData(torch.optim.SGD, lr = .001),
                 scheduler_data : SchedulerData = SchedulerData(torch.optim.lr_scheduler.ExponentialLR, gamma = .9),
                 random_state : int = 0,
                 verbose : int = 0,
                 device = 'cpu'):
        
        super().__init__(model_name,
                         max_epochs,
                         batch_size,
                         accumulate_batch_size,
                         loss,
                         accuracy,
                         optimizer_data,
                         scheduler_data,
                         random_state,
                         verbose,
                         device)
        
        self.start_char_pad = 0
        self.end_char_pad   = 1
        
        self.token_reconstruction_start_offset = 0
        self.token_reconstruction_end_offset = 0
    
    #used to define the initial value of self.last_layer_features
    def get_encoder_n_units(self, encoder):
        print("gpt2 size: ", self.config.n_embd)
        return self.config.n_embd
    
    #creates the gpt2 tokenizer and transformer
    def create_encoder(self):
        self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast = True)
        # self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name, trim_offsets = False)
        
        self.config = GPT2Config.from_pretrained(self.model_name)
        
        #some tokens extra informations (mainly padding ids etc)
        self.cls_token = self.tokenizer.bos_token 
        self.sep_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': self.pad_token})
        
        self.max_tokenization_length = self.config.n_positions 
        
        self.cls_id = self.config.bos_token_id
        self.sep_id = self.config.eos_token_id
        self.pad_id = 50257
        
        print("self.config.eos_token_id = ", self.config.eos_token_id)
        
        
        
    #gpt2 feed forward
    def encoder_forward(self, X, y = None):
        encoded = self.encoder(input_ids      = X[0],
                               attention_mask = X[1])[0].to(self.device)
        # print("encoded.shape = ", encoded.shape)
        return encoded
    
    #splits the samples in X_raw in a way such that they require less than the 
    #requested amount of tokens.
    def split_samples_into_acceptable_lengths(self, X_raw, y_raw = None):
        X = X_raw.copy(deep = True)
        
        y = None
        if(y_raw is not None):
            y = y_raw.copy(deep = True)
            # y = pd.DataFrame(rows = np.arange(start = 0, len(y_raw)), columns = ['abstract_id', 'data'])
        
        X, y = super().split_samples_into_acceptable_lengths(X, y)
        
        # if(y_raw is not None):
        #     print("y_raw 1", y_raw)
        
        #adds a space at the begin of the sentence otherwise gpt2 tokenizer screws up
        y_dataframe_list = []
        for i in range(len(X)):
            # X.iloc[i]['data'] = " " + X.iloc[i]['data']
            # print(X.iloc[i]['data'])
            
            if(y_raw is not None):
                
                entities_list = []
                
                for j in range(len(y_raw.iloc[i]['data'])):
                    #we are forced to do an "indirect copy" since otherwise
                    #it keeps the reference to the original y.iloc[i]['data'] list
                    #and modifying it would change the original list as well.
                    entities_list.append((y.iloc[i]['data'][j][0],
                                          y.iloc[i]['data'][j][1],
                                          y.iloc[i]['data'][j][2],
                                          y.iloc[i]['data'][j][3]))
                
                y_dataframe_list.append([y_raw.iloc[i]['abstract_id'], entities_list])
            
                
        
        if(y_raw is not None):
            y = pd.DataFrame(y_dataframe_list, columns = ['abstract_id', 'data'])
        
        return X, y
    
    #merges th samples with the same article ID
    def reconstruct_sample_into_single_arcticles(self, X_raw, y_raw = None):
        X = cp.deepcopy(X_raw) #it's a list of strings to attach
        
        y = None
        if(y_raw is not None):
            y = cp.deepcopy(y_raw)
        
        #removes the initial space and fixes the offset of the entities
        for i in range(len(X)):
            X[i] = X[i][1:]
        
        if(y_raw is not None):
            for i in range(len(y)):
                for j in range(len(y[i])):
                    y[i][j] = (y[i][j][0],
                               y[i][j][1],  
                               y[i][j][2],
                               y[i][j][3])
        
        return super().reconstruct_sample_into_single_arcticles(X, y)
    
    #used by seqeval for the f1 score.
    def get_targets_as_list_of_lists(self, X_raw, y_raw):
        p = []
        for k in range(len(X_raw)):
            #current sample
            X = cp.deepcopy(X_raw.iloc[[k]])
            y = cp.deepcopy(y_raw.iloc[[k]])

            _, y2 = self.preprocess_dataset_dataframe_format(X, y)

            l = []
            for i in range(y2.size(0)):
                curr_sample = y2[i, :].int().tolist()

                try:
                    end_index = curr_sample.index(self.unique_list[self.sep_token])
                except:
                    end_index = self.max_tokenization_length

                for t in curr_sample[0:end_index]:
                    l = l + [self.inverse_unique_list[t]]

            p.append(l)
        return p
    
    #predicts the tokens for an input dataset
    def predict(self, X_raw):
        self.eval()
        p = []
        for k in range(len(X_raw)):
            #current sample
            X_strings = cp.deepcopy(X_raw.iloc[[k]])
            
            #splits into tokens of self.max_tokenization_length and proceeds
            X_raw_2, _ = self.split_samples_into_acceptable_lengths(X_strings)
            X_strings, _ = self.preprocess_dataset_dataframe_format(X_raw_2)
            
            #gets the tokens, the attention matrix and the mapping data
            tokens = X_strings[0]
            attention = X_strings[1]
            
            #it goes one by one instead than feeding the whole thing in order to
            #avoid out-of-memory issues.
            predictions = torch.zeros((len(tokens), self.max_tokenization_length)).to(self.device)
            
            for i in range(len(tokens)):
                o, _ = self([tokens[i].unsqueeze(0), attention[i].unsqueeze(0)])
                
                #converts from probabilities into labels
                o = torch.argmax(o, dim = 2).to(self.device).detach()
                
                #attachs it into the predictions vector
                predictions[i,:] = o
                
            if(self.apply_get_entities_predict_postprocessing):
                predictions = self.get_entities_predict_postprocessing(predictions)
            
            l = []
            for i in range(predictions.size(0)):
                curr_sample = predictions[i, :].int().tolist()
                
                try:
                    end_index = curr_sample.index(self.unique_list[self.sep_token])
                except:
                    end_index = self.max_tokenization_length
                
                
                for t in curr_sample[0:end_index]:
                    l = l + [self.inverse_unique_list[t]]
                    
            p.append(l)
        self.train()
        return p
        
