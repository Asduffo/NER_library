# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:56:50 2022

@author: Admin
"""

import torch
import numpy as np
import pandas as pd
import copy as cp

from NERBert import NERBert
from NERBase import OptimizerData, LayerData, SchedulerData
from CustomAccuracy import MaskedJaccard

from transformers import BertConfig, BatchEncoding

#Unused class. Essentially, it's a variant of NERBert which also adds an LSTM
#layer above BERT.
class NERBertLSTM(NERBert):
    def __init__(self,
                 #same parameters as NERBert
                 model_name : str = 'bert-base-uncased',
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
                 apply_get_entities_predict_postprocessing = False,
                 
                 lstm_hidden_size = 32,         #hidden dimension of the LSTM
                 lstm_num_layers = 1,           #number of layers of the LSTM
                 lstm_internal_dropout_p = .25, #LSTM's internal dropout probability
                 bi_lstm = True,                #TRUE => bidirectional LSTM
                 
                 lstm_final_dropout_p = .25):   #LSTM's output dropout probability
        
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
                         device,
                         use_softmax_output,
                         apply_get_entities_predict_postprocessing)
        
        self.lstm_hidden_size           = lstm_hidden_size
        self.lstm_num_layers            = lstm_num_layers
        self.lstm_internal_dropout_p    = lstm_internal_dropout_p
        self.bi_lstm                    = bi_lstm
        self.lstm_final_dropout_p       = lstm_final_dropout_p
        
    #creates bert ant then the LSTM
    def create_encoder(self):
        super().create_encoder()
        
        #L = sequence_length = self.sentence_max_length
        #input_size = H_in = features of the cnn (word_embedding_dim + char_cnn_out_channels)
        lstm_input_size = BertConfig.from_pretrained(self.model_name).hidden_size
        
        self.LSTM = torch.nn.LSTM(input_size = lstm_input_size,
                                  hidden_size = self.lstm_hidden_size,
                                  num_layers = self.lstm_num_layers,
                                  batch_first = True,
                                  dropout = self.lstm_internal_dropout_p,
                                  bidirectional = self.bi_lstm,
                                  ).to(self.device)
        
        self.LSTM_final_dropout = torch.nn.Dropout(p = self.lstm_final_dropout_p).to(self.device)
        
        self.add_module("LSTM", self.LSTM)
        
    #the output size of the LSTM depends on whether it's bidirectional or not.
    def get_encoder_n_units(self, encoder):
        multiplier = 1
        if(self.bi_lstm == True):
            multiplier = 2
        return multiplier * self.lstm_hidden_size
        
    #LSTM feed forward
    def forward_lstm(self, X):
        # output = torch.transpose(X, 1, 2)
        output = X
        # print(output.size())
        
        #feed forward
        output, _ = self.LSTM(output)
        
        #swaps the channels back in place
        output = torch.transpose(output, 1, 2)
        
        output = self.LSTM_final_dropout(output).to(self.device)
        
        return output
    
    #feeds X into the Bert and then the LSTM.
    def encoder_forward(self, X, y = None):
        output = super().encoder_forward(X, y)
        
        output = self.forward_lstm(output)
        output = torch.transpose(output, 1, 2)
        
        return output.to(self.device)