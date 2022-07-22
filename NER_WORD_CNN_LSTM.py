# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 11:58:13 2022

@author: devenv
"""

import torch
import numpy as np
import pandas as pd

from NERBase import  OptimizerData, LayerData, SchedulerData
from NER_WORD_CNN import NER_WORD_CNN
from CustomAccuracy import MaskedJaccard

#class which implements the CNN+LSTM network
class NER_WORD_CNN_LSTM(NER_WORD_CNN):
    def __init__(self,
                 #see the parent classes for these parameters
                 max_epochs : int = 10,
                 batch_size : int = 1,
                 accumulate_batch_size = None,
                 loss : torch.nn = torch.nn.CrossEntropyLoss(),
                 accuracy = MaskedJaccard(pad_id = 1),
                 optimizer_data : OptimizerData = OptimizerData(torch.optim.SGD, lr = .01),
                 scheduler_data : SchedulerData = SchedulerData(torch.optim.lr_scheduler.ExponentialLR, gamma = .9),
                 random_state : int = 0,
                 verbose : int = 0,
                 device = 'cpu',
                 use_softmax_output = False,
                 use_whole_word_embeddings = True,
                 embedding_file_path = None,
                 freeze_word_embedding_weights = False,
                 sentence_max_length = 512,
                 char_cnn_out_channels = 4,
                 char_embedding_dim = 60,
                 char_cnn_kernel_size = 3,
                 word_embedding_n_words = 20000, #irrelevant if we use pretrained embeddings
                 word_embedding_dim = 100,       #same
                 word_embedding_dropout_p = .5,
                 
                 
                 lstm_hidden_size = 32,         #size of the LSTM output
                 lstm_num_layers = 1,           #number of LSTM layers
                 lstm_internal_dropout_p = .25, #dropout probability for the LSTM
                 bi_lstm = True,                #TRUE = bidirectional LSTM
                 lstm_final_dropout_p = .25):   #LSTM's output dropout probability
        
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
                         use_softmax_output,
                         
                         use_whole_word_embeddings,
                         embedding_file_path,
                         freeze_word_embedding_weights,
                         sentence_max_length,
                         char_cnn_out_channels,
                         char_embedding_dim,
                         char_cnn_kernel_size,
                         word_embedding_n_words, 
                         word_embedding_dim,
                         word_embedding_dropout_p)
        
        self.lstm_hidden_size           = lstm_hidden_size
        self.lstm_num_layers            = lstm_num_layers
        self.lstm_internal_dropout_p    = lstm_internal_dropout_p
        self.bi_lstm                    = bi_lstm
        self.lstm_final_dropout_p       = lstm_final_dropout_p
        
    def create_encoder(self):
        #creates the CNN
        super().create_encoder()
        
        #L = sequence_length = self.sentence_max_length
        #input_size = H_in = features of the cnn (word_embedding_dim + char_cnn_out_channels)
        lstm_input_size = self.word_embedding_dim + self.char_cnn_out_channels
        
        #and the LSTM
        self.LSTM = torch.nn.LSTM(input_size = lstm_input_size,
                                  hidden_size = self.lstm_hidden_size,
                                  num_layers = self.lstm_num_layers,
                                  batch_first = True,
                                  dropout = self.lstm_internal_dropout_p,
                                  bidirectional = self.bi_lstm,
                                  ).to(self.device)
        
        self.LSTM_final_dropout = torch.nn.Dropout(p = self.lstm_final_dropout_p).to(self.device)
        
        self.add_module("LSTM", self.LSTM)
        
    ###########################################################################
    #the LSTM output's size depends on whether it's bidirectional or not
    def get_encoder_n_units(self, encoder):
        multiplier = 1
        if(self.bi_lstm == True):
            multiplier = 2
        return multiplier * self.lstm_hidden_size
        
    #LSTM feed forward (some transpositions are necessary)
    def forward_lstm(self, X):
        # swaps L and H_in
        output = torch.transpose(X, 1, 2)
        # output = X
        
        #feed forward
        output, _ = self.LSTM(output)
        
        #swaps the channels back in place
        output = torch.transpose(output, 1, 2)
        
        output = self.LSTM_final_dropout(output).to(self.device)
        
        return output
    
    #input => CNN => LSTM => output
    def encoder_forward(self, X, y = None):
        output = self.forward_cnn(X)
        # print("CNN output size = ", output.size())
        
        output = self.forward_lstm(output)
        output = torch.transpose(output, 1, 2)
        
        return output.to(self.device)