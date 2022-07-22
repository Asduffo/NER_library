# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 11:58:13 2022

@author: devenv
"""

from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import pandas as pd
import math as mt

from BatchGenerator import BatchGenerator
from NERBase import NERBase, OptimizerData, LayerData, SchedulerData
from CustomAccuracy import MaskedJaccard

from gensim import models

#class which implements a 1D CNN for NER
class NER_WORD_CNN(NERBase):
    def __init__(self,
                 #see NERBase for these parameters
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
                 
                 
                 use_whole_word_embeddings = True,      #whether we want to use word embeddings as well
                 embedding_file_path = None,            #path of the word embeddings data. If None, do not use word embeddings/use a random embedding (if use_whole_word_embeddings = True)
                 freeze_word_embedding_weights = False, #whether to train the embeddings weights or not
                 
                 sentence_max_length = 512,             #maximum number of tokens in a sentence
                 char_cnn_out_channels = 4,             #if zero => don't use CNN embedding. Only word level
                 char_embedding_dim = 60,               #do not change this parameter (how many unique characters do the text use?)
                 char_cnn_kernel_size = 3,              #CNN kernel size
                 
                 #should be higher or equal than the number of different words
                 #IN BOTH TRAINING AND TEST SET (+2 because of UNK and SPACE)
                 word_embedding_n_words = 20000, 
                 word_embedding_dim = 100,              #embeddings size
                 word_embedding_dropout_p = .5,         #embedding dropout p
                 ):
        
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
        
        self.X_unique_list      = None
        self.X_unique_char_list = None
        self.y_unique_list      = None
        
        self.use_whole_word_embeddings     = use_whole_word_embeddings,
        self.embedding_file_path           = embedding_file_path
        self.freeze_word_embedding_weights = freeze_word_embedding_weights,
        
        self.sentence_max_length   = sentence_max_length
        self.char_cnn_out_channels = char_cnn_out_channels
        self.char_embedding_dim    = char_embedding_dim
        self.char_cnn_kernel_size  = char_cnn_kernel_size
        
        self.word_embedding_dim       = word_embedding_dim
        self.word_embedding_n_words   = word_embedding_n_words
        self.word_embedding_dropout_p = word_embedding_dropout_p
        
        # self.output_layer_data = LayerData(torch.nn.Softmax, dim = 12)
        
    def create_encoder(self):
        self.encoder = None
        
        #initializes the cnn
        self.char_cnn = torch.nn.Conv2d(in_channels = 1,
                                        #number of output features
                                        out_channels = self.char_cnn_out_channels, 
                                        
                                        #sliding window. rows = embed dim, cols = parameter
                                        kernel_size=(self.char_embedding_dim, self.char_cnn_kernel_size),
                                        
                                        #this ensures that it will process even words with only one character
                                        padding=(0, self.char_cnn_kernel_size - 1)).to(self.device)
        
        #initializes the word embedding (either as a random embedding)
        if(self.embedding_file_path == None):
            self.word_embedding = torch.nn.Embedding(self.word_embedding_n_words,self.word_embedding_dim).to(self.device)
            self.init_embedding(self.word_embedding.weight)
        else: #...or from a pretrained file
            model = models.KeyedVectors.load_word2vec_format(self.embedding_file_path, binary=True)
            weights = torch.FloatTensor(model.vectors)
            self.word_embedding = torch.nn.Embedding.from_pretrained(weights).to(self.device)

            self.word_embedding_n_words = weights.size(0)
            self.word_embedding_dim = weights.size(1)

            self.X_unique_list = {token: token_index for token_index, token in enumerate(model.index_to_key)}
            print("self.X_unique_list = ", type(self.X_unique_list))
        
        if(self.freeze_word_embedding_weights):
            self.word_embedding.weight.requires_grad = False
        
        self.word_embedding_dropout = torch.nn.Dropout(p = self.word_embedding_dropout_p).to(self.device)
        
        self.add_module("CNN", self.char_cnn)
        self.add_module("embed", self.word_embedding)
    
    #embedding's weights initialization
    def init_embedding(self, input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        torch.nn.init.uniform(input_embedding, -bias, bias)
    
    #removes the special characters, lowers the upper characters, and split by special character
    def tokenize_sentence(self, sentence):
        return sentence.replace('\\', '/').lower().split(' ')
    
    #dataset preprocessing (calls preprocess_dataset_dataframe_format)
    def preprocess_dataset(self, X_raw, y_raw = None):
        #dataframes requires moooore work
        if(type(X_raw) == pd.DataFrame):
            X, y = self.preprocess_dataset_dataframe_format(X_raw, y_raw)
            return X, y
        
        X = []
        y = None
        
        #turns X_raw into an array containing two tensors (data and attention mask)
        if(type(X_raw) == list and (type(y_raw) == list)):
            X, y = self.preprocess_dataset_numpy_format(X_raw, y_raw)
            return X, y
        else:
            raise Exception("NERBert.py - preprocess_dataset: X_raw unrecognized ",
                            "data format: ", type(X_raw))
        
        return X, y
    
    #gets the raw dataset and returns the dataset in the right format used by the network
    #(hence, tokenized)
    def preprocess_dataset_dataframe_format(self, X_raw, y_raw = None, add_unseen = True):
        X = [[] for x in range(len(X_raw))]  #tokenized sequences
        y = [[] for x in range(len(X_raw))]  #labels
        
        dataset_string = '<>'
        
        #for each element (sentence/abstract_id) in the text 
        for i in range(0, len(X_raw)):
            curr_sample_text = X_raw.iloc[i]["data"]
            dataset_string+=curr_sample_text
            
            tokens = self.tokenize_sentence(curr_sample_text)
            
            #creates the targets (initialize all tokens to 'O')
            targets = ["O" for x in range(len(tokens))]
            
            
            if(y_raw is not None):
                #get all labels records with same abstract_id 
                tags = y_raw.iloc[i]['data']
                
                #for each tag/entity in the instance
                for j in range(len(tags)):
                    #gets the index of the block containing character of index y_raw[i][j][0]
                    #(the first in an entity) by tokenizing the string behind it and
                    #counting how many tokens it generates. The resulting number + 1 is
                    #the index of the initial token
                    #NB: (tags[j][0] + 1): the +1 adds the first character of the entity
                    start_str = curr_sample_text[0:(tags[j][0] + 1)]
                    # print("\nstart_str = ", start_str)
                    
                    start_str_tok = self.tokenize_sentence(start_str)
                    
                    start_block_idx = len(start_str_tok) - 1
                    
                    #gets the index of the token containing character of index y_raw[i][j][1]
                    #(the last one in the current entity) by summing to start_block_idx
                    #the number of tokens composing the entity (minus one)
                    entity_tok = self.tokenize_sentence(tags[j][2])
                    
                    end_block_idx = start_block_idx + len(entity_tok) - 1

                    #labels the tokens associated to the current entity (the ones from
                    #start_block_idx to end_block_idx)
                    targets[start_block_idx] = "B-" + tags[j][3]
                    for k in range(start_block_idx + 1, end_block_idx + 1):
                        targets[k] = "I-" + tags[j][3]
            
            X[i] = tokens 
            y[i] = targets
            
            #fixes an annoying bug where at the end of the tokens it might put a
            #'' empty slot before the <sep> tag:
            if(X[i][-1] == ''):
                X[i].pop()
                y[i].pop()
        
        padding_key = '<pad>' #ID of paddings
        X = [[ '<cls>' ] + list(o) + [ '<sep>' ] for o in X]
        X = np.array([xi+[padding_key]*(self.sentence_max_length-len(xi)) for xi in X])
        
        #creates the attention matrix
        a = np.where(X == padding_key, 0, 1)
        
        padding_string = '<pad>'
        y = [['<cls>'] + list(o) + ['<sep>'] for o in y]
        y = np.array([xi+[padding_string]*(self.sentence_max_length-len(xi)) for xi in y])
        
        ###########################################################################
        
        
        #initializes the dictionary if we didn't do already (useful if we want
        #to encode another dataset (eg: validation/test set) with the same label
        #order)
        X_uniques = np.unique(X)
        X_labels = np.arange(start = 0, stop = len(X_uniques))
        
        
        if(self.X_unique_list == None):
            self.X_unique_list = {}
            for A, B in zip(X_uniques, X_labels):
                self.X_unique_list[A] = B
        elif(add_unseen):
            #expands the set in the unlucky times where there are elements in the
            #validation set (for example) which were not present in the training set.
            for A, B in zip(X_uniques, X_labels):
                if(not (A in self.X_unique_list)):
                    self.X_unique_list[A] = B
        
        
        ###########################################################################
        
        #coverts the dataset string into a large array of characters
        #(lowers the upper characters first)
        dataset_string = self.split(dataset_string.lower())
        
        #gets the unique characters
        X_uniques_char = np.unique(dataset_string)
        X_labels_char = np.arange(start = 0, stop = len(X_uniques_char))
        
        #initializes the char dictionary:
        if(self.X_unique_char_list == None):
            self.X_unique_char_list = {}
            for A, B in zip(X_uniques_char, X_labels_char):
                self.X_unique_char_list[A] = B
        elif(add_unseen):
            #expands the set in the unlucky times where there are elements in the
            #validation set (for example) which were not present in the training set.
            for A, B in zip(X_uniques_char, X_labels_char):
                if(not (A in self.X_unique_char_list)):
                    self.X_unique_char_list[A] = B
        
        ###########################################################################
        y_uniques = np.unique(y) #labels (int format)
        y_labels = np.arange(start = 0, stop = len(y_uniques)) #labels (int format)
        if(self.y_unique_list == None):
            #maps
            self.y_unique_list = {}
            for A, B in zip(y_uniques, y_labels):
                self.y_unique_list[A] = B
                
            #maps the labels
            """
            self.class_weights = compute_class_weight('balanced',
                                                      classes = y_uniques.tolist(),
                                                      y = y.flatten().tolist())
            self.loss.weight = torch.Tensor(self.class_weights).type(torch.FloatTensor).to(self.device)
            """
            # print("y_uniques = ", y_uniques)
            # print("self.class_weights = ", self.class_weights)
        elif(add_unseen):
            for A, B in zip(y_uniques, y_labels):
                if(not (A in self.y_unique_list)):
                    self.y_unique_list[A] = B
             
        
        
        y = np.vectorize(self.y_unique_list.get)(y) 
        
        #a and y can be tensors. X is the only one which has to stay ndarray
        #since it's made of strings until it reaches the cnn
        a = torch.from_numpy(a).type(torch.LongTensor).to(self.device)
        y = torch.from_numpy(y).type(torch.LongTensor).to(self.device)
        
        return [X, a], y
    
    #actually unused since we always use the dataframe format and not the numpy format
    def preprocess_dataset_numpy_format(self, X_raw, y_raw = None, add_unseen = True):
        X = [[] for x in range(len(X_raw))]  #tokenized sequences
        y = [[] for x in range(len(X_raw))]  #labels
        
        dataset_string  = '<>'
        padding_key     = '<pad>' #ID of paddings
        
        #for each element (sentence/abstract_id) in the text 
        for i in range(0, len(X_raw)):
            tokens = X_raw[i]
            
            
            targets = y_raw[i]
            for j in range(len(tokens)):
                dataset_string += tokens[j]
                tokens[j] = self.tokenize_sentence(tokens[j])[0]
            
            X[i] = tokens
            y[i] = targets
            
            #fixes an annoying bug where at the end of the tokens it might put a
            #'' empty slot before the <sep> tag:
            if(len(X[i]) > 0):
                if(X[i][-1] == ''):
                    X[i].pop()
                    y[i].pop()
                    
            #the -2 is for the cls/sep
            X[i] = ['<cls>'] + X[i] + ['<sep>'] + [padding_key]*(self.sentence_max_length-len(X[i]) - 2)
            y[i] = ['<cls>'] + y[i] + ['<sep>'] + [padding_key]*(self.sentence_max_length-len(y[i]) - 2)
            # print(len(X[i]), ", ", len(y[i]))
        
        X = np.vstack(X)
        y = np.vstack(y)
        
        #creates the attention matrix
        a = np.where(X == padding_key, 0, 1)
        
        print("X.shape = ", X.shape)
        
        ###########################################################################
        
        
        #initializes the dictionary if we didn't do already (useful if we want
        #to encode another dataset (eg: validation/test set) with the same label
        #order)
        X_uniques = np.unique(X)
        X_labels = np.arange(start = 0, stop = len(X_uniques))
        
        # print("X_uniques len = ", len(X_uniques))
        # print("X_uniques = ", X_uniques)
        
        if(self.X_unique_list == None):
            self.X_unique_list = {}
            for A, B in zip(X_uniques, X_labels):
                self.X_unique_list[A] = B
        elif(add_unseen):
            #expands the set in the unlucky times where there are elements in the
            #validation set (for example) which were not present in the training set.
            for A, B in zip(X_uniques, X_labels):
                # print(A)
                # print(type(A))
                # print("self.X_unique_list = ", type(self.X_unique_list))
                if(not (A in self.X_unique_list)):
                    self.X_unique_list[A] = B
        
        ###########################################################################
        
        #coverts the dataset string into a large array of characters
        #(lowers the upper characters first)
        dataset_string = self.split(dataset_string.lower())
        
        #gets the unique characters
        X_uniques_char = np.unique(dataset_string)
        X_labels_char = np.arange(start = 0, stop = len(X_uniques_char))
        # print("X_uniques_char = ", X_uniques_char)
        
        #initializes the char dictionary:
        if(self.X_unique_char_list == None):
            self.X_unique_char_list = {}
            for A, B in zip(X_uniques_char, X_labels_char):
                self.X_unique_char_list[A] = B
        elif(add_unseen):
            #expands the set in the unlucky times where there are elements in the
            #validation set (for example) which were not present in the training set.
            for A, B in zip(X_uniques_char, X_labels_char):
                if(not (A in self.X_unique_char_list)):
                    self.X_unique_char_list[A] = B
        
        # print(self.X_unique_char_list)
        
        ###########################################################################
        y_uniques = np.unique(y) #labels (int format)
        y_labels = np.arange(start = 0, stop = len(y_uniques)) #labels (int format)
        if(self.y_unique_list == None):
            #maps
            self.y_unique_list = {}
            for A, B in zip(y_uniques, y_labels):
                self.y_unique_list[A] = B
                
            #maps the labels
            """
            self.class_weights = compute_class_weight('balanced',
                                                      classes = y_uniques.tolist(),
                                                      y = y.flatten().tolist())
            self.loss.weight = torch.Tensor(self.class_weights).type(torch.FloatTensor).to(self.device)
            """
        elif(add_unseen):
            for A, B in zip(y_uniques, y_labels):
                if(not (A in self.y_unique_list)):
                    self.y_unique_list[A] = B
        
        y = np.vectorize(self.y_unique_list.get)(y)
        
        #a and y can be tensors. X is the only one which has to stay ndarray
        #since it's made of strings until it reaches the cnn
        a = torch.from_numpy(a).type(torch.LongTensor).to(self.device)
        y = torch.from_numpy(y).type(torch.LongTensor).to(self.device)
        
        return [X, a], y
    
    def split(self, word):
        return [char for char in word]

    #CNN forward: converts each word into a series of vectors, applies a 1D CNN,
    #and then max pooling + dropout.
    #Also, concatenates the output to the word embedding (if any)
    def forward_cnn(self, X):
        data = X[0]
        
        word_embedding_dim = 0
        char_cnn_out_channels = 0
        
        #gets some informations on the size of the embeddings
        if(self.use_whole_word_embeddings):
            word_embedding_dim = self.word_embedding_dim
        else:
            word_embedding_dim = 0
            
        #and of the cnn output
        if(self.char_cnn_out_channels > 0):
            char_cnn_out_channels = self.char_cnn_out_channels
        else:
            char_cnn_out_channels = self.char_cnn_out_channels
        
        
        embedding_total_dim = word_embedding_dim + char_cnn_out_channels
        if(embedding_total_dim == 0):
            embedding_total_dim = 1
        
        batch_embedding = torch.zeros((data.shape[0],
                                       embedding_total_dim,
                                       self.sentence_max_length))
        
        # for each sentence
        # for i in range(1):
        for i in range(data.shape[0]):
            #for each word in the sentence:
            for j in range(self.sentence_max_length):
                current_word = data[i, j]
                
                #this happens when we have more than one space between two words
                if(current_word == ''): current_word = 'unk_word'
                
                characters = self.split(current_word)
                
                ###############################################################
                #gets the word id from X_unique
                try:
                    word_id = torch.Tensor([self.X_unique_list[current_word]]).type(torch.LongTensor).to(self.device)
                except:
                    #this happens in case of unknown words.
                    print("unseen word: ", current_word)
                    word_id = torch.Tensor([self.word_embedding_n_words - 1]).type(torch.LongTensor).to(self.device)
                
                
                #creates the word embedding (if any. Otherwise it's just the word ID
                #found in the dictionary created while creating the dataset)
                if(self.use_whole_word_embeddings):
                    curr_word_whole_embedding = self.word_embedding(word_id).view((-1)).to(self.device)
                else:
                    curr_word_whole_embedding = torch.tensor([word_id]).to(self.device)
                
                ###############################################################
                
                #creates the char embeddings for the current word
                if(self.char_cnn_out_channels > 0):
                    #binary char embeddings
                    curr_word_char_embeddings = np.zeros((self.char_embedding_dim, len(characters)))
                    for k in range(len(characters)):
                        vector = np.zeros(shape = (self.char_embedding_dim))
                        index_of_one = self.X_unique_char_list[characters[k]]
                        
                        vector[index_of_one] = 1
                        
                        curr_word_char_embeddings[:, k] = vector
                    
                    #TODO: dropout non credo sia necessario qua visto che è tutto binario
                    #converts to torch format
                    curr_word_char_embeddings = torch.from_numpy(curr_word_char_embeddings).float().to(self.device)
                    curr_word_char_embeddings = curr_word_char_embeddings[None, None, :, :].float().to(self.device)
                    
                    #applies the feed forward through the cnn
                    curr_word_char_embeddings = self.char_cnn(curr_word_char_embeddings).to(self.device)
                    
                    #max pooling
                    curr_word_char_embeddings = torch.nn.functional.max_pool2d(curr_word_char_embeddings,
                        kernel_size=(1, curr_word_char_embeddings.size(3))).view((-1)).to(self.device)
                    
                    #concatenates the word embedding (if any)
                    final_word_embedding = torch.cat((curr_word_char_embeddings,
                                                      curr_word_whole_embedding)).view(-1,).to(self.device)
                else:
                    final_word_embedding = curr_word_whole_embedding.view(-1,).to(self.device)
                
                #concatenates the WORD embedding at the end.
                batch_embedding[i, :, j] = final_word_embedding
            
        #debug
        # return batch_embedding.detach().numpy()
        batch_embedding = self.word_embedding_dropout(batch_embedding).to(self.device)
        return batch_embedding
    
    ######################################################################
    #size of the output of the encoder (== size of the embeddings)
    def get_encoder_n_units(self, encoder):
        return self.word_embedding_dim + self.char_cnn_out_channels
    
    #size of the output layer (== number of classes)
    def get_output_layer_size(self, X, y):
        # print("torch.unique(y).size(0) = ", torch.unique(y).size(0))
        
        if(type(y) == np.ndarray):
            # print("type(np.unique(y)) = ", type(np.unique(y)))
            return len(np.unique(y))
        else:
            return torch.unique(y).size(0)
        
    #number of samples
    def get_n_samples(self, X, y = None):
        return X[0].shape[0]
    
    #returns one single batch
    def get_batch(self, X, y, adresses):
        return (X[0][adresses,:], X[1][adresses,:]), y[adresses]
    
    #CNN feed forward
    def encoder_forward(self, X, y = None):
        output = self.forward_cnn(X)
        output = torch.transpose(output, 1, 2)
        return output.to(self.device)
    
    #torch's forward method
    def forward(self, X, y = None):
        #each encoder might process the input in a different way. as a result,
        #we proceed to have a customized method
        o = self.encoder_forward(X).to(self.device)
        
        #feed forwards through the other layers
        for layer in self.layers:
            o = layer(o).to(self.device)
        
        loss = None #batch loss
        if(y != None):
            d = torch.clone(y).to(self.device)
            
            active_loss = (X[1].view(-1) == 1).to(self.device)
            
            #flattens curr_out
            active_logits = o.view(-1, self.output_layer_size).to(self.device)
            
            active_labels = torch.where(
                active_loss, 
                d.view(-1), 
                torch.tensor(self.loss.ignore_index).type_as(d).to(self.device)
            )
            
            loss = self.loss(active_logits, active_labels).to(self.device)
            
            del active_labels
            del active_logits
            del active_loss
            del d
        
        return o, loss
    
    #method to call for training the model. Pass the dataset and the targets to it.
    def fit(self, X_raw, y_raw, X_vl_raw = None, y_vl_raw = None):
        #crates the cnn
        self.create_encoder()
        
        #dataset preprocessing
        X, y = self.preprocess_dataset(X_raw, y_raw)
        
        X_vl = None
        y_vl = None
        if(X_vl_raw is not None):
            X_vl, y_vl = self.preprocess_dataset(X_vl_raw, y_vl_raw)
        
        #first layer (the encoder: can be bert of whatever)
        self.last_layer_features = self.get_encoder_n_units(self.encoder)
        
        print("self.last_layer_features = ", self.last_layer_features)
        
        #output layer size = number of classes
        self.output_layer_size = self.get_output_layer_size(X, y)
        print("self.output_layer_size = ", self.output_layer_size)
        
        #creates the output layer
        self.create_output_layer()
        
        #rest of the model. we assume that layer_data has been setup by using add_layer
        self.create_model()
        
        self.initialize_optimizer()
        
        self.tot_epochs = 0
        
        #number of samples in the training set (used for the batch generator)
        n_samples_training = self.get_n_samples(X, y)
        batch_generator_training = BatchGenerator(n_samples_training, self.batch_size,
                                                  random_state = self.random_state)
        
        #same thing for the validation set
        if(X_vl is not None):
            n_samples_validation = self.get_n_samples(X_vl, y_vl)
            batch_generator_validation = BatchGenerator(n_samples_validation, self.batch_size,
                                                        random_state = self.random_state)
        
        #training metrics history
        self.tr_loss = []
        self.vl_loss = []
        self.tr_acc  = []
        self.vl_acc  = []
        
        #while (not done):
        while not self.stopping_criterion():
            
            step = 0
            
            #get a batch's addresses
            adr = batch_generator_training.get_batches()
            
            epoch_loss = 0
            
            #actual training step
            for batch in adr:
                def closure():
                    self.optimizer.zero_grad()
                    #gets the batch
                    X_b, y_b = self.get_batch(X, y, batch)
                    
                    #forward
                    output, step_loss = self(X = X_b, y = y_b)
                    
                    #loss backprop
                    step_loss.backward()
                    
                    #accuracy update
                    self.accuracy.update_state(y_b, output)
                    
                    del output
                    return step_loss
                
                #step loss
                step_loss = self.optimizer.step(closure)
                
                #while epoch loss
                epoch_loss += step_loss.detach().cpu().numpy()
                
                step_acc = self.accuracy.result()
                
                #verbose
                if(self.verbose >= 2):
                    print("epoch ", self.tot_epochs, " step ", step,
                          " loss = ", step_loss,
                          ", acc = ", step_acc)
                
                del step_loss, step_acc
                ###########################################################
                step += 1
                
            #metrics update
            epoch_accuracy = self.accuracy.result()
            self.accuracy.reset_state()
            self.tr_acc.append(epoch_accuracy)
            
            
            #validation step
            if(X_vl != None):
                epoch_vl_losses = 0
                vl_adr = batch_generator_validation.get_batches()
                
                vl_step = 0
                for vl_batch in vl_adr:
                    X_v, y_v = self.get_batch(X_vl, y_vl, vl_batch)
                    
                    self.eval()
                    vl_output, vl_step_loss = self(X = X_v, y = y_v)
                    self.train()
                    
                    epoch_vl_losses += vl_step_loss.detach().cpu().numpy()
                    vl_step_acc = self.accuracy.result()
                    
                    self.accuracy.update_state(y_v, vl_output)
                    
                    if(self.verbose >= 3):
                        print("epoch ", self.tot_epochs, " VALIDATION step ", 
                              vl_step, " vl_loss = ", vl_step_loss,
                              ", vl_acc = ", vl_step_acc)
                    
                    del vl_output, vl_step_loss
                    vl_step += 1
                    
                self.vl_loss.append(epoch_vl_losses)
                
                epoch_vl_acc = self.accuracy.result()
                self.accuracy.reset_state()
                self.vl_acc.append(epoch_vl_acc)
                
            self.tr_loss.append(epoch_loss)
            
            if(self.verbose >= 1):
                to_print = "Iteration {:d} loss = {:.7f}, acc = {:.7f}".format(self.tot_epochs, epoch_loss, epoch_accuracy)
                if(not X_vl is None):
                    to_append = "; vl loss = {:.7f}, vl acc = {:.7f}".format(epoch_vl_losses, epoch_vl_acc)
                    to_print = to_print + to_append
                print(to_print)
            
            self.tot_epochs += 1
            