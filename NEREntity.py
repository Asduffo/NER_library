# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:08:14 2022

@author: devenv
"""

import torch
import numpy as np
import pandas as pd
import copy as cp

from NERBert import NERBert
from NERBase import OptimizerData, LayerData, SchedulerData
from CustomAccuracy import MaskedJaccard

from transformers import BertConfig, BatchEncoding

#NERBert variant for entity linking
class NEREntity(NERBert):
    def __init__(self,
                 #same parameters as before
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
                 apply_get_entities_predict_postprocessing = False):
        
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
            
    #like NERBert, but it also gets the input_ids column, which is what we need for this task
    def preprocess_dataset_dataframe_format(self, X_raw, y_raw = None):
        #modifies the dataset such that the entries which require more
        #than 512 tokens are safely split into different entries.
        X_raw2, y_raw2 = self.split_samples_into_acceptable_lengths(X_raw, y_raw)
        
        
        X_strings = X_raw2.data.tolist()
        
        X_batch_encoding = self.tokenizer(X_strings,
                                          max_length = self.max_tokenization_length,
                                          pad_to_multiple_of = self.max_tokenization_length, 
                                          padding = 'max_length',
                                          truncation = 'only_first',
                                          return_offsets_mapping = True,
                                          )
        
        y = None
        if(y_raw2 is not None):
            y = np.empty(shape = (len(X_batch_encoding['input_ids']),
                                  len(X_batch_encoding['input_ids'][0])),
                         dtype = object)
        
        #for each element (sentence/abstract_id) in the text 
        for i in range(0, len(X_strings)):
            # print("###########################################################")
            
            if(y_raw2 is not None):
                #get all labels records with same abstract_id 
                entities = y_raw2.iloc[i]['data']
                
                
                #for each tag/entity in the instance
                for j in range(len(entities)):
                    entity_data = entities[j]
                    
                    start_char = entity_data[0]
                    #-1 since entity_data[1] is the space after the last character of the entity and it points to nowhere 
                    end_char = entity_data[1] - 1
                    entity_class = entity_data[4]
                    
                    if(entity_data[2] != X_strings[i][(start_char + self.start_char_pad):(end_char + self.end_char_pad)]):
                        raise Exception("control and calculated entity differs:", entity_data[2],
                                        X_strings[i][(start_char + self.start_char_pad):(end_char + self.end_char_pad)])
                    
                    start_token = X_batch_encoding.char_to_token(i, start_char)
                    end_token =  X_batch_encoding.char_to_token(i, end_char)
                    
                    #end_token == None means that we went over 
                    #self.max_tokenization_length tokens
                    if(end_token == None):
                        break
                    
                    y[i, start_token] = str(entity_class)
                    if(end_token > start_token):
                        for k in range(start_token + 1, end_token + 1):
                            y[i, k] = str(entity_class)
        
        X_matrix = np.array(X_batch_encoding['input_ids'])
        a_matrix = np.array(X_batch_encoding['attention_mask'])
        
        if(y_raw2 is not None):
            y = np.where(X_matrix == self.cls_id, self.cls_token, y)
            y = np.where(X_matrix == self.sep_id, self.sep_token, y)
            y = np.where(a_matrix == self.pad_id, self.pad_token, y)
            y = np.where(y == None, 'O', y)
            
            #initializes the dictionary if we didn't do already (useful if we want
            #to encode another dataset (eg: validation/test set) with the same label
            #order)
            if(self.unique_list == None):
                uniques = np.unique(y) #labels (int format)
                labels = np.arange(start = 0, stop = len(uniques)) #labels (int format)
                
                
                #maps
                self.unique_list                = {}
                self.inverse_unique_list        = {}
                
                for A, B in zip(uniques, labels):
                    self.unique_list[A]         = B
                    self.inverse_unique_list[B] = A
            
            #vectorization
            # y = np.vectorize(self.unique_list.get)(y)
            c = 0
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    try:
                        y[i, j] = self.unique_list[y[i, j]]
                    except:
                        # print("WARNING: the key ", y[i, j], " was unrecognized")
                        y[i, j] = self.unique_list['O']
                        c+=1
            
            print(c, " entity ids were not present already")
            print("Padding key = ", self.unique_list['[PAD]'])
            y = torch.from_numpy(y.astype(np.int32)).type(torch.LongTensor).to(self.device)
        
        
        #DEBUG CELL
        # X_matrix = np.array(X_matrix, dtype = np.str_)
        # for i in range(X_matrix.shape[0]):
        #     res = self.tokenizer.convert_ids_to_tokens(X_matrix[i])
        #     X_matrix[i, :] = res
        # return X_matrix, y
        
        X = torch.from_numpy(X_matrix).type(torch.LongTensor).to(self.device)
        a = torch.from_numpy(a_matrix).type(torch.LongTensor).to(self.device)
        
        return [X, a, X_batch_encoding], y
    
    """
    returns a list of entities for each line in X_raw
    """
    def get_entities(self, X_raw : pd.DataFrame):
        X_strings = cp.deepcopy(X_raw)
        
        #splits into tokens of self.max_tokenization_length and proceeds
        X_raw_2, _ = self.split_samples_into_acceptable_lengths(X_strings)
        X_strings, _ = self.preprocess_dataset_dataframe_format(X_raw_2)
        
        #gets the tokens, the attention matrix and the mapping data
        tokens = X_strings[0]
        attention = X_strings[1]
        batch_data = X_strings[2]
        
        #it goes one by one instead than feeding the whole thing in order to
        #avoid out-of-memory issues.
        predictions = torch.zeros((len(tokens), self.max_tokenization_length)).to(self.device)
        self.eval()
        for i in range(len(tokens)):
            o, _ = self([tokens[i].unsqueeze(0), attention[i].unsqueeze(0)])
            
            #converts from probabilities into labels
            o = torch.argmax(o, dim = 2).to(self.device).detach()
            
            #attachs it into the predictions vector
            predictions[i,:] = o
            
        if(self.apply_get_entities_predict_postprocessing):
            predictions = self.get_entities_predict_postprocessing(predictions)
        # self.train()
        
        # print("self.inverse_unique_list", self.inverse_unique_list)
        
        X_raw2_entities = pd.DataFrame(columns = ['abstract_id', 'data'])
        
        #for each input sample
        for i in range(predictions.size(0)):
            curr_sample_entities = []
            
            previous_entity = '[CLF]'
            
            #for each token in the input sample
            #NB: (1, predictions.size(1) - 1) in order to take off cls and sep
            for j in range(self.token_reconstruction_start_offset, predictions.size(1) - self.token_reconstruction_end_offset):
                # print("predictions.size(1) = ", predictions.size(1))
                
                #predictions[i, j] itself is a tensor
                prediction_scalar = predictions[i, j].item()
                
                entity_label = self.inverse_unique_list[prediction_scalar]
                # print("token_label =", token_label)
                
                #safety check
                if(not(entity_label in ['O', '[CLF]', '[SEP]', '[PAD]'])):
                    #if it was classified as B it's a new entity
                    if(entity_label != previous_entity):
                        try:
                            entity_start, entity_end = batch_data.token_to_chars(i, j)
                        except:
                            #we have reached a dead ent on this sample
                            continue
                        
                        #not-so-elegant way to fix an issue with GPT-2 in which it
                        #keeps in a token the space preceeding the token itself,
                        #if present. 
                        #NB: DO THIS ONLY FOR THE FIRST TOKEN of an entity (EG: with tag B-entity)
                        #since intermediate tokens with I- are allowed to begin with a space in there
                        if(X_raw_2.iloc[i]['data'][entity_start] == ' '):
                            entity_start += 1
                        
                        last_entity = (entity_start, 
                                       entity_end, 
                                       X_raw_2.iloc[i]['data'][entity_start:entity_end],
									   '-',
                                       entity_label)
                        
                        curr_sample_entities.append(last_entity)
                    else:
                        last_entity = curr_sample_entities[-1]
                        last_entity_label = last_entity[4]
                        
                        # print("token: ", j, "entity_label: ", entity_label,"last_entity: ", 
                        # last_entity ,"Last_entity_label: ", last_entity_label)

                        #if this is a continuation of the last entity
                        if(entity_label == last_entity_label):
                            #gets the last entity start character
                            last_entity_start = last_entity[0]
                            
                            #gets the final character of this current token
                            try:
                                _, entity_end = batch_data.token_to_chars(i, j)
                            except:
                                #we have reached a dead ent on this sample
                                continue
                            
                            #updates the entity by updating its last character
                            curr_sample_entities[-1] = (last_entity_start, 
                                                        entity_end, 
                                                        X_raw_2.iloc[i]['data'][last_entity_start:entity_end],
                                                        '-',
                                                        entity_label)
                previous_entity = entity_label
                    
            #the whole sample had been inspected. append the found entities here.
            curr_entities = pd.DataFrame(columns = ['abstract_id', 'data'])
            curr_entities.loc[0] = [X_raw_2.iloc[i]['abstract_id'], curr_sample_entities]
            
            X_raw2_entities = pd.concat([X_raw2_entities, curr_entities])
        
        data, entities = self.reconstruct_dataset(X_raw_2, X_raw2_entities)
        self.train()
        return entities
    
    #jaccard as requested by the challenge
    def calculate_accuracy(self, 
                           X_raw : pd.DataFrame, 
                           y_raw : pd.DataFrame,
                           remove_unseen = False):
        jaccards = []
        
        #predicts the entities, and gets them directly into a nice dataframe forma
        predicted_entities = self.get_entities(X_raw)

        #######################################################################
        for i in range(len(y_raw)):
            # print(y_raw.iloc[i])

            #gets the target and predicted entities for a training sample
            sample_tgt_entities = y_raw.iloc[i]['data'].copy()
            sample_prd_entities = predicted_entities.iloc[i]['data'].copy()

            sample_tgt_entities_final = []
            
            #we remove the entities which weren't present
            for j in range(len(sample_tgt_entities)):
                if((sample_tgt_entities[j][4] in self.unique_list and remove_unseen) or (remove_unseen == False)):
                    sample_tgt_entities_final.append(sample_tgt_entities[j])
            
            intersection = []
            if(len(sample_tgt_entities_final) == 0):
                jaccards.append(1)
            else:
                #calculates the intersection of the two sets. 
                for j in range(len(sample_tgt_entities_final)):
                    curr_entity_set_1 = sample_tgt_entities_final[j]
                    
                    for k in range(len(sample_prd_entities)):
                        curr_entity_set_2 = sample_prd_entities[k]
                        
                        #matching
                        if((curr_entity_set_1[0] == curr_entity_set_2[0]) and
                            (curr_entity_set_1[1] == curr_entity_set_2[1]) and
                            (curr_entity_set_1[2] == curr_entity_set_2[2]) and
                            (curr_entity_set_1[4] == curr_entity_set_2[4])):

                            #adds the common element in the intersection
                            # print("found intersection at ", j, ", ", k, " = ", curr_entity_set_1)
                            intersection.append(curr_entity_set_1)

                #sample jaccard: |P⋂O| / (|P| + |O| - |P⋂O|)
                #where P and O are the predicted and target entities (which is which
                #doesn't really matter here)
                curr_jaccard = len(intersection)/(len(sample_tgt_entities_final) + len(sample_prd_entities) - len(intersection))
                
                jaccards.append(curr_jaccard)
        
        #mean jaccard = mean 
        return float(sum(jaccards))/float(len(jaccards)), jaccards
    
    def split_samples_into_acceptable_lengths(self, X_raw_o, y_raw_o = None):
        X_raw = X_raw_o.copy()
        
        y_raw = None
        if(y_raw_o is not None):
            y_raw = y_raw_o.copy()
        
        columns_names = ['abstract_id', 'data']
        
        X = pd.DataFrame(columns = columns_names)
        
        y = None
        if(y_raw is not None):
            y = pd.DataFrame(columns = columns_names)
        
        for i in range(len(X_raw)):
            # print("###########################################################")
            # print("iteration", i)
            
            current_string = X_raw.iloc[i]['data']
            
            # print("type(current_string) =", type(current_string))
            # print("current_string =", current_string)
            
            #first of all tokenizes the entity in order to see if we can leave
            #the row as is (hence, #tokens <= self.max_tokenization_length)
            tokens = self.tokenizer(current_string,
                                    
                                    #cannot do much about it. max length must
                                    #be a very high number.
                                    max_length = self.max_tokenization_length * 1000,
                                    
                                    padding = False,
                                    pad_to_multiple_of = self.max_tokenization_length,
                                    truncation = False)['input_ids']
            tokens_length = len(tokens)
            # print(tokens_length)
            
            #the sample is small enough to not require splitting
            if(tokens_length <= self.max_tokenization_length):
                to_append = X_raw.iloc[i].to_frame().T
                X = pd.concat([X, to_append], ignore_index = True)
                
                if(y_raw is not None):
                    to_append_y = y_raw.iloc[i].to_frame().T
                    y = pd.concat([y, to_append_y], ignore_index = True)
                
            else: #else we split
                current_id = X_raw.iloc[i]['abstract_id']
                
                current_entities = None
                if(y_raw is not None):
                    current_entities = y_raw.iloc[i]['data']
                
                # print("\n\ncurrent_entities", current_entities)
                # print("type(current_entities) = ", type(current_entities))
                
                #splits only on the first occurrence
                splitted_string = current_string.split(". ", 1)
                
                #if we cannot split it, there is nothing we can do.
                if(len(splitted_string) < 2):
                    X = pd.concat([X, X_raw.iloc[i]], ignore_index = True)
                    
                    if(y_raw is not None):
                        y = pd.concat([y, y_raw.iloc[i]], ignore_index = True)
                    
                    #moves straight to the next element
                    continue
                else:
                    #gets left and right string
                    string_sx = splitted_string[0]
                    string_dx = splitted_string[1]
                    
                    #since the split operation deletes the '. ', we reinsert it here
                    string_sx += '. '
                    
                    string_sx_entities = []
                    string_dx_entities = []
                    
                    if(y_raw is not None):
                        for j in range(len(current_entities)):
                            curr_entity = current_entities[j]
                            string_sx_len = len(string_sx)
                            
                            if(curr_entity[1] <= string_sx_len):
                                #the entity is inside the left string completely
                                #so we can just append it.
                                
                                string_sx_entities.append(curr_entity)
                            else:
                                #else it is either partially or completely on the right
                                if(curr_entity[0] >= string_sx_len):
                                    #completely on the right
                                    #we need to decrease start and end
                                    transformed_entity = (curr_entity[0] - string_sx_len,
                                                          curr_entity[1] - string_sx_len,
                                                          curr_entity[2],
														  '-',
                                                          curr_entity[4])
                                    
                                    string_dx_entities.append(transformed_entity)
                                else:
                                    #painful part: it's half on the left and half on
                                    #the right.
                                    entity_sx = (curr_entity[0], #start is unchanged
                                                 string_sx_len,  #end of the string
                                                 curr_entity[2],
												 '-',
												 curr_entity[4])
                                    
                                    entity_dx = (0, #start = string begin
                                                 curr_entity[1] - string_sx_len, #takes string_sx_len from the end
                                                 curr_entity[2], 
												 '-',
                                                 curr_entity[4])
                                    
                                    string_sx_entities.append(entity_sx)
                                    string_dx_entities.append(entity_dx)
                    
                    X_sx = pd.DataFrame(columns = columns_names)
                    X_sx.loc[0] = [current_id, string_sx]
                
                    
                    #converts string_dx and string_dx_entities into a dataframe
                    X_dx = pd.DataFrame(columns = columns_names)
                    X_dx.loc[0] = [current_id, string_dx]
                    
                    y_sx = None
                    y_dx = None
                    if(y_raw is not None):
                        y_dx = pd.DataFrame(columns = columns_names)
                        y_dx.loc[0] = [current_id, string_dx_entities]
                        
                        y_sx = pd.DataFrame(columns = columns_names)
                        y_sx.loc[0] = [current_id, string_sx_entities]
                    
                    X_ret, y_ret = self.split_samples_into_acceptable_lengths(X_dx, y_dx)
                    
                    # return X_ret, None
                    
                    X = pd.concat([X, X_sx], ignore_index = True)
                    X = pd.concat([X, X_ret], ignore_index = True)
                    
                    if(y_raw is not None):
                        y = pd.concat([y, y_sx], ignore_index = True)
                        y = pd.concat([y, y_ret], ignore_index = True)
        
        # y["abstract_id"] = pd.to_numeric(y["abstract_id"])
        return X, y
    
    
    """
    this method finds the entities with the same ID in a list and merges them together
    """
    def reconstruct_dataset(self, X_raw, y_raw = None):
        #gets the various articles_ids
        articles_ids = X_raw.abstract_id.unique()
        
        #creates the dataframes to return
        to_return_X = pd.DataFrame(columns = ['abstract_id', 'data'])
        to_return_y = None
        if(y_raw is not None):
            to_return_y = pd.DataFrame(columns = ['abstract_id', 'data'])
        
        i = 0
        #reconstructs one article at a time
        for art_id in articles_ids:
            # print(art_id)
            
            #current article
            X_abstracts = X_raw.loc[X_raw['abstract_id'] == art_id]
            
            #current entities
            y_entities = None
            if(y_raw is not None):
                y_entities = y_raw.loc[y_raw['abstract_id'] == art_id]
            
            X_abstracts = X_abstracts['data'].tolist()
            y_entities  = y_entities['data'].tolist()
            
            # return X_abstracts, y_entities
            
            rec_x, rec_y = self.reconstruct_sample_into_single_arcticles(X_abstracts, y_entities)
            
            to_append_x = pd.DataFrame(columns = ['abstract_id', 'data'])
            to_append_x.loc[0] = [art_id, rec_x]
            to_return_X = pd.concat([to_return_X, to_append_x])
            
            to_append_y = pd.DataFrame(columns = ['abstract_id', 'data'])
            to_append_y.loc[0] = [art_id, rec_y]
            to_return_y = pd.concat([to_return_y, to_append_y])
            
            i += 1
        
        return to_return_X, to_return_y
    
    #NOTE: this reconstructs only ONE element
    def reconstruct_sample_into_single_arcticles(self, X_raw, y_raw = None):
        current_string = X_raw[0]
        
        current_entities = None
        if(y_raw is not None):
            current_entities = y_raw[0]
        
        #1 cause X_raw[0] is already in current_string
        #iterates over the next <string chunk, entities of that string chunk> pairs
        for i in range(1, len(X_raw)):
            old_str_len = len(current_string)
            current_string = current_string + X_raw[i]
            
            
            curr_string_ent = y_raw[i]
                    
            for j in range(len(curr_string_ent)):
                curr_ent = curr_string_ent[j]
                
                # print("len(curr_ent) =", len(curr_ent))
                # print("type(curr_ent) =", type(curr_ent))
                # print("curr_ent =", curr_ent)
                
                new_entity = (int(curr_ent[0]) + old_str_len,
                              int(curr_ent[1]) + old_str_len,
                              curr_ent[2], 
							  ' ',
                              curr_ent[4])
                
                current_entities.append(new_entity)
        return current_string, current_entities
		
    #unused by this particular class
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
                
                try :
                    start_index = curr_sample.index(self.unique_list[self.cls_token])
                except:
                    start_index = 1
                    
                try:
                    end_index = curr_sample.index(self.unique_list[self.sep_token])
                except:
                    end_index = self.max_tokenization_length
                
                
                for t in curr_sample[start_index + 1:end_index]:
                    try:
                        l = l + [self.inverse_unique_list[t]]
                    except:
                        l = l + ["O"]
            p.append(l)
        return p