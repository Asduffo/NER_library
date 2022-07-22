# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:21:17 2022

@author: devenv
"""

import copy as cp
import torch
import matplotlib.pyplot as plt

from BatchGenerator import BatchGenerator
from CustomAccuracy import MaskedJaccard

#this allows us to initialize the optimizer only when the parameters have been defined
#(hence: after create_model()). Otherwise we would need to declare it immediately and
#this is not particularly flexible 
class OptimizerData():
    def __init__(self,
                 optimizer_type : torch.optim,
                 **kwargs):
        self.optimizer_type = optimizer_type
        self.kwargs         = kwargs
        
#Like OptimizerData, but for the learning rate scheduler
class SchedulerData():
    def __init__(self,
                 scheduler_type : torch.optim.lr_scheduler,
                 **kwargs):
        self.scheduler_type = scheduler_type
        self.kwargs         = kwargs
        
#Hidden layers' data
class LayerData():
    def __init__(self,
                 layer,
                 **kwargs):
        self.layer = layer   #layer type
        self.kwargs = kwargs #layer's parameters

#abstrat class containing the core of all the networks used
class NERBase(torch.nn.Module):
    def __init__(self,
                 max_epochs : int = 10,                         #number of training epochs
                 batch_size : int = 1,                          #batch size
                 accumulate_batch_size : int = None,            #if not None, accumulate the gradients in order to simulate a batch size of "accumulate_batch_size" elements
                 loss : torch.nn = torch.nn.CrossEntropyLoss(), #loss
                 accuracy = MaskedJaccard(pad_id = 14),         #accuracy
                 optimizer_data : OptimizerData = OptimizerData(torch.optim.SGD, lr = .01), #optimizer
                 scheduler_data : SchedulerData = SchedulerData(torch.optim.lr_scheduler.ExponentialLR, gamma = .9), #scheduler
                 random_state : int = 0,                        #batch generator's random state
                 verbose : int = 0,                             #0: don't print anything during the training. 1: print the epoch loss, 2: print the epoch and its step's loss
                 device = 'cpu',                                #cpu or cuda
                 use_softmax_output = False):                   #unused: whether we should use a linear output or a softmax output
        super().__init__()
        
        #TODO: check vari
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.accumulate_batch_size = accumulate_batch_size
        self.loss = loss
        self.accuracy = accuracy
        self.optimizer_data = optimizer_data
        self.scheduler_data = scheduler_data
        
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        
        self.use_softmax_output = use_softmax_output
        
        self.output_layer_data = LayerData(torch.nn.Linear, out_features = 10)
        
        self.layers_data = []    #layer_data contains the PRELIMINARY DATA (LayerData)
        self.layers = []         #whole layers contains actual torch.nn.layers
        
        if(self.accumulate_batch_size == None):
            self.accumulate_batch_size = self.batch_size
        
    #abstract method which is supposed to implement the encoder (BERT/GPT2/LSTM etc) instantiation
    def create_encoder(self):
        pass
    
    #initializes the optimizer (we do it here because we need the finished model
    #in order to call self.parameters()).
    #CALL ONLY AFTER create_model(),
    def initialize_optimizer(self):
        self.optimizer = self.optimizer_data.optimizer_type(self.parameters(),
                                                            **(self.optimizer_data.kwargs))
        self.scheduler = self.scheduler_data.scheduler_type(self.optimizer,
                                                            **(self.scheduler_data.kwargs))
    
    #add_layer itself just saves the LayerData of the layer. create_model is
    #the one who actually initializes it.
    def add_layer(self, new_layer_data : LayerData):
        self.layers_data.append(new_layer_data)
    
    def initialize_layer(self, data : LayerData, name : str):
        
        print("==============================================")
        print("type(data.layer) = ", data.layer.__name__)
        print("data.kwargs = ", data.kwargs)
        
        if(data.layer.__name__ == 'Linear'):
            print("adding linear layer")
            print("self.last_layer_features = ", self.last_layer_features)
            layer = data.layer(**(data.kwargs), in_features = self.last_layer_features)
            self.last_layer_features = layer.out_features
        else:
            print("adding non linear layer")
            layer = data.layer(**(data.kwargs))
        
        print("type(layer) = ", type(layer))
        self.add_module(name, layer.to(self.device))
        self.layers.append(layer)
        print("==============================================")
    
    #initializes the hidden layer according to the given data.
    def create_model(self):
        i = 0
        for data in self.layers_data:
            self.initialize_layer(data, "layer_" + str(i))
            i = i + 1
        
        #adds the hidden layer at the end
        self.initialize_layer(self.output_layer_data, "layer_out")
    
    #used to define the initial value of self.last_layer_features
    def get_encoder_n_units(self, encoder):
        pass
    
    #used to retrieve the output's size (hence, how many classes do we have)
    def get_output_layer_size(self, X, y):
        pass
    
    #tells us how many samples are there in the dataset.
    #each dataset might be structured in a different way depending on the encoder
    #used, so we delegate the task to the child classes
    def get_n_samples(self, X, y = None):
        pass
    
    #given a subset of the dataset described by the indexes in addresses, returns
    #such subset from X and y. Each dataset might be different (eg: X might be made
    #just by token ids or by an array of ids and attention masks), we delegate it to
    #the child class.
    def get_batch(self, X, y, adresses):
        pass
    
    #preprocesses the dataset (abstract method)
    def preprocess_dataset(self, X_raw, y_raw = None):
        return X_raw, y_raw
    
    #instantiates the final layer
    def create_output_layer(self):
        if(self.use_softmax_output):
            print("creating softmax")
            self.add_layer(LayerData(layer = torch.nn.Linear, out_features = self.output_layer_size))
            self.output_layer_data = LayerData(layer = torch.nn.Softmax, dim = 2)
        else:
            self.output_layer_data = LayerData(layer = torch.nn.Linear, out_features = self.output_layer_size)
    
    #training method. Must be callse by the user. X_raw_o and y_raw_o are the
    #dataframes containing the training instances and the targets. 
    #X_vl_raw_o and y_vl_raw_o are the validation set
    def fit(self, X_raw_o, y_raw_o, X_vl_raw_o = None, y_vl_raw_o = None):
        self.create_encoder()
        
        #safety copy
        X_raw = cp.deepcopy(X_raw_o)
        y_raw = cp.deepcopy(y_raw_o)
        
        #preprocesses the dataset
        X, y = self.preprocess_dataset(X_raw, y_raw)
        
        X_vl = None
        y_vl = None
        if(X_vl_raw_o is not None):
            X_vl_raw = cp.deepcopy(X_vl_raw_o)
            y_vl_raw = cp.deepcopy(y_vl_raw_o)
            
            X_vl, y_vl = self.preprocess_dataset(X_vl_raw, y_vl_raw)
        
        #first layer (the encoder: can be bert of whatever)
        self.last_layer_features = self.get_encoder_n_units(self.encoder)
        
        print("self.last_layer_features = ", self.last_layer_features)
        
        self.output_layer_size = self.get_output_layer_size(X, y)
        print("self.output_layer_size = ", self.output_layer_size)
        
        self.create_output_layer()
        
        
        #rest of the model. we assume that layer_data has been setup by using add_layer
        self.create_model()
        
        #optimizer setup
        self.initialize_optimizer()
        
        self.tot_epochs = 0
        
        #retrieves some useful informations for the batch generator
        n_samples_training = self.get_n_samples(X, y)
        batch_generator_training = BatchGenerator(n_samples_training, self.batch_size,
                                                  random_state = self.random_state)
        if(self.accumulate_batch_size == 0):
            self.accumulate_batch_size = n_samples_training
        
        #same thing for the validation set
        if(X_vl is not None):
            n_samples_validation = self.get_n_samples(X_vl, y_vl)
            batch_generator_validation = BatchGenerator(n_samples_validation, self.batch_size,
                                                        random_state = self.random_state)
        
        self.tr_loss = []
        self.vl_loss = []
        self.tr_acc  = []
        self.vl_acc  = []
        
        #real training loop
        while not self.stopping_criterion():
            step = 0
            
            #gets the current batch's addresses
            adr = batch_generator_training.get_batches()
            
            epoch_loss = 0
            
            #gradient reset
            self.optimizer.zero_grad()
            
            #actual training step
            for batch in adr:
                #gets the training batch data
                X_b, y_b = self.get_batch(X, y, batch)
                
                #feed forward
                output, step_loss = self(X = X_b, y = y_b)
                
                #backprop
                step_loss.backward()
                
                #batch accumulation + optimizer call
                if(self.accumulate_batch_size != None):
                    if (step+1) % self.accumulate_batch_size == 0 or (step+1) == len(adr):
                        self.optimizer.step()  # update the weights only after accumulating k small batches
                        
                        # if(self.verbose >= 2):
                        #     print("epoch ", self.tot_epochs, " step ", step, " optimization step.")
                else:
                    self.optimizer.step()  # update the weights only after accumulating k small batches
                
                self.optimizer.zero_grad()  # reset gradients for accumulation for the next large_batch

                #metrics update                
                self.accuracy.update_state(y_b, output)
                step_acc = self.accuracy.result()
                
                epoch_loss += step_loss.detach().cpu().numpy()
                
                #verbose
                if(self.verbose >= 2):
                    print("epoch ", self.tot_epochs, " step ", step,
                          " loss = ", step_loss,
                          ", acc = ", step_acc)
                    
                del output, step_loss, step_acc
                
                step += 1
                
            #other metrics updates
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
                    
                self.vl_loss.append(epoch_vl_losses/n_samples_validation)
                
                epoch_vl_acc = self.accuracy.result()
                self.accuracy.reset_state()
                self.vl_acc.append(epoch_vl_acc)
                
            self.tr_loss.append(epoch_loss/n_samples_training)
            
            if(self.verbose >= 1):
                to_print = "Iteration {:d} loss = {:.7f}, acc = {:.7f}".format(self.tot_epochs, epoch_loss/n_samples_training, epoch_accuracy)
                if(not X_vl is None):
                    to_append = "; vl loss = {:.7f}, vl acc = {:.7f}".format(epoch_vl_losses/n_samples_validation, epoch_vl_acc)
                    to_print = to_print + to_append
                print(to_print)
            
            self.tot_epochs += 1
            
            self.scheduler.step()
        
    #stopping criterion (here it's only current epoch <= number of epochs) 
    def stopping_criterion(self):
        condition1 = self.tot_epochs >= self.max_epochs
        
        return condition1
        
    #feed forward of BERT/GPT2/whatever
    def encoder_forward(self, X, y = None):
        pass
        
    
    #uses the accuracy metric
    def score(self, X, y):
        X_b, y_b = self.preprocess_dataset(X, y)
        
        with torch.no_grad():
            self.eval()
            output, loss = self(X = X_b, y = y_b)
            self.train()
        
        self.accuracy.update_state(y_b, output)
        acc = self.accuracy.result()
        
        l = loss.detach()
        del loss, X_b, y_b
        
        return l, acc
    
    #used by __call__
    def forward(self, X):
        pass
        
    #predicts the output of one single sample
    def predict(self, X):
        X_b, _ = self.preprocess_dataset(X, None)
        
        with torch.no_grad():
            self.eval()
            output, _ = self(X_b)
            self.train()
        
        result = torch.argmax(output, dim=2).to(self.device).detach()
        del output, X_b
        
        return result
    
    #plots the loss (ended up unused)
    def plot(self, name, ylim = None):
        """ Plot the results """
        plt.figure(dpi=500)
        plt.xlabel('epoch')
        
        assert(len(name) <= 2), "NeuralNetwork.plot: max 2 losses!"
        
        
        plot_colors = ['b-', 'r--']
        c = 0
        
        for line in name:
            if(line == 'tr_loss'):
                plt.plot(self.tr_loss, plot_colors[c], label='Training loss')
                plt.ylabel("Loss")
            elif(line == 'tr_acc'):
                plt.plot(self.tr_acc, plot_colors[c], label='Training accuracy')
                plt.ylabel("Accuracy")
            elif(line == 'vl_loss'):
                plt.plot(self.vl_loss, plot_colors[c], label='Validation loss')
                plt.ylabel("Loss")
            elif(line == 'vl_acc'):
                plt.plot(self.vl_acc, plot_colors[c], label='Validation accuracy')
                plt.ylabel("Accuracy")
            else:
                raise Exception("Unrecognized loss name")
            c = c + 1
        
        if(not ylim is None):
            plt.ylim(ylim)
        
        plt.legend(fontsize  = 'large')
        plt.show()
    
