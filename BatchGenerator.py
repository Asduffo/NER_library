# -*- coding: utf-8 -*-
"""
Contain the class implenting the generator of the batches for our machine learning model

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np
from random import Random

class BatchGenerator():
    """ Implement the generator of the batches """
    
    def __init__(self,
                 dataset_size,
                 batch_size,
                 random_state = None):
        """
        Parameters:
        dataset_size (int) : Dataset size. 
        batch_size (int): Batch size. If it is equal to 0, we use batch learning.
        random_state (int, None): Seed for the rng. If not None, the results are replicable.
        """
        
        assert(isinstance(dataset_size, int) and dataset_size > 0), "BatchGenerator.__init__: dataset_size must be an integer > 0"
        assert(isinstance(batch_size, int)), "BatchGenerator.__init__: batch_size must be an integer"
        assert(isinstance(random_state, (int, type(None)))), "BatchGenerator.__init__: random_state must be an integer or None"
        
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.random_state = random_state
        
        # In the case of batch_size higher than the dataset itself, we just repair it
        # and set it as the dataset size
        if((self.batch_size > self.dataset_size) or
           self.batch_size <= 0):
            self.batch_size = self.dataset_size
    
    def get_batches(self):
        """
        Returns: 
        np.ndarray: List of lists, where each list contains the indexes for a certain batch. 
        """
        
        indexes = np.arange(start = 0, 
                            stop = self.dataset_size, 
                            step = 1, dtype=int)
        if(self.random_state == None):
            Random().shuffle(indexes)
        else:
            Random(self.random_state).shuffle(indexes)
            self.random_state = self.random_state + 1
        
        adr = []
        start = 0
        while(start < self.dataset_size):
            batch = indexes[start:(start + self.batch_size)]            
            adr.append(batch)
            start = start + self.batch_size

        return adr