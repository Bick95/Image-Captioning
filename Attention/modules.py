# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

# Imports
import tensorflow as tf


class SoftAttention(tf.keras.Model):
    def __init__(self, units):
        """
            units:      number of internal units per layer
        """
        super(SoftAttention, self).__init__()
        
        # TODO
        
        pass

    def call(self, features, hidden):
        """
            features:   features observed from image
            hidden:     hidden state from previous iteration
        """
        
        # TODO
        
        return None
    
class HardAttention(tf.keras.Model):
    def __init__(self, units):
        """
            units:      number of internal units per layer
        """
        super(HardAttention, self).__init__()
        
        # TODO
        
        pass

    def call(self, features, hidden):
        """
            features:   features observed from image
            hidden:     hidden state from previous iteration
        """
        
        # TODO
        
        return None
    

