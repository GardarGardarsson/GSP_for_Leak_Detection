#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+---------------------------+
|                           |
|   D A T A   L O A D  E R  | 
|                           |
+---------------------------+

Created on Fri Jul 23 15:24:59 2021

@author: gardar
"""

import numpy as np
import pandas as pd

# Function to load the timeseries datasets of the BattLeDIM challenge
def dataLoader(observed_nodes, n_nodes=782, path='./BattLeDIM/', file='2018_SCADA_Pressures.csv'):
    
    # Read the file at the passed destination into a Pandas DataFrame
    df = pd.read_csv(str(path + file), sep=';', decimal=',')
    
    # Set the 'Timestamp' column as the index
    df = df.set_index('Timestamp')
    
    # Set the column names as the numeric list passed into the function
    # which states what nodes of the graphs are observed
    df.columns = observed_nodes
    
    # Generate a temporary image of the DataFrame, that's been filled with zeros
    # at the un-observed nodes
    temp = df.T.reindex(list(range(1,n_nodes+1)),fill_value=0.0)
    
    # Create a "mask" array, that's set to 1 at the observed nodes and 0 otherwise
    arr2 = np.array(temp.mask(temp>0.0,1).astype('int'))

    # Create a numpy array from the temporary image
    arr1 = np.array(temp)

    # Stack and transpose the observation and mask arrays
    result = np.stack((arr1,arr2),axis=0).T
    
    # Return the results
    return result