#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+----------------------------+
|                            |
|   D A T A   L O A D E R S  | 
|                            |
+----------------------------+

Created on Fri Jul 23 15:24:59 2021

@author: gardar
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Function to load the timeseries datasets of the BattLeDIM challenge
def dataLoader(observed_nodes, n_nodes=782, path='./BattLeDIM/', file='2018_SCADA_Pressures.csv'):
    '''
    Function for loading the SCADA .csv datasets of the BattLeDIM competition
    and returning it in a dataformat suitable for the GNN model to ingest.

    Parameters
    ----------
    observed_nodes : list of ints
        A list of numerical values indicating the sensors nodal placement.
    n_nodes : int, optional
        Total no. of nodes in the network. The default is 782.
    path : str, optional
        Directory name containing SCADA data. The default is './BattLeDIM/'.
    file : str, optional
        Filename. The default is '2018_SCADA_Pressures.csv'.

    Returns
    -------
    result : np.array(n_obs,n_nodes,2)
        An array of size (n_observations x n_nodes x 2).
        These are then n number of 2-d matrices where the 1st dimension is
        nodal pressure value, and the 2nd dimension is a mask, 1 if the
        pressure value is present (at the observed nodes) and 0 if not
        
        E.g.:
        
        [21.57, 1    <- n1, pressure at node 1 is observed
         0.0  , 0    <- n2, pressure at node 2 is unknown
         0.0  , 0    <- n3
         22.43, 1    <- n4
         0.0  , 0    <- n5
         ...     ]   etc.
    '''
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

# Function to clean the nominal pressure dataframe 
def dataCleaner(pressure_df, observed_nodes, rescale=None):
    '''
    Function for cleaning the pressure dataframes obtained by simulation of the
    nominal system model supplied with the BattLeDIM competition.
    The output format is suitable for ingestion by the GNN model.
    
    Parameters
    ----------
    pressure_df : pd.DataFrame
        Pandas dataframe where: 
            columns (x) = nodes
            index   (y) = observations
    sensor_list : list of ints
        A list of numerical values indicating the sensors nodal placement..
    scaling : str
        'standard' - standard scaling
        'minmax'   - min/max scaling
    Returns
    -------
    x : np.array(n_obs,n_nodes,2)
        The incomplete pressure signal matrix w/ 'n' number of observations.
        This is the feature vector (x) for the GNN model
        
        x =
        [21.57, 1    <- n1, pressure at node 1 is observed
         0.0  , 0    <- n2, pressure at node 2 is unknown
         0.0  , 0    <- n3, ... unknown
         22.43, 1    <- n4, ... observed
         0.0  , 0    <- n5, ... unknown
         ...     ]   etc.
        
    y : np.array(n_obs,n_nodes,2)
        The complete pressure signal matrix w/ 'n' number of observations.
        With this we may train the GNN in a supervised manner.
        
        y =
        [21.57    <- n1, all values are observed
         21.89    <- n2, 
         22.17    <- n3
         22.43    <- n4
         23.79    <- n5
         ...  ]   etc.
        
    '''     
    # The number of nodes in the passed dataframe
    n_nodes = len(pressure_df.columns)
    
    # Rename the columns (n1, n2, ...) to numerical values (1, 2, ...)
    pressure_df.columns = [number for number in range(1,n_nodes+1)]
    
    # Perform scaling on the initial Pandas Dataframe for brevity
    # This is less trivial than applying it on the later generated numpy arrays
    
    # Standard scale:
    if rescale == 'standard':
        scaler      = StandardScaler()
        pressure_df = pd.DataFrame(scaler.fit_transform(pressure_df), columns=pressure_df.columns)
    # Min/max scaling (normalising):
    elif rescale == 'minmax':
        scaler      = MinMaxScaler()
        pressure_df = pd.DataFrame(scaler.fit_transform(pressure_df), columns=pressure_df.columns)
    # Perform no scaling
    else:
        pass
    
    # DataFrame where the index is the node number holding the sensor and the value is set to 1
    sensor_df = pd.DataFrame(data=[1 for i in observed_nodes],index=observed_nodes)
    
    # Filled single row of DataFrame with the complete number of nodes, the unmonitored nodes are set to 0 
    sensor_df = sensor_df.reindex(list(range(1,n_nodes+1)),fill_value=0)
    
    # Find the number of rows in the DataFrame to be masked...
    n_rows = len(pressure_df)
    
    # ... and complete a mask DataFrame, where all the observations to keep are set to 1 and the rest to 0
    mask_df = sensor_df.T.append([sensor_df.T for i in range(n_rows-1)],ignore_index=True)
    
    # Enforce matching indices of the two DataFrames to be broadcast together
    mask_df.index = pressure_df.index
    
    # Generating the incomplete feature matrix (x)
    x_mask = np.array(mask_df)
    x_arr  = np.array(pressure_df.where(cond=mask_df==1,other = 0.0))
    x      = np.stack((x_arr,x_mask),axis=2)
    
    # Generating the complete label matrix (y)
    y_arr  = np.array(pressure_df)
    y      = np.stack((y_arr, ),axis=2)
    
    return x,y
