#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:45:18 2021

@author: gardar
"""

import os
import epynet
import yaml
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------
# Importing custom libraries
# --------------------------

# To make sure we don't raise an error on importing project specific 
# libraries, we retrieve the path of the program file ...
filepath = os.path.dirname(os.path.realpath(__file__))

# ... and set that as our working directory
os.chdir(filepath)

from matplotlib.animation import FuncAnimation
from utils.epanet_simulator import epanetSimulator
from utils.data_loader import dataCleaner
from utils.epanet_loader import get_nx_graph
from utils.visualisation import visualise

def read_prediction(filename='predictions.csv', scale=1, bias=0, start_date='2018-01-01 00:00:00'):
    df = pd.read_csv(filename, index_col='Unnamed: 0')
    df.columns = ['n{}'.format(int(node)+1) for node in df.columns]
    df = df*scale+bias
    df.index = pd.date_range(start='2018-01-01 00:00:00',
                             periods=len(df),
                             freq = '5min')
    return df

if __name__ == '__main__':
    
    # Runtime configuration
    path_to_wdn     = './data/L-TOWN.inp'
    path_to_data    = './data/l-town-data/'
    scaling         = 'minmax'
    
    # Import the .inp file using the EPYNET library
    wdn = epynet.Network(path_to_wdn)
    
    # Solve hydraulic model for a single timestep
    wdn.solve()
    
    # Convert the file using a custom function, based on:
    # https://github.com/BME-SmartLab/GraphConvWat 
    G , pos , head = get_nx_graph(wdn, weight_mode='unweighted', get_head=True)
    
    # Instantiate the nominal WDN model
    nominal_wdn_model = epanetSimulator(path_to_wdn, path_to_data)
    
    # Run a simulation
    nominal_wdn_model.simulate()
    
    # Retrieve the nodal pressures
    nominal_pressure = nominal_wdn_model.get_simulated_pressure()
    
    # Open the dataset configuration file
    with open(path_to_data + 'dataset_configuration.yml') as file:
    
        # Load the configuration to a dictionary
        config = yaml.load(file, Loader=yaml.FullLoader) 
    
    # Generate a list of integers, indicating the number of the node
    # at which a  pressure sensor is present
    sensors = [int(string.replace("n", "")) for string in config['pressure_sensors']]
    
    x,y,scale,bias = dataCleaner(pressure_df    = nominal_pressure, # Pass the nodal pressures
                                 observed_nodes = sensors,          # Indicate which nodes have sensors
                                 rescale        = scaling)          # Perform scaling on the timeseries data
    
    p18 = read_prediction(filename='2018_predictions.csv',
                          scale=scale,
                          bias=bias,
                          start_date='2018-01-01 00:00:00')
    p19 = read_prediction(filename='2019_predictions.csv',
                          scale=scale,
                          bias=bias,
                          start_date='2019-01-01 00:00:00')
    
    diff = (p18-p19).copy()
    
    cmap      = plt.get_cmap('hot')                                         # Generate a colourmap
    pred_cmap = [cmap(diff.iloc[i].to_numpy()) for i in range(len(diff))]   # of the predicted signals
    
    print(os.environ['_'])
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,10), sharex=True, sharey=True) 
    
    # Visualise the the model using our visualisation utility
    ax = visualise(G, pos=pos, color=pred_cmap[0], figsize=(16,10), edge_labels=False, axis=ax, cmap=pred_cmap[0])
    
    # Initialise a colorbar
    sm    = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    
    # Plot colorbar
    plt.colorbar(sm, ax=ax)
    
    # Print a heading
    plt.suptitle("Pressure difference 2018-2019", fontsize='xx-large')
    
    def update(n, pos=pos, G=G):
        plt.cla()
        ax.set_title(diff.index[n])
        nx.draw_networkx(G, pos=pos, node_color=pred_cmap[n], with_labels=True, node_size=175, font_size=7, font_color='w')
    
    ani = FuncAnimation(plt.gcf(), update, interval=50, frames=len(pred_cmap), repeat=False, cache_frame_data=True)
    
    plt.rcParams['animation.ffmpeg_path'] = '/opt/anaconda3/envs/GSP/bin/ffmpeg'
    
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=-1)
    
    ani.save('./diff_2018-19.mp4', writer=writer)
