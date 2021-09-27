# Graph-Based Learning for Leak Detection
**Graph Signal Processing** (GSP) refers to a branch of **signal processing** models and algorithms that handle data supported on graphs.

This research project will investigate the application of graph-based learning methods in **water distribution networks** (WDN) with the objective of detecting and localising (isolating) leaks.

This `GitHub` repository will be used for version management, development and sharing of code by the researcher, *Gardar Orn Gardarsson*, with the project's supervisors, *Dr. Francesca Boem* and *Dr. Laura Toni*.



# `01 Graph Import` 

**Geographical Information System** (GIS) data, in the `shapefile` data format, is used for the creation of a `networkx` graph, the result of which is shown in the figure above.

This graph is representative of a physical, topological network structure, from which a computation graph is derived. The below **District Metered Area** is that of *Álftanes* in the capital region of Iceland. The import procedure of the GIS model and its conversion to a graph is contained in the directory

<img src="./01 Graph Import/images/wdn_as_a_graph.png" alt="Álftanes WDN as a Graph" style="zoom:20%;" />

 # `02 Data Pre-Processing`

This directory holds notebooks where pre-processing and **Exploratory Data Analysis** (EDA) is performed on the time-series sensor data accommodating the real-world data from *Álftanes* DMA.

<img src="./02 Data Preprocessing/images/time_series_distributions.png" alt="Time-Series Distributions" style="zoom:70%;" />

# `03 Hydraulic Simulations`

Various *open-source datasets* are available for the leakage detection problem. The approach for this project will be to develop a graph-based learning method for leakage detection on such a dataset, validating it on an open-source *benchmark* for comparison with the state-of-the-art, and finally applying the learned model, or the methodology, to the real-world data in `01-02`.

This folder thus contains `Python` scripts and `Jupyter` development notebooks for working with the **BattLeDIM** (**Batt**le of the **Le**akage **D**etection and **I**solation **M**ethods) dataset. 

This dataset delivers the hypothetical WDN of *L-Town*, the topology of which is contained within an **EPANET** input file, `.inp`. EPANET is a hydraulic simulation tool, and a niche toolset is required to import the model and convert it to a `networkx` graph. Having converted the model to a graph, one may generate renderings of it, as per the image below. 

<img src="./Map.png" alt="L-Town" style="zoom:100%;" />

Unmonitored nodes are plotted <span style="color:blue"> *blue* </span> and nodes with pressure sensors installed are plotted <span style="color:red"> *red* </span>. <br>From this, the objective will be to *infer* the pressure signals at the unobserved nodes, i.e. reconstructing the graph signal, and will be achieved with a GNN that uses a Chebyshev polynomial kernel.<br>Another model is trained to *predict* the pressure signals for the next timestep, from a window of previous observations. 

Having done so, an opportunity is created for validating the predictions against the reconstructions, thereby obtaining a prediction error signal, a residual signal.

With it, one may start measuring discrepancies.

Using deviation based thresholding, we can then nominate a list of candidates as leaky using this approach.

This directory, primarily holds the files that have to do constructing the GNNs (*predictor* and *reconstructor*) from the WDN topology, and their training.

# `04 Leakage Detection`

This directory holds the many notebooks (`00-09 Leak Detection: ... `) that were created to work with the outputs from the GNNs.<br>Predictions and reconstructions were generated for the whole of the year 2018, and 2019 with the GNNs. <br>The 2018 data is training data where the leakages are known and we may validate our detection method on. <br>The goal is then to predict the leaks that occurred in 2019. 

The notebooks show the development of a statistical method to classify leakages from the residual signals.

While the method presented in the dissertation is that of [*Boem et al*](https://discovery.ucl.ac.uk/id/eprint/10051702/1/DF_dataseries_final_sub.pdf), an implementation of `CUSUM` was further tested, and a time-series classifier, `InceptionTime` was also trained to classify the leakages. 

This directory holds the code behind this development, i.e. going from residual signals to leak detection.



# Acknowledgements

The code for importing the *EPANET* files, and converting it to a graph for signal reconstruction builds on the work of Gergely Hajgató et al., available on: https://github.com/BME-SmartLab/GraphConvWat.