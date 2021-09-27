clc; clear all; close all; 

error_by_pipe = readtable('error_by_pipe_2019.csv');

%%

IndicesOfQuantFeatures = varfun(@isnumeric,error_by_pipe,'output','uniform');
numericFeatures        = error_by_pipe(:,IndicesOfQuantFeatures);

%%
pipe_vec               = table2array(error_by_pipe(:,'p31'))

%%
cusum(pipe_vec)

%%
X = linspace(1, length(pipe_vec), 1)
%%
pipe_vec(2)

%%
figure

plot(X,pipe_vec)

axis padded
