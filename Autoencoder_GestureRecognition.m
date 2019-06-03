clear;
clc;
tic;

%Loading Data
fprintf('\nLoading data.\n')

%Dataset 1
%parameters for using this data
%hiddenSize1 = 300
%autoenc1 iterations = 300
%trainSoftmaxLayer iterations = 300
%Accuracy = 87.5
% load('training_label_4g200i_singlesubcarrier.mat')
% load('training_data_4g200i_singleinstance.mat')
% load('testing_label_4g200iSingleInstance.mat')
% load('testing_data_4g200i_singleinstance.mat')

%Dataset 2
%parameters for using this data
%hiddenSize1 = 150
%autoenc1 iterations = 750
%trainSoftmaxLayer Iterations = 500
%Accuracy = 70.5
% load('data_clapsnap_30.mat')


%Dataset 3
%parameters for using this data
%hiddenSize1 = 300
%autoenc1 iterations = 1000
%trainSoftmaxLayer iterations = 300
%Accuracy = 88.3
load('data_3p2g_singleinstance.mat')

training_data = training_data';
testing_data = testing_data';

%random number generator seed.
rng('default')

hiddenSize1 = 300;

fprintf('\nTraining Autoencoder.\n')
autoenc1 = trainAutoencoder(training_data,hiddenSize1, ...
    'MaxEpochs',1000, ...
    'L2WeightRegularization',0.001, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.15, ...
    'ScaleData', true);

feat1 = encode(autoenc1,training_data);

% Training a Softmax Layer 
fprintf('\nTraining Softmax Layer.\n')

softnet = trainSoftmaxLayer(feat1,training_label,'MaxEpochs',300);

% Forming a stacked neural network

stackednet = stack(autoenc1,softnet);

% view(stackednet);

train1 = stackednet(training_data);
test1 = stackednet(testing_data);

% Fine tune Traning Data
fprintf('\nFine tuning for training data\n')
stackednet_train = train(stackednet,training_data,training_label);
% view(stackednet_train);

train2 = stackednet_train(training_data);
test2 = stackednet_train(testing_data);

plotconfusion(training_label,train1,"Training data without Fine Tuning",...
training_label,train2,"Training data with Fine Tuning",...
testing_label,test1,"Testing data without Fine Tuning", ...
testing_label,test2,"Testing data with Fine Tuning");
toc;