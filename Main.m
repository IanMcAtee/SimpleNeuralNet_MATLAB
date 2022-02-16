%Project 03: Neural Network
%Author: Ian McAtee and Stein Wiederholt
%Date: 11/23/2021
%DESCRIPTION: Main function for EE5650 Project 3. Function can find a
    %well-performing neural network for the training and test data, or
    %users can specify their own network configuration, or one can simply
    %use the weights provided in FinalWeights12.mat to examine the
    %predefined weights found by the authors with a 12 hidden node network

close all
clear all

%% DATA SETUP
%Load in data
load('training2.mat')
load('test2.mat')

%Load in weights found previously by the author
load('FinalWeights12.mat') 

%Set number of training samples for each class
N = [1000,600,400];
totalN = sum(N);

%Form the training data
train1 = class1_train(1:N(1),:);
train2 = class2_train(1:N(2),:);
train3 = class3_train(1:N(3),:);
train = [train1;train2;train3];

%Form the validation data
val1 = class1_train(N(1)+1:N(1)*2,:);
val2 = class2_train(N(2)+1:N(2)*2,:);
val3 = class3_train(N(3)+1:N(3)*2,:);
val = [val1;val2;val3];

%Form the test data
test1 = class1_test(1:N(1),:);
test2 = class2_test(1:N(2),:);
test3 = class3_test(1:N(3),:);
test = [test1;test2;test3];

%Form labels
labels = [ones(N(1),1);2*ones(N(2),1);3*ones(N(3),1)];

%Preprocess training data and get preprocessing parameters
fprintf('PREPROCESSING DATA\n')
[trainProcess,preProcessParam] = preprocessTrain(train,[N]);

%Setup a structure for the neural network architecture and parameters
%These hyperparameters can be changed by the user
initialNNParam.hiddenNodes = 18;
initialNNParam.learnRate = 0.01;
initialNNParam.maxEpochs = 1000; 

%% NEURAL NETWORK TRAINING
%Leave only one ****section**** uncommented

%Default to setting the trained network to the optimal found previously
%**********************************************************************
trainedNNParam.wj = wj;
trainedNNParam.wk = wk;
%**********************************************************************

%Train several networks and select the best
%**********************************************************************
% %Set the number of networks tested for each number of hid units instance
% %Depending on how you set these values, this can take a long time 
% numberNetworks = 10;
% maxHidUnits = 32;
% minHidUnits = 6;
% count = 0;
% %Loop through networks with differing number of hidden units
% for i = minHidUnits:2:maxHidUnits
%     fprintf('TRAINING NEURAL NETWORK\n')
%     count = count+1;
%     errors = zeros(numberNetworks,1);
%     initialNNParam.hiddenNodes = i; %Set number of hidden units
%     %Train networks and evaluate performace, tabulate error
%     for j = 1:numberNetworks
%         trainedNN(j).Param = trainNN(trainProcess,val,labels,initialNNParam,preProcessParam);
%         class = evaluateNN(test,trainedNN(j).Param,preProcessParam);
%         errors(j) = length(find(labels-class))/totalN * 100;
%     end
%     %Find network with lowest error
%     optimal = find(errors == min(errors));
%     opError(count) = errors(optimal(1));
%     trainedNNParam.wj = trainedNN(optimal(1)).Param.wj;
%     trainedNNParam.wk = trainedNN(optimal(1)).Param.wk;
%     %Save networks as optimal for that num of hid units 
%     save(int2str(i),'-struct','trainedNNParam');
% end
% %Set the NN to the best performing network overall
% overallOp = find(opError == min(opError));
% overallOpIndex = minHidUnits + 2*(overallOp(1)-1);
% load(strcat(int2str(overallOpIndex),'.mat'));
% trainedNNParam.wj = wj;
% trainedNNParam.wk = wk;
% save(strcat('OptimalNetwork',int2str(overallOpIndex)),'-struct','trainedNNParam');
%**********************************************************************

%Train a single new network
%**********************************************************************
%fprintf('TRAINING NEURAL NETWORK\n')
%trainedNNParam = trainNN(trainProcess,val,labels,initialNNParam,preProcessParam);
%**********************************************************************


%% NEURAL NETWORK EVALUATION 
fprintf('CLASSIFYING TEST DATA\n')

%Perform the classification of the test data with the network
class = evaluateNN(test,trainedNNParam,preProcessParam);

%Find the percent error and accuracy of classification
mis = find(labels-class);
mis1 = find(labels(1:N(1))-class(1:N(1)));
mis2 = find(labels(N(1)+1:N(1)+N(2))-class(N(1)+1:N(1)+N(2)));
mis3 = find(labels(N(1)+N(2)+1:totalN)-class(N(1)+N(2)+1:totalN)); 
error = length(mis)/totalN * 100;
error1 = length(mis1)/N(1) * 100;
error2 = length(mis2)/N(2) * 100;
error3 = length(mis3)/N(3) * 100;

%Display classification metrics
fprintf('Results:\n')
fprintf('\tCLASS 1:\n')
fprintf('\t\tError:     %.2f%%\n',error1)
fprintf('\t\tAccuracy:  %.2f%%\n',100-error1)
fprintf('\tCLASS 2:\n')
fprintf('\t\tError:     %.2f%%\n',error2)
fprintf('\t\tAccuracy:  %.2f%%\n',100-error2)
fprintf('\tCLASS 3:\n')
fprintf('\t\tError:     %.2f%%\n',error3)
fprintf('\t\tAccuracy:  %.2f%%\n',100-error3)
fprintf('\tOVERALL:\n')
fprintf('\t\tError:     %.2f%%\n',error)
fprintf('\t\tAccuracy:  %.2f%%\n',100-error)

%% GENERATE PLOTS

%Run the function to generate the plots for this project
genProj3Plots(train,test,val,mis,N,trainedNNParam,preProcessParam);
