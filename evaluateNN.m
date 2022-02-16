%FUNCTION: evaluateNN.m
%AUTHOR: Ian McAtee
%DATE: 11/29/2021
%DESCRIPTION: Function to perform the neural network evaluation
    % (classification) of test data
%INPUT:
    %test: A nxD matirx of test data samples
    %trainedNNParam: A structure containing wj (the learned input to hidden
        %layer weights and wk (the learned hidden to output layer weights) 
    %preProcessParameters: A structure containing the weighted mean of the
        %original training data, as well as the phi and lamda used in the
        %preprocessing whitening transfrom, used to preprocess the 
        %validation data
%OUTPUT: 
    %class: A nx1 vector of classifications determined by the network for
        %the test data

function class = evaluateNN(test,trainedNNParam,preProcessParam)

%Find length of test data
lenTest = length(test);

%Extract the trained weights
wj = trainedNNParam.wj;
wk = trainedNNParam.wk;

%Extract the preprocessing parameters
weightMean = preProcessParam.weightMean;
lam = preProcessParam.lam;
phi = preProcessParam.phi;

%Preprocess test data
%Make the entire test set zero mean
test = test-weightMean';

%Whiten the varience of the test data via a whitening transform
Y = phi*test';
test = (sqrt(lam)\ Y)';
%Uncomment below to see that mean ~= 0 and cov ~= I
% mean(test)
% cov(test)

%Form the augmented test data
augTest = [ones(lenTest,1),test];

%Preallocate classification vector
class = zeros(lenTest,1);

%Forward propagate the test data through trained network
for n = 1:lenTest
        %FORWARD PROPAGATION
        netj = wj'*augTest(n,:)'; 
        yj = sigmoidF(netj); 
        y = [1;yj]; 

        netk = wk'*y; 
        zk = sigmoidF(netk); 
        
        %Perform classifcation by taking max of output
        [~,class(n)] = max(zk);
end

fprintf('Classification Successful\n')
end