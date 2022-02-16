%FUNCTION: preProcessTrain.m
%AUTHOR: Ian McAtee
%DATE: 11/20/2021
%DESCRIPTION: Function to perform the preprocessing of training data prior
    %to its input to a neural network. Transform the data such that the 
    % mean is approximately zero and the covariance is the identity matrix
%INPUT:
    %train: A nxD matrix of training data samples
    %N: A 1xc row vector containing the number of samples in each class 
    %preProcessParameters: A structure containing the weighted mean of the
        %original training data, as well as the phi and lamda used in the
        %preprocessing whitening transfrom, used to preprocess the 
        %validation data
%OUTPUT: 
    %processedTrain: A nxD matrix of preprocessed training data samples
    %preProcessParameters: A structure containing the weighted mean of the
        %original training data, as well as the phi and lamda used in the
        %preprocessing whitening transfrom, used to preprocess the 
        %validation data

function [processedTrain,preProcessParam] = preprocessTrain(train,N)

%Get necessary variables
numFeat = size(train,2); %Number of feature dimensions
numClasses = length(N); %Number of classes
totalN = sum(N); %Total number of samples

%Preallocate a numFeatures x numClasses mean matrix
means = zeros(numFeat,numClasses);
weightMean = zeros(numFeat,1);

%Calculate the weighted mean of the training data
startIndex = 1;
endIndex = 0;
for i = 1:numClasses
    endIndex = endIndex + N(i);
    means(:,i) = mean(train(startIndex:endIndex,:))'; %Mean
    weightMean = weightMean + (N(i)/totalN)*means(:,i); %Weighted mean
    startIndex = startIndex + N(i);  
end

%Make the training data approximately zero mean 
train = train-weightMean';

%Use a whitening transformation make the training covariance = I
covTrain = cov(train);
[phi,lam] = eig(covTrain);
Y = phi*train';
processedTrain = (sqrt(lam)\ Y)';

%Set outputs preprocessing parameters
preProcessParam.weightMean = weightMean;
preProcessParam.lam = lam;
preProcessParam.phi = phi;

%Find mean and covariance of processed training data
m = mean(processedTrain);
c = cov(processedTrain);

%Error threshold for determining successful processing
thresh = 1e-10;

%Preprocessing Successful if mean and cov within thresh of 0 and I
if (sum(m)<thresh)&&((sum(eye(numFeat)-c,'all'))<thresh)
    fprintf('Training Data Preprocessing Successful\n')
%Preporcessing failed
else
    error('Training Data Preprocessing FAILED')
end

%If dimensionality 2, display the preprocessed train data mean and cov
if (numFeat==2)
    fprintf('\tPreprocessed Training Data Mean:\n')
    fprintf('\t%.2f\n',m);
    fprintf('\tPreprocessed Training Data Covariance:\n')
    fprintf('\t%.2f \t%.2f\n',c(1,1),c(1,2));
    fprintf('\t%.2f \t%.2f\n',c(2,1),c(2,2));
end


