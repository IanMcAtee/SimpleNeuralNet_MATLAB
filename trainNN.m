%FUNCTION: trainNN.m
%AUTHOR: Ian McAtee
%DATE: 11/29/2021
%DESCRIPTION: Function to perform the neural network training via the
    %backpropagation algorithm and validation stoppage
%INPUT:
    %train: A nxD matrix of preprocessed training data samples
    %val: A nValxD matrix of validation data samples
    %labels: a nX1 vector of class labels of the training data
    %initialNNParam: A structure containing the number of desired hidden
        %nodes, the learning rate, and max number of training epochs
    %preProcessParameters: A structure containing the weighted mean of the
        %original training data, as well as the phi and lamda used in the
        %preprocessing whitening transfrom, used to preprocess the 
        %validation data
%OUTPUT: 
    %trainedNNParam: A structure containing wj (the learned input to hidden
        %layer weights and wk (the learned hidden to output layer weights)

function trainedNNParam = trainNN(train,val,labels,initialNNParam,preProcessParam)

%Find length of training and validation data
lenTrain = length(train);
lenVal = length(val);

%Find the number of classes from the labels
numClasses = length(unique(labels));

%Find the number of samples in each class
N = zeros(1,numClasses);
for i = 1:numClasses
    N(i) = length(labels(labels==i));
end

%Form target values 
t = [];
for i = 1:numClasses
    target = -1*ones(1,numClasses);
    target(i) = 1;
    t = [t;repmat(target,N(i),1)];
end

%Extract the preprocessing parameters
weightMean = preProcessParam.weightMean;
lam = preProcessParam.lam;
phi = preProcessParam.phi;

%Preprocess val data
%Make the entire val set zero mean
val = val-weightMean';

%Whiten the varience of the validation data via a whitening transform
Y = phi*val';
val = (sqrt(lam)\ Y)';

%Form augmented training and validation data
augTrain = [ones(lenTrain,1),train];
augVal = [ones(lenVal,1),val];

%Find number of features
numFeat = size(train,2);

%Setup Neural Network Architecture
numInNodes = numFeat + 1;
numHidNodes = initialNNParam.hiddenNodes;
numOutNodes = numClasses;

%Preallocate and Initialize weight matrices
%Uncomment below for gaussian distributed weights
%wj = randn(numInNodes,numHidNodes); 
%wk = randn(numHidNodes+1,numOutNodes); 

%Initialize weights as uniform dist. between -1 and 1
a = -1; b = 1;
wj = (b-(a)).*rand(numInNodes,numHidNodes) + a;
wk = (b-(a)).*rand(numHidNodes+1,numOutNodes) + a;

%Extract the hyperparameters
numEpochs = initialNNParam.maxEpochs;
eta = initialNNParam.learnRate;

%Preallocate variables to hold training metrics
classTrain = zeros(lenTrain,1);
classVal = zeros(lenVal,1);
errorTrain = zeros(1,numEpochs);
errorVal = zeros(1,numEpochs);
cost = zeros(lenTrain,1);
costPerEpoch = zeros(numEpochs,1);

%Set the training stoppage error threshold
errorThresh = 6.0; % error of 6%

%Neural Network Training
%Loop through epochs
fprintf('Training: Epoch ');
for epoch = 1:numEpochs

    %Format the console output to keep track of epochs
    numBackSpace = length(num2str(epoch));
    backSpace = repmat('\b',1,numBackSpace);
    fprintf('%d',epoch);

    %Loop stochastically through training samples
    for n = randperm(lenTrain) %Random Sample

        %%%%%%%% FORWARD PROPAGATION %%%%%%%%
        netj = wj'*augTrain(n,:)'; %Net of hidden nodes
        yj = sigmoidF(netj); %Output of hidden nodes
        y = [1;yj]; %Add bias 

        netk = wk'*y; %Net of output nodes
        zk = sigmoidF(netk); %Output of output nodes
        
        %Sace the cost
        cost(n) = 0.5*sum((t(n)'-zk).^2);

        %%%%%%%% BACKPROPAGATION %%%%%%%%
        %Find Wkj Update Deltas
        for k = 1:numOutNodes
            deltaWk(:,k) = (eta*(t(n,k)-zk(k))*sigDeriv(netk(k)).*y)'; 
        end

        %Find Wji Update Deltas
        for j = 1:numHidNodes 
            delJ = sigDeriv(netj(j))*sum(wk(j+1,:)'.*(t(n,:)'-zk).*sigDeriv(netk));
            deltaWj(:,j) = eta*delJ.*augTrain(n,:)';
        end

        %Update weights
        wj = wj + deltaWj;
        wk = wk + deltaWk;

    end

    %Find the average cost per epoch
    costPerEpoch(epoch) = sum(cost)/lenTrain;
    
    %%%%%%%% VALIDATION %%%%%%%%
    %Use the network at current epoch to classify training data
    for n = 1:lenTrain
            netj = wj'*augTrain(n,:)'; %Net of hidden nodes
            yj = sigmoidF(netj); %Output of hidden nodes
            y = [1;yj]; %Add bias 
    
            netk = wk'*y; %Net of output nodes
            zk = sigmoidF(netk); %Output of output nodes
            
            %Save classifications
            [~,classTrain(n)] = max(zk);
    end
    
    %Use the network at current epoch to classify validation data
    for n = 1:lenVal
            netj = wj'*augVal(n,:)'; %Net of hidden nodes
            yj = sigmoidF(netj); %Output of hidden nodes
            y = [1;yj]; %Add bias 
    
            netk = wk'*y; %Net of output nodes
            zk = sigmoidF(netk); %Output of output nodes
            
            %Save classifications
            [~,classVal(n)] = max(zk);
    end

    %Find the training and validation classification errors
    errorTrain(epoch) = length(find(labels-classTrain))/lenTrain * 100;
    errorVal(epoch) = length(find(labels-classVal))/lenVal * 100;
    
    %End training if both train and val errors are below error threshold
    if((errorVal(epoch)<errorThresh)&&(errorTrain(epoch)<errorThresh))
        epochStop = epoch;
        fprintf('\nTraining Successful \n')
        break
    end

    %Save the epoch that one stops at
    epochStop = numEpochs;
    fprintf(backSpace);
end

%Return the learned weights
trainedNNParam.wj = wj;
trainedNNParam.wk = wk;

%Plots

%Plot Cost per Epoch
figure()
plot([1:epochStop],costPerEpoch(1:epochStop)')
title('Average Cost per Epoch')
xlim([1,epochStop])
xlabel('Epoch')
ylabel('Average Cost')

%Plot Classification Error per Epoch
figure()
plot([1:epochStop],errorTrain(1:epochStop))
hold on
plot([1:epochStop],errorVal(1:epochStop))
hold off
title('Total Classification Error per Epoch')
xlim([1,epochStop])
xlabel('Epoch')
ylabel('Error %')
legend('Training Error','Validation Error','location','northeast')

end