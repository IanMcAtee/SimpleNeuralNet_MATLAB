%FUNCTION: genProj3Plots.m
%AUTHOR: Ian McAtee
%DATE: 11/31/2021
%DESCRIPTION: Function to plot various plots corresponding to EE5650
    %project 3
%INPUT:
    %train: A nxD matrix of  training data samples
    %train: A nTestxD matrix of  test data samples
    %val: A nValxD matrix of validation data samples
    %mis: A mx1 vector containing the indices of the misclassified points
    %N: A 1xC row vector of the number of samples in each class
    %trainedNNParam: A structure containing wj (the learned input to hidden
        %layer weights and wk (the learned hidden to output layer weights)
    %preProcessParameters: A structure containing the weighted mean of the
        %original training data, as well as the phi and lamda used in the
        %preprocessing whitening transfrom, used to preprocess the 
        %validation data
%OUTPUT: 
    %Scatter plots of the training, test, and validation data
    %Scatter plots of the preprocessed training, test, and val data
    %Scatter plots of the neural network decision regions superimposed with
        %the preprocessed training, test, validation, and misclassified 
        %points

function genProj3Plots(train,test,val,mis,N,trainedNNParam,preProcessParam)

%Find indices for easier plotting
N1s = 1;
N1e = N(1);
N2s = N(1)+1;
N2e = N(1)+N(2);
N3s = N(1)+N(2)+1;
N3e = sum(N);
totalN = sum(N);

%Extract the trained weights
wj = trainedNNParam.wj;
wk = trainedNNParam.wk;

%Extract preprocessing parameters
weightMean = preProcessParam.weightMean;
lam = preProcessParam.lam;
phi = preProcessParam.phi;

%Preprocess train, test, and validation data
%Make the entire test set zero mean
trainP = train-weightMean';
testP = test-weightMean';
valP = val-weightMean';

%Whiten the varience of the test data via a whitening transform
Ytrain = phi*trainP';
Ytest = phi*testP';
Yval = phi*valP';
trainP = (sqrt(lam)\ Ytrain)';
testP = (sqrt(lam)\ Ytest)';
valP = (sqrt(lam)\ Yval)';

%Set some axis options
axisOpt = [-20,20,-15,25];
axisOpt2 = [-4,4,-4,4];

%% Plot Data

%Training Data
figure()
scatter(train(N1s:N1e,1),train(N1s:N1e,2),'.b')
hold on
scatter(train(N2s:N2e,1),train(N2s:N2e,2),'.r')
hold on
scatter(train(N3s:N3e,1),train(N3s:N3e,2),'.g')
axis(axisOpt)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Training Data')
legend('Class 1','Class 2','Class 3','Location','southwest')

%Validation Data
figure()
scatter(val(N1s:N1e,1),val(N1s:N1e,2),'.b')
hold on
scatter(val(N2s:N2e,1),val(N2s:N2e,2),'.r')
hold on
scatter(val(N3s:N3e,1),val(N3s:N3e,2),'.g')
axis(axisOpt)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Validation Data')
legend('Class 1','Class 2','Class 3','Location','southwest')

%Test Data
figure()
scatter(test(N1s:N1e,1),test(N1s:N1e,2),'.b')
hold on
scatter(test(N2s:N2e,1),test(N2s:N2e,2),'.r')
hold on
scatter(test(N3s:N3e,1),test(N3s:N3e,2),'.g')
axis(axisOpt)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Test Data')
legend('Class 1','Class 2','Class 3','Location','southwest')

%% Plot Preprocessed Data

%Prepocessed Training Data
figure()
scatter(trainP(N1s:N1e,1),trainP(N1s:N1e,2),'.b')
hold on
scatter(trainP(N2s:N2e,1),trainP(N2s:N2e,2),'.r')
hold on
scatter(trainP(N3s:N3e,1),trainP(N3s:N3e,2),'.g')
axis(axisOpt2)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Preprocessed Training Data')
legend('Class 1','Class 2','Class 3','Location','southeast')

%Prepocessed Validation Data
figure()
scatter(valP(N1s:N1e,1),valP(N1s:N1e,2),'.b')
hold on
scatter(valP(N2s:N2e,1),valP(N2s:N2e,2),'.r')
hold on
scatter(valP(N3s:N3e,1),valP(N3s:N3e,2),'.g')
axis(axisOpt2)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Preprocessed Validation Data')
legend('Class 1','Class 2','Class 3','Location','southeast')

%Prepocessed test Data
figure()
scatter(testP(N1s:N1e,1),testP(N1s:N1e,2),'.b')
hold on
scatter(testP(N2s:N2e,1),testP(N2s:N2e,2),'.r')
hold on
scatter(testP(N3s:N3e,1),testP(N3s:N3e,2),'.g')
axis(axisOpt2)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Preprocessed Test Data')
legend('Class 1','Class 2','Class 3','Location','southeast')

%% Plot Decision Regions

%Create temp x data to evaluate over
x = (-4:0.03:4); %Reduce the step size to for higher res. plots
L = length(x); %Get number of temp samples

temp = [];
%Form temp data
for i=1:L  % loop columns
    for j=1:L  % loop rows
        temp = [temp;x(i),x(j)]; %Form vector
    end
end

%Find length of temp data
lenTemp = length(temp);

%Form Augmented Temp data
augTemp = [ones(lenTemp,1),temp];

%Preallocate temp classification vector
classTemp = zeros(lenTemp,1);

%Forward propagate the temp data through trained network
for n = 1:lenTemp
        %FORWARD PROPAGATION
        netj = wj'*augTemp(n,:)'; 
        yj = sigmoidF(netj); 
        y = [1;yj]; 

        netk = wk'*y; 
        zk = sigmoidF(netk); 
        
        %Perform classifcation by taking max of output
        [~,classTemp(n)] = max(zk);
end

%Form a matrix of classification regions
classRegions = zeros(L,L);
index1 = find(classTemp == 1);
index2 = find(classTemp == 2);
index3 = find(classTemp == 3);
classRegions(index1) = 1;
classRegions(index2) = 2;
classRegions(index3) = 3;

%Plot the decison boundaries with training data
figure()
imagesc(x,x,classRegions,'AlphaData',1.0)
axis ('xy')
colormap('bone')
hold on
scatter(trainP(N1s:N1e,1),trainP(N1s:N1e,2),'.b')
hold on
scatter(trainP(N2s:N2e,1),trainP(N2s:N2e,2),'.r')
hold on
scatter(trainP(N3s:N3e,1),trainP(N3s:N3e,2),'.g')
hold off
axis(axisOpt2)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Neural Network Decision Regions')
subtitle('Preprocessed Training Data Superimposed')
legend('Class 1','Class 2','Class 3','Location','northwest')
text(0.5,3,'Region 1','Color','b')
text(-3.5,-2,'Region 3','Color','g')
text(1,-3,'Region 2','Color','r')

%Plot the decison boundaries with test data
figure()
imagesc(x,x,classRegions,'AlphaData',1.0)
axis ('xy')
colormap('bone')
hold on
scatter(testP(N1s:N1e,1),testP(N1s:N1e,2),'.b')
hold on
scatter(testP(N2s:N2e,1),testP(N2s:N2e,2),'.r')
hold on
scatter(testP(N3s:N3e,1),testP(N3s:N3e,2),'.g')
hold off
axis(axisOpt2)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Neural Network Decision Regions')
subtitle('Preprocessed Test Data Superimposed')
legend('Class 1','Class 2','Class 3','Location','northwest')
text(0.5,3,'Region 1','Color','b')
text(-3.5,-2,'Region 3','Color','g')
text(1,-3,'Region 2','Color','r')

mis1 = mis(mis<=N1e);
mis2 = mis(mis>N1e & mis<=N2e);
mis3 = mis(mis>N2e & mis<=N3e);

%Plot the decison boundaries with misclassified
figure()
imagesc(x,x,classRegions,'AlphaData',1.0)
axis ('xy')
colormap('bone')
hold on
scatter(testP(mis1,1),testP(mis1,2),'.b')
hold on
scatter(testP(mis2,1),testP(mis2,2),'.r')
hold on
scatter(testP(mis3,1),testP(mis3,2),'.g')
hold off
axis(axisOpt2)
axis ('square')
box on
xlabel('x_1')
ylabel('x_2')
title('Neural Network Decision Regions')
subtitle('Misclassified Preprocessed Test Data Superimposed')
legend('Class 1','Class 2','Class 3','Location','northwest')
text(0.5,3,'Region 1','Color','b')
text(-3.5,-2,'Region 3','Color','g')
text(1,-3,'Region 2','Color','r')


end