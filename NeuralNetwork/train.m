function [Model]=train(X,Y)

%this is neuralNetwork train

feature= ExtractFeature(X); % feature is nData*nFeature matrix
nData= size(X,1);
nFeature= size(feature,2);

%define neural network
nInput= nFeature; % 50 as extracted
nHidden= 20;    % tune this value
nOutput= 10;    % 10 labels
w1= zeros(nInput,nHidden);
w2= zeros(nHidden,nOutput);
dOut= zeros(nOut);
dHidden= zeors(nHidden);
%finish defining hidden layer




save('Model.mat','Model');