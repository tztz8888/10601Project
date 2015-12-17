function [Model]=htrain()
%this is neuralNetwork train

load('Model.mat');
nTrain= size(TrainFeatures,1);
nFeature= size(TrainFeatures,2);
Y= TrainLabel+1;
feature= TrainFeatures;

%define neural network
nInput= nFeature; % 496 as extracted
nHidden= 200;    % tune this value
nOutput= 10;    % 10 labels
w1= rand(nInput,nHidden)/10;
w1_0= rand(nHidden,1)/10; %constant
w2= rand(nHidden,nOutput)/10;
w2_0= rand(nOutput,1)/10; %constant

dOutput= zeros(1,nOutput);
dHidden= zeros(1,nHidden);
hidden= zeros(1,nHidden);
output= zeros(1,nOutput);
%finish defining hidden layer

nIter=100;
u= 0.1;
tOutput= zeros(nTrain,nOutput);
for i=1:nTrain
    tOutput(i,Y(i))= 1;
end

e= ones(1,nHidden)*exp(1);
o= ones(1,nHidden);
e2= ones(1,nOutput)*exp(1);
o2= ones(1,nOutput);
o3= ones(1,nOutput);
o4= ones(1,nHidden);
for k=1:nIter
    disp 'start one inter'
    labels= zeros(nTrain,1);
    
    for i=1:nTrain      
       hidden= feature(i,:)* w1+ w1_0';
       hidden=o./(1+e.^(-hidden));
       output= hidden*w2+w2_0';
       output= o2./(1+e2.^(-output) );
       
       [val idx] = max(output);
       labels(i)=idx;
       
       dOutput= output.*(o3-output).*( tOutput(i,:)- output);
       tmp= dOutput*w2';
       dHidden= hidden.*(o4-hidden).*tmp;
       
        d= feature(i,:)'* dHidden; %update
        w1=w1+u*d;
        d2= hidden'*dOutput;
        w2=w2+u*d2;
        w1_0= w1_0+ u*dHidden';
        w2_0= w2_0+ u*dOutput';
    end
    rst= sum(Y==labels)/nTrain
    
end

Model= struct('w1',w1,'w2',w2,'w1_0',w1_0,'w2_0',w2_0);
save('hModel.mat','Model');
% load('small_data_batch_5.mat');
% [Y]= classify(Model,data);
% acc= sum(Y==labels)/size(data,1);