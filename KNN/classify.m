function [ Y ] = classify( Model,X )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

K=3;    %tune to find the best
nLabel=10; % given

%load('newModel.mat');
TrainFeatures=Model.TrainFeatures;
TrainLabel=Model.TrainLabel;
%load('Model.mat');
feature= GenerateHOGForImg(X);


nTest= size(feature,1);
nFeature= size(feature,2);
nTrain= size(TrainFeatures,1);
distMetrics= zeros(nTest,nTrain);
Y= zeros(nTest,1);

for i=1:nTest
    for j=1:nTrain
         distMetrics(i,j)= norm( feature(i,:)-TrainFeatures(j,:) );
    end
end


for i=1:nTest
    flagKNN= zeros(K,1); % list of the first K nearest neibor Label
    distKNN=zeros(K,1); %list of the first K nearest neibor distance
    for j=1:K
       distKNN(j)=2^8-1;
    end
    for j=1:nTrain
        for k=1:K
            if distMetrics(i,j)<distKNN(k)
                for m=K:-1:k+1
                    distKNN(m)=distKNN(m-1);
                end
                distKNN(k)= distMetrics(i,j);
                flagKNN(k)= TrainLabel(j);
                break;
            end
        end
    end
   
    maxCount=0;
    labelMaxCount=0;
    count= zeros(nLabel,1);
    for j=1:K
        count(flagKNN(j)+1)= count(flagKNN(j)+1)+1;
        if  count(flagKNN(j)+1)>maxCount
            maxCount= count(flagKNN(j)+1);
            labelMaxCount= flagKNN(j);
        end
    end
    
    Y(i)= labelMaxCount;
end
end
