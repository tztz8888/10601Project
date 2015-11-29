function [Y]= classify()
%k nearest neighbor
K=20;    %tune to find the best
nLabel=10; % given

load('newModel.mat');
load('small_data_batch_5.mat');

feature= ExtractFeatureForImg(data,vocab);

nTest= length(feature(:,1) );
nFeature= length(feature(1,:) );
nTrain= length(Model(:,1));

distMetrics= zeros(nTest,nTrain);
for i=1:nTest
    for j=1:nTrain
         distMetrics(i,j)= L2Distance( feature(i,:),Model(j,1:nFeature) );
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
                flagKNN(k)= Model(j,nFeature+1);
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