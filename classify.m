function [Y]= classify(Model,X)
%k nearest neighbor
K=3;
nLabel=10;
feature= ExtractFeature(X);
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
    flagKNN= zeros(K,1);
    distKNN=zeros(K,1);
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
        count(flagKNN(j))= count(flagKNN(j))+1;
        if  count(flagKNN(j))>maxCount
            maxCount= count(flagKNN(j));
            labelMaxCount= flagKNN(j);
        end
    end
    
    Y(i)= labelMaxCount;
end