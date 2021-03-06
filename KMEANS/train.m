function [Model]=train()

load('Model.mat');
%feature= ExtractFeature(X); % feature is nData*nFeature matrix
feature= TrainFeatures;
nData= size(TrainFeatures,1);
nFeature= size(TrainFeatures,2);
% feature=[1 2 3;
%         2 3 4;
%         3 2 1;
%         ];
% nData= 3;
% nFeature=3;
% K=2;

K=10;   %K clusters
centroids= rand(K,nFeature);  %centers for kmeans
cluster= zeros(nData,1);   %each data belongs to which cluser(1:K)
clusterLabel= zeros(K,1);

while true
    numInCluster= zeros(K,1);   %hom many datas in each cluster
    
    for i=1:nData  %assign
        l2dist= zeros(K,1);
        for j=1:K
            l2dist(j)= L2Distance( feature(i,:), centroids(j,:) );
        end
        [val idx]= min(l2dist);
        cluster(i)= idx;
        numInCluster(idx)= numInCluster(idx)+1;
    end
    
    tmpSum= zeros(K,nFeature);
    for i=1:nData
        tmpSum(cluster(i),:)= tmpSum(cluster(i),:)+ feature(i,:);
    end
    for i=1:K
        if numInCluster(i)~=0
            tmpSum(i,:)= tmpSum(i,:)/numInCluster(i);
        else
            tmpSum(i,:)= centroids(i,:);
        end
    end
    
    if isequal(tmpSum,centroids)
        disp 'converged'
        break;
    end
    
    centroids= tmpSum;
    disp 'aaa'
    
end
TrainLabel

nLabelInCluster= zeros(K,10);
for i=1:nData
    nLabelInCluster(cluster(i),TrainLabel(i))=nLabelInCluster(cluster(i),TrainLabel(i))+1;
end
for k=1:K
    [val idx]= max( nLabelInCluster(k,:) );
    clusterLabel(k)= idx;
end
centroids
clusterLabel
save('newModel.mat','centroids','clusterLabel','cluster');




