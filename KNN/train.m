function [Model]=train()
%Model is the whole feature dataset in K-nearest-neighbor

load('Model.mat');
%feature= ExtractFeature(X); % feature is nData*nFeature matrix
nData= size(TrainFeatures,1);
nFeature= size(TrainFeatures,2);

Model= zeros(nData,nFeature+1);

for i= 1:nData
    Model(i,1:nFeature)= TrainFeatures(i,:);
    Model(i,nFeature+1)= TrainLabel(i);
end

save('newModel.mat','Model','vocab');


