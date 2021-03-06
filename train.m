function [Model]=train(X,Y)
%Model is the whole feature dataset in K-nearest-neighbor

load('Manga/TrainBagOfSift.mat');
%feature= ExtractFeature(X); % feature is nData*nFeature matrix
nData= 5000;
nFeature= 50;

Model= zeros(nData,nFeature+1);

for i= 1:nData
    Model(i,1:nFeature)= TrainFeatures(i,:);
    Model(i,nFeature+1)= TrainLabel(i);
end

save('Model.mat','Model');


