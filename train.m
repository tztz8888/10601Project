function [Model]=train(X,Y)
%Model is the whole feature dataset in K-nearest-neighbor

feature= ExtractFeature(X); % feature is nData*nFeature matrix
nData= length(feature(1,:) );
nFeature= length(feature(:,1) );

Model= zeros(nData,nFeature+1);

for i= 1:nData
    Model(i,1:nFeature)= feature(i,:);
    Model(i,nFeature+1)= Y(i);
end




