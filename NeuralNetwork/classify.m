function [Y]= classify(Model,X)

%neural network
%load('Model.mat');
%feature= ExtractFeatureForImg(data,vocab);
%feature= ExtractFeatureForImg(X,vocab);
w1= Model.w1;
w2= Model.w2;
w1_0= Model.w1_0;
w2_0= Model.w2_0;

feature=GenerateHOGForImg( X );

nTest= length(feature(:,1) );
nFeature= length(feature(1,:) );

nInput= nFeature;
nHidden= size(w1,2);
nOutput= size(w2,2);
hidden= zeros(nTest,nHidden);
output= zeros(nTest,nOutput);
Y= zeros(nTest,1);

e= ones(nTest,nHidden)*exp(1);
o= ones(nTest,nHidden);
w1_0_m= repmat(w1_0',nTest,1);
hidden= feature * w1+w1_0_m;
hidden=o./(1+e.^(-hidden));

e2= ones(nTest,nOutput)*exp(1);
o2= ones(nTest,nOutput);
w2_0_m= repmat(w2_0',nTest,1);
output= hidden * w2 + w2_0_m;
output= o2./(1+e2.^(-output) );

[val idx]= max(output');
Y= idx'-1;

%labels=labels+1;
%rst= sum(Y==labels)/1000;