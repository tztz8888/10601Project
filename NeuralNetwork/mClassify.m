function [Y]= mClassify(Model,X)

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

save('tmp.mat','feature','w1','w1_0');
return

hidden= feature * w1;
for i= 1:nTest
    hidden(i,:)=hidden(i,:)+w1_0';
end
return
output= hidden * w2;
for i= 1:nTest
    output(i,:)=output(i,:)+w2_0';
end

output(1,:)

%[val idx]= max(output)
%Y= idx'-1


%labels=labels+1;
%rst= sum(Y==labels)/nTest;