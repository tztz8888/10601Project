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
hidden= zeros(nHidden,1);
output= zeros(nOutput,1);
Y= zeros(nTest,1);

for i=1:nTest % all test cases
    for j=1:nHidden %forward to hidden
        tmp=w1_0(j); %compute sigmoid
        for l=1:nInput
            tmp= tmp+ w1(l,j)*feature(i,l);
        end
        hidden(j)= 1/(1+exp(-tmp));
    end
    
    for j=1:nOutput %compute forward from hidden to output
         tmp=w2_0(j);
         for l=1:nHidden
           tmp= tmp+ w2(l,j)*hidden(l);
         end
         output(j)= 1/(1+exp(-tmp));
    end
    
    [val idx] = max(output);
    Y(i)= idx-1;
end

%labels=labels+1;
%rst= sum(Y==labels)/nTest;