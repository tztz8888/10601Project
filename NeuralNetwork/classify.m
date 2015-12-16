function [rst]= classify()

%neural network
load('newModel.mat');
load('small_data_batch_5.mat');
%load('Model.mat');
feature= ExtractFeatureForImg(data,vocab);
%feature= ExtractFeatureForImg(X,vocab);

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
    Y(i)= idx;
end

labels=labels+1;
rst= sum(Y==labels)/nTest;