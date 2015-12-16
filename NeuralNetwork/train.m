function [Model]=train()
%this is neuralNetwork train

load('Model.mat');
%feature= ExtractFeature(X); % feature is nData*nFeature matrix
%nData= size(X,1);
%nFeature= size(feature,2);
nData= size(TrainFeatures,1);
nFeature= size(TrainFeatures,2);
Y= TrainLabel+1;
feature= TrainFeatures;

%define neural network
nInput= nFeature; % 496 as extracted
nHidden= 200;    % tune this value
nOutput= 10;    % 10 labels
w1= rand(nInput,nHidden)/10;
w1_0= rand(nHidden,1)/10; %constant
w2= rand(nHidden,nOutput)/10;
w2_0= rand(nOutput,1)/10; %constant
dOutput= zeros(nOutput,1);
dHidden= zeros(nHidden,1);
input= zeros(nInput,1);
hidden= zeros(nHidden,1);
output= zeros(nOutput,1);
%finish defining hidden layer

nIter=200;
u= 0.1;

%save('tmp.mat','feature','w1','w1_0','w2_0','w2');
%return

for k=1:nIter
    labels= zeros(nData,1);
    disp 'start one inter'
    for i=1:nData
        
       for j=1:nHidden %compute forward from input to hidden
           tmp=w1_0(j); %compute sigmoid
           for l=1:nInput
               tmp= tmp+ w1(l,j)*feature(i,l);
           end
           hidden(j)= 1.0/(1.0+exp(-tmp));
       end
       
       for j=1:nOutput %compute forward from hidden to output
           tmp=w2_0(j);
           for l=1:nHidden
               tmp= tmp+ w2(l,j)*hidden(l);
           end
           output(j)= 1/(1+exp(-tmp));
       end
       
     %  save('tmp.mat','output','hidden');
     %  return
  
       [val idx] = max(output);
       labels(i)=idx;
       
       t_output= zeros(nOutput,1);
       t_output(Y(i))= 1;  %only this label is 1; already start label from 1
       for j=1:nOutput %compute delta for output
           dOutput(j)= output(j)*(1-output(j))*(t_output(j)-output(j) );
       end
       
       for j=1:nHidden %compute delta for hidden
           tmp=0;
           for l=1:nOutput
               tmp= tmp+ w2(j,l)*dOutput(l);
           end
           dHidden(j)= hidden(j)*(1-hidden(j))* tmp;
       end
       
       %save('tmp.mat','dHidden','dOutput');
       %return
       
       for j=1:nInput %update w1
           for l= 1:nHidden
                d= u* dHidden(l)* feature(i,j);
                w1(j,l)= w1(j,l)+d;
           end
       end
       
       for j=1:nHidden
           d= u* dHidden(j);
           w1_0(j)=w1_0(j)+d;
       end
       
       for j=1:nHidden %updata w2
           for l= 1:nOutput
               d= u* dOutput(l)* hidden(j);
               w2(j,l)=w2(j,l)+d;
           end
       end
       
       for j=1:nOutput
           d= u* dOutput(j);
           w2_0(j)=w2_0(j)+d;
       end
       
    end
    
   rst= sum(Y==labels)/nData
end

Model= struct('w1',w1,'w2',w2,'w1_0',w1_0,'w2_0',w2_0);
save('newModel.mat','Model');