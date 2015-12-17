function [Model]=KCrossValidationTrain()
%this is neuralNetwork train

feature= zeros(5000,496);
label= zeros(5000,1);

load('small_data_batch_1.mat');
feature(1:1000,:)= GenerateHOGForImg( data );
label(1:1000)= labels;

load('small_data_batch_2.mat');
feature(1001:2000,:)= GenerateHOGForImg( data );
label(1001:2000)= labels;

load('small_data_batch_3.mat');
feature(2001:3000,:)= GenerateHOGForImg( data );
label(2001:3000)= labels;

load('small_data_batch_4.mat');
feature(3001:4000,:)= GenerateHOGForImg( data );
label(3001:4000)= labels;

load('small_data_batch_5.mat');
feature(4001:5000,:)= GenerateHOGForImg( data );
label(4001:5000)= labels;

nFeature= size(feature,2);

%define neural network
nInput= nFeature; % 496 as extracted
nHidden= 20;    % tune this value
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

nIter=1;   %tune this value to find best N
u= 0.1;
maxAcc=0;
maxw1=zeros(nInput,nHidden);
maxw2=zeros(nHidden,nOutput);
maxw1_0=zeros(nHidden,1);
maxw2_0=zeros(nOutput,1);
acc= zeros(5,1);

for k=1:5
    TrainFeature=zeros(4000,nFeature);
    TrainLabel=zeros(4000,1);
    nData=4000;
    count=0;
    for i=1:5
        if i~=k
            TrainFeature(count*1000+1:(count+1)*1000,:)= feature((i-1)*1000+1:i*1000,:);
            TrainLabel(count*1000+1:(count+1)*1000,:)= label((i-1)*1000+1:i*1000,:) +1;
            count=count+1;
        end
    end

    disp 'start one fold'
    for i=1:nData

       for j=1:nHidden %compute forward from input to hidden
           tmp=w1_0(j); %compute sigmoid
           for l=1:nInput
               tmp= tmp+ w1(l,j)*TrainFeature(i,l);
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

       t_output= zeros(nOutput,1);
       t_output(TrainLabel(i))= 1;  %only this label is 1; already start label from 1
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

       for j=1:nInput %update w1
           for l= 1:nHidden
                d= u* dHidden(l)* TrainFeature(i,j);
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

    end  %finish one fold

    disp 'finish one fold'

    TestLabels= label(1000*(k-1)+1:1000*k);
    TestFeature= feature(1000*(k-1)+1:1000*k,:);
    nTest=1000;
    Y=zeros(1000,1);
    for i=1:nTest % all test cases
        for j=1:nHidden %forward to hidden
            tmp=w1_0(j); %compute sigmoid
            for l=1:nInput
                tmp= tmp+ w1(l,j)*TestFeature(i,l);
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
    acc(k)= sum(Y==TestLabels)/1000

end
    
    meanAcc= mean(acc)
    if meanAcc> maxAcc
        maxAcc= meanAcc;
        maxw1= w1;
        maxw2= w2;
        maxw1_0= w1_0;
        maxw2_0= w2_0;
    end
    
end

Model= struct('w1',w1,'w2',w2,'w1_0',w1_0,'w2_0',w2_0);
save('newModel.mat','Model');
Model= struct('w1',maxw1,'w2',maxw2,'w1_0',maxw1_0,'w2_0',maxw2_0);
save('FVModel.mat','Model');