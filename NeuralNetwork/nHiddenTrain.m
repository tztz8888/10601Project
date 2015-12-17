function [acclist]= nHiddenTrain()

acclist= zeros(10,1);

for i=1:10
    [Model]=train(i*50);
    load('small_data_batch_5.mat');
    [Y]= classify(Model,data);
    acclist(i)= sum(Y==labels)/size(data,1);
end

x=zeros(10,1);
for i=1:10
    x(i)= i*50;
end
plot(x,acclist)
xlabel('number of hidden layer nodes')
ylabel('accuracy rate')