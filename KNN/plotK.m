function [acc]= plotK(Model,X,labels)


acc= zeros(30,1);

for k=1:30
    Y = classify( Model,X,k );
    acc(k)= sum(Y==labels)/size(X,1);
end

plot((1:30),acc);
xlabel('K value');
ylabel('accuracy');
    