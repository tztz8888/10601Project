function [w,b] = train( fileName)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    load(fileName);
    c=1e-2;
    w=zeros(10,496);
    bias=zeros(10,1);
    for i=1:10
        x=Model.TrainFeatures;
        t=zeros(size(Model.TrainLabel));
        t(Model.TrainLabel==(i-1))=1;
        t(Model.TrainLabel~=(i-1))=-1;
        N=size(x,1);
        K=x*x';
        H=(t*t').*K + 1e-5*eye(N);
        f = repmat(1,N,1);
        A = [];b = [];
        LB = repmat(0,N,1); UB = repmat(inf,N,1);
        UB = repmat(c,N,1);
        Aeq = t';beq = 0;
        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB);
        w(i,:)=sum(repmat(alpha.*t,1,496).*x,1);
        fout = sum(repmat(alpha.*t,1,N).*K,1)';
        pos = find(alpha>c*1e-6);
        bias(i) = mean(t(pos)-fout(pos));
    end
    Model1=struct('W',w,'b',bias)
    save('Model_linear_1e-2.mat','Model1');

end

