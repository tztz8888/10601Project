function [w,b] = train( Model)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    margin=1;
    w=zeros(10,496);
    b=zeros(10,1);
    for i=1:10
        x=Model.TrainFeatures;
        t=Model.TrainLabel;
        t(t~=1)=-1;
        N=size(x,1);
        K=x*x';
        H=(t*t').*K + 1e-5*eye(N);
        f = repmat(1,N,1);
        A = [];b = [];
        LB = repmat(0,N,1); UB = repmat(inf,N,1);
        UB = repmat(margin,N,1);
        Aeq = t';beq = 0;
        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB);
        w(i)=sum(repmat(alpha.*t,1,496).*x,1);
        fout = sum(repmat(alpha.*t,1,N).*K,1)';
        pos = find(alpha>1e-6);
        b(i) = mean(t(pos)-fout(pos));
    end

end

