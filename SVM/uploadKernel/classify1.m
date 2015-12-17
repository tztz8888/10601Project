function [ Y ] = classify1( Model1,X )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here


    feature= GenerateHOGForImg(X);
    numTest=size(feature,1);
    score=zeros(numTest,10);
    
    for i=1:10
        
        a=repmat(Model1(i).supportcoeff',numTest,1);% numTest*numSupport
        bias=repmat(Model1(i).b,numTest,1);% numTest*1
        K=(feature*(Model1(i).supportvec)'+1).^2; % k(numTest,numSupport)
        temp=a.*K;
        
        score(:,i)=sum(temp,2)+bias;
    end
    
    [M,I]=max(score');
    Y=I'-1;
end

