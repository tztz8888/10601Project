function [ Y ] = classify1( Model1,X )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

    w=Model1.W;
    b=Model1.b;
    feature= GenerateHOGForImg(X);
    Y=zeros(size(feature,1),1);
    for i=1:size(feature,1)
        score=zeros(10,1);
        for j=1:10
            score(j)=feature(i,:)*w(j,:)'+b(j);           
        end
        [M,I]=max(score);
        Y(i)=I-1;
    end
end

