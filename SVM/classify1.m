function [ Y ] = classify1( Model,X )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

    feature= GenerateHOGForImg(X);
    Y=feature*(Model.W)'+Model.b;
    Y(Y>=0)=1;
    Y(Y<0)=-1;
end

