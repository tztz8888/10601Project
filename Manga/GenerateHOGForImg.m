function [ feat_vec] = GenerateHOGForImg( data )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    num_data = size(data,1);
    feat_vec= zeros(num_data,496);
    for i=1:num_data
        newImage=zeros(32,32,3,'uint8');
        newImage(:,:,1)=reshape(data(1:1024),32,32)';
        newImage(:,:,2)=reshape(data(1025:2048),32,32)';
        newImage(:,:,3)=reshape(data(2049:3072),32,32)';
        feat_vec(i,:)=extract_HOGfeature(newImage);
    end
end

