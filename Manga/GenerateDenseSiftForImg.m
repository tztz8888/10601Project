% input: data, a 1*3072 vector
% output: dense sift descriptor ,128*(num of sift)

function [descrp] = GenerateDenseSiftForImg(data)
    newImage=zeros(32,32,3,'uint8');
    newImage(:,:,1)=reshape(data(1:1024),32,32)';
    newImage(:,:,2)=reshape(data(1025:2048),32,32)';
    newImage(:,:,3)=reshape(data(2049:3072),32,32)';
    [drp,descrp]= vl_dsift(im2single(rgb2gray(newImage)));
end