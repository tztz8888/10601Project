function [ featureVecs ] = ExtractFeatureForImg( data ,vocab)
% Extract a 50-dimension feature for each img
% input:
%       data - a N*3072 img data matrix
%       vocab - a 128*50 cluster
% output:
%       featureVecs - a 1*50 feature vec
    vocab_size = size(vocab, 2);
    num_data = size(data,1);
    featureVecs=zeros(num_data,vocab_size);
    for i=1:num_data
        descrp=GenerateDenseSiftForImg(data(i,:));
        [drop, binsa] = min(vl_alldist(vocab, single(descrp)), [], 1) ; 
        for k=1:vocab_size
            featureVecs(i,k)=sum(binsa==k);
        end
        featureVecs(i,:)=featureVecs(i,:)/sum(featureVecs(i,:));
    end
end

