function [] = GetTrainImgFeat(numTrainBatch)
    load('vocab.mat')
    vocab_size = size(vocab, 2);
    TrainLabel=zeros(numTrainBatch*1000,1);
    TrainFeatures=zeros(numTrainBatch*1000,vocab_size);
    for i=1:numTrainBatch
        filename=sprintf('small_data_batch_%d',i);
        load(filename);
        for j=1:size(data,1)
%             tempdata=data(j,:);
%             newImage=zeros(32,32,3,'uint8');
%             newImage(:,:,1)=reshape(tempdata(1:1024),32,32)';
%             newImage(:,:,2)=reshape(tempdata(1025:2048),32,32)';
%             newImage(:,:,3)=reshape(tempdata(2049:3072),32,32)';
%             [drop,descrp]= vl_dsift(im2single(rgb2gray(newImage)));
            descrp=GenerateDenseSiftForImg(data(j,:));
            [drop, binsa] = min(vl_alldist(vocab, single(descrp)), [], 1) ; 
            hist=zeros(1,vocab_size);
            for k=1:vocab_size
                hist(k)=sum(binsa==k);
            end
            TrainLabel((i-1)*1000+j)=labels(j);
            TrainFeatures((i-1)*1000+j,:)=hist/sum(hist);
        end
        clear data;
        clear labels;       
    end
    save('Model','TrainLabel','TrainFeatures','vocab');

end