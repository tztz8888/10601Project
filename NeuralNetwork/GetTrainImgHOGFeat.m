function [TrainFeatures] = GetTrainImgHOGFeat(numTrainBatch)
    load('vocab.mat')
    TrainLabel=zeros(numTrainBatch*1000,1);
    TrainFeatures=zeros(numTrainBatch*1000,496);
    for i=1:numTrainBatch
        filename=sprintf('small_data_batch_%d',i);
        load(filename);
        for j=1:size(data,1)
            tempdata=data(j,:);
            newImage=zeros(32,32,3,'uint8');
            newImage(:,:,1)=reshape(tempdata(1:1024),32,32)';
            newImage(:,:,2)=reshape(tempdata(1025:2048),32,32)';
            newImage(:,:,3)=reshape(tempdata(2049:3072),32,32)';
            feat=extract_HOGfeature(newImage);
            TrainLabel((i-1)*1000+j)=labels(j);
            TrainFeatures((i-1)*1000+j,:)=feat;
        end
        clear data;
        clear labels;       
    end
    save('Model','TrainLabel','TrainFeatures');

end
