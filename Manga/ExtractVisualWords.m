function []=ExtractVisualWords(numTrainBatch,numWord)
    descrp={};
    for i=1:numTrainBatch
        filename=sprintf('small_data_batch_%d',i);
        load(filename);
        for j=1:size(data,1) 
           descrp{(i-1)*1000+j}= GenerateDenseSiftForImg(data(j,:));
        end
        %descrp=vl_colsubset(cat(2,descrp{:}),numFeats);
        descrp=cat(2,descrp{:});
        descrp= single(descrp);
        disp(size(descrp));
        vocab=vl_kmeans(descrp, numWord, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
        save('vocab.mat','vocab');
        
    end
end



