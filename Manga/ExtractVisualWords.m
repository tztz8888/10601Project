function []=ExtractVisualWords(numWord)
    descrp={};
        filename='small_data_batch_1';
        load(filename);
        for j=1:size(data,1)
           descrp{j}= GenerateDenseSiftForImg(data(j,:));
        end
        %descrp=vl_colsubset(cat(2,descrp{:}),numFeats);
        descrp=cat(2,descrp{:});
        descrp= single(descrp);
        disp(size(descrp));
        vocab=vl_kmeans(descrp, numWord, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
        save('vocab.mat','vocab');
end



