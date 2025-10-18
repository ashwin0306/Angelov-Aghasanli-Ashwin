function [sil, db] = clustering_quality(Y, labels)
    Y = double(Y);
    labels = labels + 1;
    labels = round(labels);
    labels(labels < 1) = 1;
    
    sil = mean(silhouette(Y, labels));
    db = evalclusters(Y, labels, 'DaviesBouldin').CriterionValues;
end 