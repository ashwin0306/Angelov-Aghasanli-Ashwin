function [X, y] = make_blob_dataset(n_samples, n_features, centers, cluster_std, seed)
    if nargin < 1, n_samples = 6000; end
    if nargin < 2, n_features = 20; end
    if nargin < 3, centers = 10; end
    if nargin < 4, cluster_std = 2.0; end
    if nargin < 5, seed = 42; end
    
    rng(seed);
    center_coords = randn(centers, n_features) * 10;
    samples_per_cluster = floor(n_samples / centers);
    remainder = mod(n_samples, centers);
    
    X = zeros(n_samples, n_features, 'single');
    y = zeros(n_samples, 1);
    
    idx = 1;
    for c = 1:centers
        if c <= remainder
            n_cluster = samples_per_cluster + 1;
        else
            n_cluster = samples_per_cluster;
        end
        cluster_samples = randn(n_cluster, n_features) * cluster_std + center_coords(c, :);
        X(idx:idx+n_cluster-1, :) = single(cluster_samples);
        y(idx:idx+n_cluster-1) = c - 1;
        idx = idx + n_cluster;
    end
end
