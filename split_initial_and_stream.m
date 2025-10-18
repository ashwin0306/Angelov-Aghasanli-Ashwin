function [X_init, y_init, X_stream, y_stream] = split_initial_and_stream(X, y, split_ratio, seed)
    %Half goes to stream, half to initial
    if nargin < 3, split_ratio = 0.5; end
    if nargin < 4, seed = 42; end
    
    rng(seed);
    n = size(X, 1);
    idx = randperm(n);
    n0 = floor(split_ratio * n);
    init_idx = idx(1:n0);
    stream_idx = idx(n0+1:end);
    
    X_init = X(init_idx, :);
    y_init = y(init_idx);
    X_stream = X(stream_idx, :);
    y_stream = y(stream_idx);
end