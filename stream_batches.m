function batches = stream_batches(X, y, batch_size)
    if nargin < 3, batch_size = 500; end
    
    n = size(X, 1);
    n_batches = ceil(n / batch_size);
    batches = cell(n_batches, 1);
    
    %since matlab does not have yield features like python, returns a cell array of batches. There is more memory usage.
    for i = 1:n_batches
        start_idx = (i-1) * batch_size + 1;
        end_idx = min(i * batch_size, n);
        batches{i} = {X(start_idx:end_idx, :), y(start_idx:end_idx)};
    end
end