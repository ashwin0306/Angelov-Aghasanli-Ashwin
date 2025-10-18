function run_benchmark_features(varargin)
    % Parsing through input arguments
    p = inputParser;
    addParameter(p, 'prefix', '');
    addParameter(p, 'features', '');
    addParameter(p, 'labels', '');
    addParameter(p, 'split', 0.5);
    addParameter(p, 'seed', 42);
    addParameter(p, 'K', 100);
    addParameter(p, 'batch', 1000);
    addParameter(p, 'eta', 10.0);
    addParameter(p, 'iters', 2);
    addParameter(p, 'per_class', inf);
    addParameter(p, 'plots', false);
    addParameter(p, 'plot_prefix', 'bench_features');
    parse(p, varargin{:});
    args = p.Results;
    
    % Load features and labels
    if ~isempty(args.features)
        if isempty(args.labels)
            error('labels parameter required when features is provided');
        end
        X = single(readNPY(args.features));
        y = readNPY(args.labels);
        prefix_name = args.features;
    elseif ~isempty(args.prefix)
        X = single(readNPY_simple([args.prefix '_features.npy']));
        y = readNPY_simple([args.prefix '_labels.npy']);
        prefix_name = args.prefix;
    else
        error('Either prefix or features+labels must be provided');
    end
    
    if ~isinf(args.per_class)
        fprintf('Capping to %d samples per class...\n', args.per_class);
        [X, y] = cap_per_class(X, y, args.per_class, args.seed);
    end
    
    % Stratified split
    [X_init, y_init, X_rem, y_rem] = stratified_split(X, y, args.split, args.seed);
    n0 = length(y_init);
    D = size(X, 2);
    
    print_block(sprintf('Dataset: %s | %d samples, D=%d | init=%d, stream=%d (split=%.2f)', ...
                        prefix_name, size(X,1), D, n0, length(X_rem), args.split));
    
    K_safe = min(args.K, max(2, n0 - 1));
    results = {};
    
    warning('off', 'stats:pdist2:ConvertingToDouble');
    
    % i-RSNE
    print_block('i-RSNE');
    irsne = IRSNE(K_safe, args.eta, max(1, args.iters), args.seed);
    tic; %starting the timer
    irsne.fit_init(X_init, y_init);
    batches = stream_batches(X_rem, y_rem, args.batch);
    for i = 1:length(batches)
        Xb = batches{i}{1};
        yb = batches{i}{2};
        for j = 1:size(Xb, 1)
            irsne.add_point(Xb(j, :), yb(j));
        end
    end
    t_i = toc; %stopping the timer
    [~, Y_all_i, labels_i] = irsne.get_embedding();
    results{end+1} = eval_and_print('i-RSNE', Y_all_i, labels_i, n0, t_i);
    if args.plots
        scatter_embedding(Y_all_i, labels_i, [args.plot_prefix '_irsne.png'], 'i-RSNE (features)');
    end
    
    % Bi-RSNE
    print_block('Bi-RSNE');
    birsne = BiRSNE(K_safe, args.eta, max(2, args.iters), args.seed);
    tic; 
    birsne.fit_init(X_init, y_init);
    batches = stream_batches(X_rem, y_rem, args.batch);
    for i = 1:length(batches)
        Xb = batches{i}{1};
        yb = batches{i}{2};
        birsne.add_batch(Xb, yb);
    end
    t_bi = toc;
    [~, Y_all_b, labels_b] = birsne.get_embedding();
    results{end+1} = eval_and_print('Bi-RSNE', Y_all_b, labels_b, n0, t_bi);
    if args.plots
        scatter_embedding(Y_all_b, labels_b, [args.plot_prefix '_birsne.png'], 'Bi-RSNE (features)');
    end
    
    % Barnes-Hut t-SNE
    print_block('Barnes-Hut t-SNE (full)');
    rng(args.seed);
    tic;
    Y_full = tsne(X, 'NumDimensions', 2, 'Algorithm', 'barneshut');
    t_tsne = toc;
    Y_full = single(Y_full);
    results{end+1} = eval_and_print('BH t-SNE', Y_full, y, n0, t_tsne);
    if args.plots
        scatter_embedding(Y_full, y, [args.plot_prefix '_bh.png'], 'Barnes-Hut t-SNE (features)');
    end
    
    % Summary
    print_block('Summary');
    fprintf('%12s | %8s | %10s | %8s\n', 'Method', 'time(s)', 'Silhouette', 'DB');
    fprintf('%s\n', repmat('-', 1, 70));
    for i = 1:length(results)
        r = results{i};
        fprintf('%12s | %8.2f | %10.4f | %8.4f\n', r.name, r.time, r.silhouette, r.db);
    end
end


function [X_sub, y_sub] = cap_per_class(X, y, per_class, seed)
    % Cap to per_class samples per class 
    rng(seed);
    classes = unique(y);
    idx_keep = [];
    
    for c = classes'
        idx_c = find(y == c);
        idx_c = idx_c(randperm(length(idx_c)));
        n_take = min(per_class, length(idx_c));
        idx_keep = [idx_keep; idx_c(1:n_take)];
    end
    
    idx_keep = idx_keep(randperm(length(idx_keep)));
    X_sub = X(idx_keep, :);
    y_sub = y(idx_keep);
end


function [X_init, y_init, X_rem, y_rem] = stratified_split(X, y, split_ratio, seed)
    rng(seed);
    classes = unique(y);
    
    X_init_cell = {};
    y_init_cell = {};
    X_rem_cell = {};
    y_rem_cell = {};
    
    for c = classes'
        idx_c = find(y == c);
        idx_c = idx_c(randperm(length(idx_c)));
        n0 = floor(split_ratio * length(idx_c));
        
        X_init_cell{end+1} = X(idx_c(1:n0), :);
        y_init_cell{end+1} = y(idx_c(1:n0));
        X_rem_cell{end+1} = X(idx_c(n0+1:end), :);
        y_rem_cell{end+1} = y(idx_c(n0+1:end));
    end
    
    X_init = vertcat(X_init_cell{:});
    y_init = vertcat(y_init_cell{:});
    X_rem = vertcat(X_rem_cell{:});
    y_rem = vertcat(y_rem_cell{:});
end


function print_block(title)
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('%s\n', title);
    fprintf('%s\n', repmat('=', 1, 70));
end


function result = eval_and_print(name, Y_all, labels, n0, t_sec)
    [sil, db] = clustering_quality(Y_all, labels);
    fprintf('%12s | time: %7.2fs | Silhouette: %6.4f | DB: %6.4f\n', name, t_sec, sil, db);
    result = struct('name', name, 'time', t_sec, 'silhouette', sil, 'db', db);
end


% %% Benchmark CLIP features
% fprintf('=== Benchmarking CLIP ===\n');
% run_benchmark_features('prefix', 'cifar10_5k_clip', ...
%                       'K', 50, ...
%                       'batch', 500, ...
%                       'plots', true, ...
%                       'plot_prefix', 'results_clip');

% %% Benchmark DINOv2 features
% fprintf('\n=== Benchmarking DINOv2 ===\n');
% run_benchmark_features('prefix', 'cifar10_5k_dinov2', ...
%                       'K', 50, ...
%                       'batch', 500, ...
%                       'plots', true, ...
%                       'plot_prefix', 'results_dinov2');