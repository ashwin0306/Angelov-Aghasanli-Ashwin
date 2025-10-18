function run_benchmark(varargin)
    % Parse arguments
    p = inputParser;
    addParameter(p, 'samples', 8000);
    addParameter(p, 'features', 20);
    addParameter(p, 'centers', 12);
    addParameter(p, 'std', 2.0);
    addParameter(p, 'seed', 42);
    addParameter(p, 'split', 0.5);
    addParameter(p, 'K', 60);
    addParameter(p, 'batch', 800);
    addParameter(p, 'eta', 10.0);
    addParameter(p, 'iters', 2);
    addParameter(p, 'plots', false);
    addParameter(p, 'plot_prefix', 'bench_blobs');
    parse(p, varargin{:});
    args = p.Results;
    
    % Generate dataset
    [X, y] = make_blob_dataset(args.samples, args.features, args.centers, args.std, args.seed);
    [X_init, y_init, X_rem, y_rem] = split_initial_and_stream(X, y, args.split, args.seed);
    n0 = length(y_init); 
    
    print_block(sprintf('Dataset: %d samples, %dD, centers=%d | init=%d, stream=%d (split=%.2f)', ...
        args.samples, args.features, args.centers, n0, length(X_rem), args.split));
    
    K_safe = min(args.K, max(2, n0 - 1));
    results = {};
    
    % i-RSNE
    print_block('i-RSNE');
    irsne = IRSNE(K_safe, args.eta, max(1, args.iters), args.seed);
    tic;
    irsne.fit_init(X_init, y_init);
    batches = stream_batches(X_rem, y_rem, args.batch);
    for i = 1:length(batches)
        Xb = batches{i}{1};
        yb = batches{i}{2};
        for j = 1:size(Xb, 1)
            irsne.add_point(Xb(j, :), yb(j));
        end
    end
    t_i = toc;
    [X_all_i, Y_all_i, labels_i] = irsne.get_embedding();
    results{end+1} = eval_and_print('i-RSNE', Y_all_i, labels_i, n0, t_i);
    if args.plots
        scatter_embedding(Y_all_i, labels_i, [args.plot_prefix '_irsne.png'], 'i-RSNE (blobs)');
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
    [X_all_b, Y_all_b, labels_b] = birsne.get_embedding();
    results{end+1} = eval_and_print('Bi-RSNE', Y_all_b, labels_b, n0, t_bi);
    if args.plots
        scatter_embedding(Y_all_b, labels_b, [args.plot_prefix '_birsne.png'], 'Bi-RSNE (blobs)');
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
        scatter_embedding(Y_full, y, [args.plot_prefix '_bh.png'], 'Barnes-Hut t-SNE (blobs)');
    end
    
    print_block('Summary');
    fprintf('%12s | %8s | %10s | %8s\n', 'Method', 'time(s)', 'Silhouette', 'DB');
    fprintf('%s\n', repmat('-', 1, 70));
    for i = 1:length(results)
        r = results{i};
        fprintf('%12s | %8.2f | %10.4f | %8.4f\n', r.name, r.time, r.silhouette, r.db);
    end
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