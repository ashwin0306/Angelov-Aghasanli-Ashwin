classdef BiRSNE < handle
    properties
        K
        eta
        iters
        seed
        clusters
        C_high
        C_low
        sigma
        D
        X_list
        Y_list
        y_list
    end
    
    properties (Constant)
        EPS = 1e-9;
    end
    
    methods
        function obj = BiRSNE(K, eta, iters, seed)
            if nargin < 1, K = 50; end
            if nargin < 2, eta = 10.0; end
            if nargin < 3, iters = 2; end
            if nargin < 4, seed = 42; end
            
            obj.K = K;
            obj.eta = eta;
            obj.iters = iters;
            obj.seed = seed;
            obj.clusters = [];
            obj.C_high = [];
            obj.C_low = [];
            obj.sigma = [];
            obj.D = [];
            obj.X_list = {};
            obj.Y_list = {};
            obj.y_list = [];
        end
        
        function refresh_cache(obj)
            K = length(obj.clusters);
            D = size(obj.clusters{1}.high_mean, 2);
            obj.C_high = zeros(K, D, 'single');
            obj.C_low = zeros(K, 2, 'single');
            obj.sigma = zeros(K, 1, 'single');
            
            for k = 1:K
                obj.C_high(k, :) = obj.clusters{k}.high_mean;
                obj.C_low(k, :) = obj.clusters{k}.low_mean;
                obj.sigma(k) = max(obj.clusters{k}.std, 1e-3);
            end
            obj.sigma = max(obj.sigma, 1e-3);
        end
        
        function init_clusters(obj, X_init, Y_init, labels)
            K = obj.K;
            D = size(X_init, 2);
            obj.clusters = cell(K, 1);
            
            for k = 1:K
                m = (labels == (k-1));
                if ~any(m)
                    obj.clusters{k} = struct(...
                        'high_mean', zeros(1, D, 'single'), ...
                        'low_mean', zeros(1, 2, 'single'), ...
                        'std', 1.0, ...
                        'count', 1);
                else
                    hd = X_init(m, :);
                    ld = Y_init(m, :);
                    norms = sqrt(sum(hd.^2, 2));
                    obj.clusters{k} = struct(...
                        'high_mean', mean(hd, 1, 'native'), ...
                        'low_mean', mean(ld, 1, 'native'), ...
                        'std', std(norms), ...
                        'count', sum(m));
                end
            end
            obj.refresh_cache();
        end
        
        function Y_init = fit_init(obj, X_init, y_init)
            obj.D = size(X_init, 2);
            
            % KMeans clustering
            [labels, ~] = kmeans(X_init, obj.K, 'MaxIter', 100, 'Replicates', 1);
            labels = labels - 1; % 0-indexed
            
            % t-SNE embedding
            rng(obj.seed);
            Y_init = tsne(X_init, 'NumDimensions', 2, 'Algorithm', 'barneshut');
            Y_init = single(Y_init);
            
            obj.init_clusters(X_init, Y_init, labels);
            
            % Store data
            for i = 1:size(X_init, 1)
                obj.X_list{end+1} = X_init(i, :);
                obj.Y_list{end+1} = Y_init(i, :);
            end
            obj.y_list = [obj.y_list; y_init(:)];
        end
        
        function add_batch(obj, Xb, yb)
            if isempty(Xb), return; end
            Xb = single(Xb);
            B = size(Xb, 1);
            
            % Compute distances to cluster centers
            d2 = zeros(B, obj.K, 'single');
            for k = 1:obj.K
                diff = Xb - obj.C_high(k, :);
                d2(:, k) = sum(diff.^2, 2);
            end
            
            % Find closest cluster for each point
            [~, idx] = min(d2, [], 2);
            
            % Initialize low-dimensional positions
            Yb = obj.C_low(idx, :) + 0.1 * randn(B, 2, 'single');
            
            % Gradient descent iterations
            for iter = 1:obj.iters
                % Compute P (high-dimensional affinities)
                P = exp(-d2 ./ (2 * obj.sigma'.^2));
                P = P ./ (sum(P, 2) + obj.EPS);
                
                % Compute Q (low-dimensional affinities)
                d2l = zeros(B, obj.K, 'single');
                for k = 1:obj.K
                    diff = Yb - obj.C_low(k, :);
                    d2l(:, k) = sum(diff.^2, 2);
                end
                Q = 1.0 ./ (1.0 + d2l);
                Q = Q ./ (sum(Q, 2) + obj.EPS);
                
                % Compute gradient
                coef = 2.0 * (P - Q) ./ (1.0 + d2l);
                grads = zeros(B, 2, 'single');
                for k = 1:obj.K
                    diff = Yb - obj.C_low(k, :);
                    grads = grads + coef(:, k) .* diff;
                end
                Yb = Yb - obj.eta * grads;
            end
            
            % Storing the data
            for i = 1:B
                obj.X_list{end+1} = Xb(i, :);
                obj.Y_list{end+1} = Yb(i, :);
            end
            obj.y_list = [obj.y_list; yb(:)];
        end
        
        function [X_all, Y_all, y_all] = get_embedding(obj)
            X_all = cell2mat(obj.X_list');
            Y_all = cell2mat(obj.Y_list');
            y_all = obj.y_list;
        end
    end
end