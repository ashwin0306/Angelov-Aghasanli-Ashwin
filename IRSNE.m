classdef IRSNE < handle
    properties
        K          % Number of cluster centers
        eta        % Learning rate for gradient descent
        iters      % Number of iterations per point
        seed       % Random seed
        clusters   % Cell array of cluster statistics
        C_high     % Cluster centers in highD 
        C_low      % Cluster centers in lowD
        sigma      % Cluster spread in highD
        D          % Dimensionality of input data
        X_list     % Cell array storing all highD points
        Y_list     % Cell array storing all lowD points
        y_list     % Array storing all labels
    end
    
    properties (Constant)
        EPS = 1e-9;
    end
    
    methods
        function obj = IRSNE(K, eta, iters, seed)
            if nargin < 1, K = 50; end
            if nargin < 2, eta = 10.0; end
            if nargin < 3, iters = 1; end
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
                        'count', 1, ...
                        'sum_sq', 0.0);
                else
                    hd = X_init(m, :);
                    ld = Y_init(m, :);
                    norms = sqrt(sum(hd.^2, 2));
                    obj.clusters{k} = struct(...
                        'high_mean', mean(hd, 1, 'native'), ...
                        'low_mean', mean(ld, 1, 'native'), ...
                        'std', std(norms), ... %spread of clusters based on norms
                        'count', sum(m), ...
                        'sum_sq', mean(norms.^2));
                end
            end
            obj.refresh_cache();
        end
        
        function Y_init = fit_init(obj, X_init, y_init)
            obj.D = size(X_init, 2);
            
            [labels, ~] = kmeans(X_init, obj.K, 'MaxIter', 100, 'Replicates', 1);
            labels = labels - 1;
            
            rng(obj.seed);
            Y_init = tsne(X_init, 'NumDimensions', 2, 'Algorithm', 'barneshut');

            Y_init = single(Y_init);
            
            obj.init_clusters(X_init, Y_init, labels);
            
            for i = 1:size(X_init, 1)
                obj.X_list{end+1} = X_init(i, :);
                obj.Y_list{end+1} = Y_init(i, :);
            end
            obj.y_list = [obj.y_list; y_init(:)];
        end 
        
        function add_point(obj, x, y_label)
            x = single(x(:)');
            
            % Finding the closest cluster
            d2h = sum((obj.C_high - x).^2, 2);
            [~, k] = min(d2h);
            
            % Initialize position
            y = obj.C_low(k, :) + 0.1 * randn(1, 2, 'single');
            
            % Gradient descent
            for iter = 1:obj.iters
                P = exp(-d2h ./ (2 * obj.sigma.^2));
                P = P / (sum(P) + obj.EPS);
                
                d2l = sum((obj.C_low - y).^2, 2);
                Q = 1.0 ./ (1.0 + d2l);
                Q = Q / (sum(Q) + obj.EPS);
                
                coef = 2.0 * (P - Q) ./ (1.0 + d2l);
                grad = sum(coef .* (y - obj.C_low), 1);
                y = y - obj.eta * grad;
            end
            
            % Update cluster statistics
            c = obj.clusters{k};
            n0 = c.count;
            m0 = c.high_mean;
            sum0 = c.sum_sq * max(n0, 1);
            
            total = n0 + 1;
            m1 = (n0 * m0 + x) / total;
            norm2 = sum(x.^2);
            sum_sq1 = (sum0 + norm2) / total;
            var = sum_sq1 - sum(m1.^2);
            
            obj.clusters{k}.high_mean = m1;
            obj.clusters{k}.sum_sq = sum_sq1;
            obj.clusters{k}.std = sqrt(max(var, 1e-9));
            obj.clusters{k}.count = total;
            
            obj.refresh_cache();
            
            obj.X_list{end+1} = x;
            obj.Y_list{end+1} = y;
            obj.y_list = [obj.y_list; y_label];
        end
        
        function [X_all, Y_all, y_all] = get_embedding(obj)
            X_all = cell2mat(obj.X_list');
            Y_all = cell2mat(obj.Y_list');
            y_all = obj.y_list;
        end
    end
end