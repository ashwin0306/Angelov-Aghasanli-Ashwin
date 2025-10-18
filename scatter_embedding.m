function scatter_embedding(Y, labels, out_path, title_str)
    if nargin < 3, out_path = ''; end
    if nargin < 4, title_str = 'Embedding'; end
    
    figure('Position', [100, 100, 800, 800]); %Size of the figure window in pixels
    hold on;
    unique_labels = unique(labels);
    colors = lines(length(unique_labels));
    
    for i = 1:length(unique_labels)
        cls = unique_labels(i);
        m = (labels == cls);
        scatter(Y(m, 1), Y(m, 2), 6, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.4);
    end
    
    set(gca, 'XTick', [], 'YTick', []); %sets properties of current axes
    title(title_str);
    
    %It is true if out_path is not an empty string
    if ~isempty(out_path)
        exportgraphics(gcf, out_path, 'Resolution', 300);
    end
    hold off;
end