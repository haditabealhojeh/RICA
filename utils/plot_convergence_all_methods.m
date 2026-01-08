function plot_convergence_all_methods(problem_name, conv_histories, method_names, out_filename)
% plot_convergence_all_methods
%   Plots the mean convergence curves of all algorithms for one problem.
%
% Inputs:
%   problem_name: string (e.g., 'dominant')
%   conv_histories: cell array {num_methods, num_datasets}, each cell is a vector (iterations)
%   method_names: cell array of method names (e.g., {'RICA','mDTMA','GA'})
%   out_filename: file name to save the plot (e.g., 'convergence_dominant.png')
%

num_methods = length(method_names);
num_datasets = size(conv_histories,2);

% Find the minimum length of all convergence curves for fair average
minlen = min(cellfun(@length, conv_histories(:)));
curve_mat = zeros(num_methods, num_datasets, minlen);

for m = 1:num_methods
    for d = 1:num_datasets
        curve = conv_histories{m, d};
        curve_mat(m, d, :) = curve(1:minlen);
    end
end

figure; hold on;
clr = lines(num_methods);
for m = 1:num_methods
    mean_curve = squeeze(mean(curve_mat(m,:,:), 2));
    std_curve = squeeze(std(curve_mat(m,:,:), 0, 2));
    plot(mean_curve, 'LineWidth', 2, 'Color', clr(m,:));
    % Optional: add shaded error bar (std)
    fill([(1:minlen) (minlen:-1:1)], ...
         [mean_curve'+std_curve'; flip(mean_curve'-std_curve')], ...
         clr(m,:), 'FaceAlpha', 0.12, 'EdgeColor', 'none');
end
xlabel('Iteration'); ylabel('Cost');
title(['Convergence Comparison: ', upper(problem_name)]);
legend(method_names, 'Location', 'northeast');
grid on; hold off;

if nargin >= 4 && ~isempty(out_filename)
    saveas(gcf, out_filename);
end

end