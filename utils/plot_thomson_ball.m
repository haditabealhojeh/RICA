function plot_thomson_ball(Ycell, varargin)
% plot_thomson_ball - Visualize points of the Thomson problem on the unit sphere.
% Ycell: cell array of vectors (each a 3D point on the sphere)
% Optional: 'Highlight' (index or indices to highlight as imperialists/best)
% Example:
%   plot_thomson_ball(all_countries, 'Highlight', best_idx)

p = inputParser;
addParameter(p, 'Highlight', []);
parse(p, varargin{:});
highlight = p.Results.Highlight;

% Prepare data
Ymat = cell2mat(reshape(Ycell,1,[]));
if size(Ymat,1) ~= 3
    error('Thomson ball plot expects 3D points! Each entry in Ycell must be 3x1.');
end
Ymat = Ymat';

% Plot the sphere
figure; hold on;
[xs, ys, zs] = sphere(80);
surf(xs, ys, zs, 'FaceAlpha', 0.07, 'EdgeColor', 'none', 'FaceColor', [0.7 0.7 1]);
colormap winter

% Plot all points
scatter3(Ymat(:,1), Ymat(:,2), Ymat(:,3), 60, 'b', 'filled');
% Optionally highlight some points
if ~isempty(highlight)
    scatter3(Ymat(highlight,1), Ymat(highlight,2), Ymat(highlight,3), 120, 'r', 'filled', 'MarkerEdgeColor', 'k');
end

axis equal; grid on;
xlabel('x'); ylabel('y'); zlabel('z');
title('Thomson Problem: Ball Plot');
view(3);

end