function plot_thomson_paper_single(X, method_name, sphere_data, cmap, view_type)
    % Draw one Thomson sphere plot in paper style
    sx = sphere_data(:,1:end/3);
    sy = sphere_data(:,end/3+1:2*end/3);
    sz = sphere_data(:,2*end/3+1:end);

    surf(sx, sy, sz, ...
        'FaceColor', 'interp', ...
        'FaceAlpha', 1.0, ...
        'EdgeColor', [0.2 0.2 0.2], ...
        'LineStyle', '-', ...
        'LineWidth', 0.5, ...
        'EdgeAlpha', 0.8);
    colormap(cmap);
    shading interp;
    hold on;

    % Thomson points
    scatter3(X(1,:), X(2,:), X(3,:), 20, 'filled', ...
        'MarkerFaceColor', [0 0.447 0.741], ...
        'MarkerEdgeColor', 'none');

    % Lighting
    axis equal off;
    lighting gouraud;
    material dull;
    camlight headlight;

    % View control
    if strcmp(view_type, 'top')
        view(0, 90);  % top
    else
        view(0, 0);   % side
    end

    title(method_name, 'FontWeight', 'bold', 'FontSize', 10);
end
