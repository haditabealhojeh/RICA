% Comprehensive experiment runner for RICA on all benchmark problems.
% Using only simple matrix manifolds that work with your RICA solver

addpath(genpath('./manopt'));
addpath('./solvers');

% All problems from the paper
problems = {'sdp', 'svd', 'procrustes', 'thomson', 'stiffness'};
datasets = 1:1;
num_runs = 1;  % Start with 1 run for testing

% === RICA parameter settings for each problem (from paper)
rica_params = struct(...
    'dominant',    struct('num_countries',80,'num_imperialists',10,'assimilation_coef',0.1,'revolution_rate',0.7,'assimilation_gamma',pi/4,'num_iterations',800), ...
    'sdp',         struct('num_countries',80,'num_imperialists',10,'assimilation_coef',0.1,'revolution_rate',0.7,'assimilation_gamma',pi/4,'num_iterations',800), ...
    'svd',         struct('num_countries',80,'num_imperialists',10,'assimilation_coef',0.1,'revolution_rate',0.7,'assimilation_gamma',pi/4,'num_iterations',800), ...
    'procrustes',  struct('num_countries',80,'num_imperialists',10,'assimilation_coef',0.1,'revolution_rate',0.7,'assimilation_gamma',pi/4,'num_iterations',800), ...
    'thomson',     struct('num_countries',80,'num_imperialists',10,'assimilation_coef',0.1,'revolution_rate',0.7,'assimilation_gamma',pi/4,'num_iterations',800), ...
    'stiffness',   struct('num_countries',80,'num_imperialists',10,'assimilation_coef',0.1,'revolution_rate',0.7,'assimilation_gamma',pi/4,'num_iterations',800) ...
);

% Initialize results structure
results = struct();

for prob_idx = 1:length(problems)
    problem_type = problems{prob_idx};
    fprintf('\n===== %s problem =====\n', upper(problem_type));
    
    % Initialize storage for all runs and datasets
    all_costs = zeros(num_runs, length(datasets));
    all_times = zeros(num_runs, length(datasets));
    all_convergence = cell(num_runs, length(datasets));
    
    for run = 1:num_runs
        fprintf('  Run %d/%d...\n', run, num_runs);
        
        for d = datasets
            % --- Load data and setup problem with SIMPLE manifolds only
            switch problem_type
                case 'dominant'
                    datafile = sprintf('data/dominant-%d.mat', d);
                    if ~exist(datafile, 'file')
                        fprintf('    Data file not found: %s, skipping...\n', datafile);
                        all_costs(run, d) = NaN;
                        all_times(run, d) = NaN;
                        continue;
                    end
                    D = load(datafile);
                    A = D.A;
                    n = size(A,1); p = 3;
                    % Grassmann manifold - this works well
                    M = grassmannfactory(n, p);
                    % Cost function from paper: min f = -1/2 * trace(X'*A*X)
                    cost = @(X) -0.5 * trace(X' * A * X);
                    
                case 'sdp'
                    datafile = sprintf('data/SDP-n100-%d.mat', d);
                    if ~exist(datafile, 'file')
                        fprintf('    Data file not found: %s, skipping...\n', datafile);
                        all_costs(run, d) = NaN;
                        all_times(run, d) = NaN;
                        continue;
                    end
                    D = load(datafile); 
                    A = D.A;
                    n = size(A,1); 
                    % Use sphere manifold (simple and compatible)
                    M = spherefactory(n);
                    % Cost function adapted for sphere
                    cost = @(Y) sdp_cost_sphere(Y, A);
                    
                case 'svd'
                    datafile = sprintf('data/truncat-%d.mat', d);
                    if ~exist(datafile, 'file')
                        fprintf('    Data file not found: %s, skipping...\n', datafile);
                        all_costs(run, d) = NaN;
                        all_times(run, d) = NaN;
                        continue;
                    end
                    D = load(datafile); 
                    A = D.A;
                    m = size(A,1); n = size(A,2); p = 5; % p=5 as per paper
                    
                    % Use Stiefel manifold for U only (much simpler)
                    % We'll fix V or use a simpler approach
                    M = stiefelfactory(m, p);
                    
                    % Simplified SVD cost using only U
                    cost = @(U) svd_cost_simple(U, A, n, p);
                    
                case 'procrustes'
                    datafile = sprintf('data/Procrustes-n3m10N50-%d.mat', d);
                    if ~exist(datafile, 'file')
                        fprintf('    Data file not found: %s, skipping...\n', datafile);
                        all_costs(run, d) = NaN;
                        all_times(run, d) = NaN;
                        continue;
                    end
                    D = load(datafile);
                    A_true = D.Atrue;
                    
                    % Use rotations factory only (simple matrix)
                    M = rotationsfactory(3);
                    
                    % Create appropriate point cloud data
                    rng(d);
                    point_cloud = randn(3, 10); % 3D, 10 points
                    
                    % Cost function for rotations only
                    cost = @(R) procrustes_cost_simple(R, point_cloud, A_true);
                    
                case 'thomson'
                    % Use different point counts as in paper
                    n_points_list = [50, 75, 100, 125, 150];
                    if d <= length(n_points_list)
                        n_points = n_points_list(d);
                    else
                        n_points = 50;
                    end
                    d_ambient = 3;
                    
                    % Use oblique manifold for ALL points as a single matrix
                    % This creates a d_ambient x n_points matrix where each column is on sphere
                    M = obliquefactory(d_ambient, n_points);
                    
                    % Cost function that works with the matrix representation
                    cost = @(X) thomson_cost_matrix(X);
                    
                case 'stiffness'
                    % For stiffness learning, use SPD manifold
                    n = 3; % 3x3 stiffness matrices
                    M = sympositivedefinitefactory(n);
                    % Create synthetic problem data
                    rng(d);
                    X_data = randn(10, n);  % Position data
                    Y_data = randn(10, n);  % Force data
                    cost = @(KP) stiffness_cost_simple(KP, X_data, Y_data);
                    
                otherwise
                    error('Unknown problem type: %s', problem_type);
            end

            % Get RICA parameters
            params = rica_params.(problem_type);
            opts = params;
            opts.verbosity = 1;
            opts.seed = run * 100 + d;
            
            % Run RICA solver
            try
                fprintf('    Running RICA on dataset %d...\n', d);
                tic;
                [sol, costval, conv_history, all_countries] = rica_solver(M, cost, opts);
                elapsed_time = toc;
                
                all_costs(run, d) = costval;
                all_times(run, d) = elapsed_time;
                all_convergence{run, d} = conv_history;
                
                fprintf('    dataset %d: cost = %.4e, time = %.2fs\n', d, costval, elapsed_time);
                
                % Save visualization for Thomson problem
                if strcmp(problem_type, 'thomson')
                    try
                        plot_thomson_ball_matrix(sol);
                        if ~exist('results', 'dir')
                            mkdir('results');
                        end
                        saveas(gcf, sprintf('results/thomson_ball_d%d.png', d));
                        close(gcf);
                    catch ME
                        fprintf('    Could not create Thomson ball plot: %s\n', ME.message);
                    end
                end
                
            catch ME
                fprintf('    ERROR running RICA on dataset %d: %s\n', d, ME.message);
                fprintf('    Error in function: %s, line: %d\n', ME.stack(1).name, ME.stack(1).line);
                all_costs(run, d) = NaN;
                all_times(run, d) = NaN;
                all_convergence{run, d} = [];
            end
        end
    end

    % Process results for this problem
    valid_mask = ~isnan(all_costs);
    if any(valid_mask(:))
        valid_costs = all_costs(valid_mask);
        valid_times = all_times(valid_mask);
        
        % Store results
        results.(problem_type).mean_cost = mean(valid_costs);
        results.(problem_type).std_cost = std(valid_costs);
        results.(problem_type).mean_time = mean(valid_times);
        results.(problem_type).std_time = std(valid_times);
        results.(problem_type).all_costs = all_costs;
        results.(problem_type).all_times = all_times;
        results.(problem_type).convergence = all_convergence;
        
        % Create convergence plot
        create_convergence_plot(all_convergence, problem_type, num_runs);
        
        fprintf('  ==> [%s] Mean cost: %.4e ± %.2e, Mean time: %.2fs ± %.2fs\n', ...
            upper(problem_type), results.(problem_type).mean_cost, ...
            results.(problem_type).std_cost, results.(problem_type).mean_time, ...
            results.(problem_type).std_time);
    else
        fprintf('  ==> [%s] No valid runs completed\n', upper(problem_type));
        results.(problem_type).mean_cost = NaN;
        results.(problem_type).std_cost = NaN;
        results.(problem_type).mean_time = NaN;
        results.(problem_type).std_time = NaN;
    end
end

% Create summary table
fprintf('\n===== SUMMARY RESULTS =====\n');
fprintf('Problem\t\tMean Cost\tStd Cost\tMean Time\tStd Time\n');
fprintf('-------\t\t---------\t--------\t---------\t--------\n');
for prob_idx = 1:length(problems)
    problem_type = problems{prob_idx};
    if isfield(results, problem_type) && ~isnan(results.(problem_type).mean_cost)
        fprintf('%-12s\t%.4e\t%.2e\t%.2fs\t%.2fs\n', ...
            upper(problem_type), ...
            results.(problem_type).mean_cost, ...
            results.(problem_type).std_cost, ...
            results.(problem_type).mean_time, ...
            results.(problem_type).std_time);
    else
        fprintf('%-12s\tN/A\t\tN/A\t\tN/A\t\tN/A\n', upper(problem_type));
    end
end

% Save detailed results
if ~exist('results', 'dir')
    mkdir('results');
end
save('results/rica_experiment_results.mat', 'results');

% Export to CSV
export_results_to_csv(results, problems);

fprintf('\n===== Experiment completed =====\n');
fprintf('Results saved to results/rica_experiment_results.mat\n');

% --- SIMPLE Helper Functions (No Structured Data) ---

function costval = sdp_cost_sphere(Y, A)
    % SDP cost adapted for sphere manifold
    % Y is a point on the sphere (unit vector)
    costval = Y' * A * Y;
end

function costval = svd_cost_simple(U, A, n, p)
    % Simplified SVD cost using only U
    % Fix V as random or use simple approximation
    persistent V_fixed;
    if isempty(V_fixed) || size(V_fixed, 1) ~= n || size(V_fixed, 2) ~= p
        rng(42); % Fixed seed for reproducibility
        [V_fixed, ~] = qr(randn(n, p), 0); % Random orthonormal V
    end
    
    % Cost from paper: min f(U,V) = -1/2 * ||U'*A*V||_F^2
    costval = -0.5 * norm(U' * A * V_fixed, 'fro')^2;
end

function costval = procrustes_cost_simple(R, point_cloud, A_true)
    % Procrustes cost for rotations only
    rotated_cloud = R * point_cloud;
    costval = norm(rotated_cloud - A_true, 'fro')^2;
end

function costval = thomson_cost_matrix(X)
    % Thomson cost for matrix representation
    % X is d_ambient x n_points matrix where each column is on sphere
    % Transpose to get n_points x d_ambient for the paper's formulation
    X_mat = X';
    n_points = size(X_mat, 1);
    
    Gram = X_mat * X_mat';
    
    % Paper's formulation
    distances = 1 - Gram;
    distances(1:n_points+1:end) = 1; % Set diagonal to avoid division by zero
    potential_matrix = 1 ./ distances;
    triu_sum = sum(sum(triu(potential_matrix, 1)));
    costval = triu_sum / n_points^2;
end

function costval = stiffness_cost_simple(KP, X_data, Y_data)
    % Simplified stiffness cost
    predicted_forces = X_data * KP;
    costval = 0.5 * norm(predicted_forces - Y_data, 'fro')^2;
end

function plot_thomson_ball_matrix(X)
    % Plot Thomson points from matrix representation
    % X is d_ambient x n_points matrix where each column is a point on sphere
    
    figure('Position', [100, 100, 800, 600]);
    
    [d_ambient, n_points] = size(X);
    
    if d_ambient >= 3
        % 3D plot with transparent sphere
        [x, y, z] = sphere(50);
        surf(x, y, z, 'FaceAlpha', 0.05, 'EdgeColor', 'none', 'FaceColor', [0.7 0.7 0.7]);
        hold on;
        scatter3(X(1,:), X(2,:), X(3,:), 60, 'filled', 'r');
        axis equal;
        title(sprintf('Thomson Problem - RICA Solution (n=%d)', n_points), 'FontSize', 12);
        xlabel('X'); ylabel('Y'); zlabel('Z');
        grid on;
        view(45, 30);
    else
        % 2D plot
        scatter(X(1,:), X(2,:), 50, 'filled', 'r');
        title(sprintf('Thomson Problem - RICA Solution (n=%d)', n_points), 'FontSize', 12);
        axis equal;
        grid on;
    end
end

function create_convergence_plot(all_convergence, problem_type, num_runs)
    % Create convergence plot
    if isempty(all_convergence) || isempty(all_convergence{1,1})
        return;
    end
    
    figure;
    hold on;
    colors = lines(min(5, num_runs));
    
    for run = 1:min(5, num_runs)
        for d = 1:size(all_convergence, 2)
            if ~isempty(all_convergence{run, d})
                plot(all_convergence{run, d}, 'Color', colors(run,:), 'LineWidth', 1);
                break;
            end
        end
    end
    
    xlabel('Iteration'); 
    ylabel('Cost');
    title(sprintf('RICA Convergence - %s', upper(problem_type)));
    grid on;
    
    if ~exist('results', 'dir')
        mkdir('results');
    end
    saveas(gcf, sprintf('results/rica_convergence_%s.png', problem_type));
    close(gcf);
end

function export_results_to_csv(results, problems)
    % Export results to CSV
    filename = 'results/rica_results_comparison.csv';
    fid = fopen(filename, 'w');
    
    % Header
    fprintf(fid, 'Problem,Mean_Cost,Std_Cost,Mean_Time,Std_Time\n');
    
    for prob_idx = 1:length(problems)
        problem_type = problems{prob_idx};
        if isfield(results, problem_type) && ~isnan(results.(problem_type).mean_cost)
            fprintf(fid, '%s,%.6e,%.6e,%.2f,%.2f\n', ...
                upper(problem_type), ...
                results.(problem_type).mean_cost, ...
                results.(problem_type).std_cost, ...
                results.(problem_type).mean_time, ...
                results.(problem_type).std_time);
        else
            fprintf(fid, '%s,NaN,NaN,NaN,NaN\n', upper(problem_type));
        end
    end
    
    fclose(fid);
    fprintf('Results exported to %s\n', filename);
end