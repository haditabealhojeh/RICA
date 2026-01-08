function run_manifold_experiments()
% Comprehensive experiment runner for ALL methods from the paper
% Running RICA, mDTMA, mDE, mPSO, conjugate gradient, and trust regions

    % --- 1. SETUP ---
    % Add necessary Manopt paths
    addpath(genpath('./manopt')); 
    addpath("./data")
    addpath("./manopt/autodiff")
    addpath("./manopt/core")
    addpath("./manopt/manifolds")
    addpath("./manopt/solvers")
    addpath("./manopt/tools")
    warning('off', 'all');
    cleanupObj = onCleanup(@() warning('on', 'all'));
    problems = {'dominant','sdp','svd','thomson'};
    datasets = 1:5;   % 5 datasets as in paper
    num_runs = 1;    % 21 independent runs as in paper

    % All six methods to compare
    %methods = {'rica', 'mDTMA', 'mDE', 'mPSO', 'mSMANN', 'conjugategradient', 'trustregions'};
    %method_names = {'RICA', 'mDTMA', 'm-DE', 'm-PSO', 'm-SMANN', 'm-conjugate', 'm-trust region'};
    methods = {'rica'};
    method_names = {'RICA'};
    common_params = struct(...
        'population_size', 80, ...
        'max_iterations', 800, ...
        'verbosity', 1 ...
    );
    w0_params = struct(...
        'thomson', 0.1, ...
        'dominant', 0.1, ...
        'svd', 0.1, ...
        'sdp', 0.4, ...
        'procrustes', 0.4, ...
        'stiffness', 0.4 ...
    );

    % Initialize comprehensive results structure
    results = struct();

    % --- 2. EXPERIMENT LOOP ---
    for prob_idx = 1:length(problems)
        problem_type = problems{prob_idx};
        fprintf('\n===== %s PROBLEM =====\n', upper(problem_type));
        
        % Initialize storage
        for m = 1:length(methods)
            method = methods{m};
            results.(problem_type).(method).all_costs = zeros(num_runs, length(datasets));
            results.(problem_type).(method).all_times = zeros(num_runs, length(datasets));
            results.(problem_type).(method).all_solutions = cell(num_runs, length(datasets));
            results.(problem_type).(method).convergence = cell(num_runs, length(datasets));
        end
        
        for run = 1:num_runs
            fprintf('  Run %d/%d...\n', run, num_runs);
            
            for d = datasets
                fprintf('    Dataset %d: ', d);
                
                % --- Problem setup (same for all methods)
                [M, cost, grad, hess, problem_data] = setup_problem(problem_type, d);
                if isempty(M)
                    fprintf('SKIP (data not found)\n');
                    continue;
                end
                
                % Run all methods on this problem instance
                current_w0 = w0_params.(problem_type);
                
                for m = 1:length(methods)
                    method = methods{m};
                    fprintf('%s ', method_names{m});
                    
                    try
                        tic;
                        switch method
                            case 'rica'
                                [sol, costval, conv_history] = run_rica(M, cost, common_params, current_w0);
                            case 'rica2'
                                [sol, costval, conv_history] = run_rica_de(M, cost,grad, hess, common_params, current_w0);
                            case 'mDTMA'
                                [sol, costval, conv_history] = run_mDTMA(M, cost, grad, common_params, current_w0);
                            case 'mDE'
                                [sol, costval, conv_history] = run_mDE(M, cost, grad, common_params);
                            case 'mPSO'
                                [sol, costval, conv_history] = run_mPSO(M, cost, grad, common_params);
                                
                            case 'conjugategradient'
                                [sol, costval, conv_history] = run_conjugategradient(M, cost, grad, hess, common_params);
                            
                            case 'mSMANN' 
                                [sol, costval, conv_history] = run_mSMANN(M, cost, grad, common_params);
                                
                            case 'trustregions'
                                % Trust regions REQUIRES gradient AND Hessian. If Hessian is missing, use CG.
                                if isempty(hess)
                                    fprintf('[Hessian Missing, Using CG for TR] ');
                                    [sol, costval, conv_history] = run_conjugategradient(M, cost, grad, hess, common_params);
                                else
                                    [sol, costval, conv_history] = run_trustregions(M, cost, grad, hess, common_params);
                                end
                        end
                        elapsed_time = toc;
                        
                        results.(problem_type).(method).all_costs(run, d) = costval;
                        results.(problem_type).(method).all_times(run, d) = elapsed_time;
                        results.(problem_type).(method).all_solutions{run, d} = sol;
                        results.(problem_type).(method).convergence{run, d} = conv_history;
                        
                    catch ME
                        fprintf('[ERROR:%s] ', method);
                        fprintf('Error: %s\n', ME.message);
                        results.(problem_type).(method).all_costs(run, d) = NaN;
                        results.(problem_type).(method).all_times(run, d) = NaN;
                        results.(problem_type).(method).all_solutions{run, d} = [];
                        results.(problem_type).(method).convergence{run, d} = [];
                    end
                end
                fprintf('\n');
            end
        end
        
        % Process results
        process_results(results, problem_type, methods, method_names);
        create_convergence_plots(results, problem_type, methods, method_names);
        if strcmp(problem_type, 'thomson')
            create_thomson_visualizations(results, problem_type, methods, method_names);
        end
        if strcmp(problem_type, 'stiffness')
            create_stiffness_visualizations(results, problem_type, methods, method_names);
        end

        
    end

    % --- 3. FINAL SUMMARY ---
    create_summary_tables(results, problems, methods, method_names);
    create_comparison_plots(results, problems, methods, method_names);
    save_comprehensive_results(results);
    fprintf('\n===== EXPERIMENT COMPLETED =====\n');
    fprintf('All results saved to results/comprehensive_results.mat\n');
end

%-----------------------------------------------------------------------------------------
%% Helper functions for each method
%-----------------------------------------------------------------------------------------

function [sol, costval, conv_history] = run_mSMANN(M, cost, grad, params)
    % 1. Construct the problem struct
    problem.M = M;
    problem.cost = cost;
    if ~isempty(grad)
        problem.egrad = grad; 
    end
    problem = ensure_log_map(problem);
    % 2. Construct the options struct
    options.verbosity = params.verbosity;
    options.initialtemperature = 100;
    options.finaltemperature = 1e-4;

    % 3. Construct the initial population (x0)
    x0 = cell(params.population_size, 1);
    for i = 1:params.population_size
        x0{i} = M.rand();
    end
    
    [sol, costval, info, ~] = simulatedannealing(problem, params.population_size, params.max_iterations, x0, options);
    conv_history = [info.cost];
end

function [sol, costval, conv_history] = run_rica(M, cost, params, w0)
    problem.M = M;
    problem.cost = cost;
    problem = ensure_log_map(problem);
    
    opts = struct();
    opts.num_countries = 100;
    opts.num_imperialists = 10;
    opts.assimilation_coef = 0.3;
    opts.revolution_rate = 0.3;
    opts.assimilation_gamma = pi/4;
    opts.num_iterations = params.max_iterations;
    opts.verbosity = params.verbosity;
    opts.problem = problem;
                                
                                % ====== NEW CONFIGURABLE MECHANISMS ======
                                % Empire Assignment strategies:
                                %   'random' (baseline), 'cost_based', 'roulette'
    opts.empire_assignment = 'power';  
                            
                                % Empire Competition strategies:
                                %   'total_power' (classic), 'min_cost', 'tournament'
    opts.empire_competition = 'probabilistic';  
                            
                                % Colony Update strategies:
                                %   'classic' (assimilation+revolution), 
                                %   'adaptive' (dynamic coefficients),
    opts.colony_update = 'stochastic';  
    [sol, costval, conv_history] = rica_solver_v_2(M, cost, opts);
end

function [sol, costval, conv_history] = run_rica_de(M, cost, grad, hess,params, w0)
    problem.M = M;
    problem.cost = cost;
    problem = ensure_log_map(problem);
    if ~isempty(grad)
        problem.egrad = grad;
    end
    if ~isempty(hess)
        problem.ehess = hess;
    end
    opts = struct();
    opts.num_countries = 150;
    opts.num_imperialists = 5;
    opts.assimilation_coef = 0.3;
    opts.revolution_rate = 0.3;
    opts.assimilation_gamma = pi/4;
    opts.num_iterations = params.max_iterations;
    opts.verbosity = params.verbosity;
    
    opts.problem = problem;
    
    % ====== NEW CONFIGURABLE MECHANISMS ======
    % Empire Assignment strategies:
    %   'random' (baseline), 'cost_based', 'roulette'
    opts.empire_assignment = 'power';  

    % Empire Competition strategies:
    %   'total_power' (classic), 'min_cost', 'tournament'
    opts.empire_competition = 'tournament';  

    % Colony Update strategies:
    %   'classic' (assimilation+revolution), 
    %   'adaptive' (dynamic coefficients),
    %   'stochastic' (randomized assimilation)
    opts.colony_update = 'classic';  
    

    [sol, costval, conv_history] = rica_solver_v_2(problem.M, problem.cost, opts);
end

function [sol, costval, conv_history] = run_mDTMA(M, cost, grad, params, w0)
    problem.M = M;
    problem.cost = cost;
    if ~isempty(grad)
        problem.egrad = grad;
    end
    problem = ensure_log_map(problem);
    x0 = cell(params.population_size, 1);
    for i = 1:params.population_size
        x0{i} = M.rand();
    end
    
    [sol, costval, info, ~] = mDTMA(problem, params.population_size, params.max_iterations, w0, x0, []);
    conv_history = [info.cost];
end

function [sol, costval, conv_history] = run_mDE(M, cost, grad, params)
    problem.M = M;
    problem.cost = cost;
    if ~isempty(grad)
        problem.egrad = grad; 
    end
    problem = ensure_log_map(problem);
    options.verbosity = params.verbosity;
    
    x0 = cell(params.population_size, 1);
    for i = 1:params.population_size
        x0{i} = M.rand();
    end
    
    [sol, costval, info, ~] = diffevolution(problem, params.population_size, params.max_iterations, x0, options); 
    conv_history = [info.cost];
end

function [sol, costval, conv_history] = run_mPSO(M, cost, grad, params)
    problem.M = M;
    problem.cost = cost;
    if ~isempty(grad)
        problem.egrad = grad; 
    end
    problem = ensure_log_map(problem);
    options.verbosity = params.verbosity;
    
    x0 = cell(params.population_size, 1);
    for i = 1:params.population_size
        x0{i} = M.rand();
    end
    
    [sol, costval, info, ~] = pso(problem, params.population_size, params.max_iterations, x0, options);
    conv_history = [info.cost];
end

function [sol, costval, conv_history] = run_conjugategradient(M, cost, grad, hess, params)
    problem.M = M;
    problem.cost = cost;
    
    if ~isempty(grad)
        problem.egrad = grad;
    end
    if ~isempty(hess)
        problem.ehess = hess;
    end
    problem = ensure_log_map(problem);
    options = struct();
    options.maxiter = params.max_iterations;
    options.verbosity = params.verbosity;
    
    [sol, costval, info, ~] = conjugategradient(problem, [], options);
    conv_history = [info.cost];
end

function [sol, costval, conv_history] = run_trustregions(M, cost, grad, hess, params)
    problem.M = M;
    problem.cost = cost;
    
    if ~isempty(grad)
        problem.egrad = grad;
    end
    if ~isempty(hess)
        problem.ehess = hess;
    end
    problem = ensure_log_map(problem);
    options = struct();
    options.maxiter = params.max_iterations;
    options.verbosity = params.verbosity;
    
    x0 = M.rand();
    [sol, costval, info, ~] = conjugategradient(problem, x0, options);
    conv_history = [info.cost];
end

%-----------------------------------------------------------------------------------------
%% Problem setup function - EXACT SAME FORMULAS AS MANOPT EXAMPLES
%-----------------------------------------------------------------------------------------
function [M, cost, grad, hess, problem_data] = setup_problem(problem_type, dataset_id)
    M = []; cost = []; grad = []; hess = []; problem_data = [];
    
    switch problem_type
        case 'dominant' % Grassmann - EXACTLY as in Manopt dominant_subspace example
            datafile = sprintf('data/dominant-%d.mat', dataset_id);
            if ~exist(datafile, 'file'), return; end
            D = load(datafile); A = D.A; n = size(A,1); p = 3;
            
            % Ensure symmetry as in Manopt examples
            A = (A + A')/2;
            
            M = grassmannfactory(n, p);
            
            % EXACT cost function from Manopt dominant_subspace example
            cost = @(X) -trace(X'*A*X)/2;
            
            % EXACT Euclidean gradient from Manopt
            egrad = @(X) -A*X;
            
            % EXACT Riemannian gradient conversion
            grad = @(X) M.egrad2rgrad(X, egrad(X));
            
            problem_data.A = A;
            
        case 'sdp' % Oblique - EXACTLY as in paper (Semidefinite programs)
            datafile = sprintf('data/SDP-n100-%d.mat', dataset_id);
            if ~exist(datafile, 'file'), return; end
            D = load(datafile); C = D.A; n = size(C,1); 
            
            % The paper's problem is trace(Y'*A*Y). The rank p (column count of Y) 
            % is not specified. Assuming p=n for a full-rank relaxation setup.
            p = n; 
            
            % Ensure symmetry as in Manopt
            C = (C + C')/2;
            
            M = obliquefactory(n, p); % <-- CHANGE: Oblique manifold
            
            % EXACT cost function from paper: min f = trace(Y'*A*Y)
            cost = @(Y) trace(Y'*C*Y); % <-- CHANGE: Matrix variable Y, trace cost
            
            % EXACT Euclidean gradient 
            egrad = @(Y) 2*C*Y; % <-- CHANGE: Matrix variable Y gradient
            
            % EXACT Riemannian gradient conversion
            grad = @(Y) M.egrad2rgrad(Y, egrad(Y));
            
            problem_data.C = C;
            
        case 'svd' % Product of Grassmann manifolds for Truncated SVD
            datafile = sprintf('data/truncat-%d.mat', dataset_id);
            if ~exist(datafile, 'file'), return; end
            D = load(datafile); A = D.A; [m, n] = size(A); p = min(5, min(m, n)); % Ensure p <= min(m,n)
            
            % Define the product manifold: Gr(m, p) x Gr(n, p)
            tuple.U = grassmannfactory(m, p);
            tuple.V = grassmannfactory(n, p);
            M = productmanifold(tuple);
            
            % Define the cost and its derivatives based on truncated_svd
            problem_data.A = A; % Store A in problem_data
            
            % Cost function: f(X) = -.5*norm(U'*A*V, 'fro')^2
            cost = @(X) svd_cost_manopt(X, A);
            
            % Euclidean gradient: egrad.U = -A*V*(A*V)'*U, egrad.V = -A'*U*(A'*U)'*V
            egrad = @(X) svd_egrad_manopt(X, A);
            
            % Euclidean Hessian (needed for 'trustregions')
            ehess = @(X, H) svd_ehess_manopt(X, A, H);
            
            % Riemannian gradient conversion
            grad = @(X) M.egrad2rgrad(X, egrad(X));
            
            % Riemannian Hessian conversion (optional, but good practice if trustregions is used)
            hess = @(X, Xdot) M.ehess2rhess(X, egrad(X), ehess(X, Xdot), Xdot);
            
        case 'procrustes' % Rotations and a reference cloud (Generalized Procrustes)
            datafile = sprintf('data/Procrustes-n3m10N50-%d.mat', dataset_id);
            if ~exist(datafile, 'file'), return; end
                D = load(datafile);
            A_measure = D.A_measure;   % 3×10×N
            Atrue = D.Atrue;           % 3×10
        
            [n, m, N] = size(A_measure);
        
            % --- Define product manifold ---
            tuple.R = rotationsfactory(n, N);
            tuple.A = euclideanfactory(n, m);
            M = productmanifold(tuple);
        
            % --- Define cost, gradient, Hessian ---
            cost = @(X) procrustes_cost(X, A_measure);
            grad = @(X) procrustes_grad(X, A_measure, M);
            hess = @(X, Xdot) procrustes_hess(X, Xdot, A_measure, M);
        
            % --- Package problem data ---
            problem_data.A_measure = A_measure;
            problem_data.Atrue = Atrue;
            problem_data.N = N;
            problem_data.n = n;
            problem_data.m = m;

    
            
        case 'thomson' % Oblique - EXACTLY as in Manopt thomson_problem example
            n_points_list = [50, 75, 100, 125, 150];
            if dataset_id <= length(n_points_list)
                n_points = n_points_list(dataset_id);
            else
                n_points = 50;
            end
            d_ambient = 3;
            M = obliquefactory(d_ambient, n_points);
            
            % EXACT cost function from Manopt thomson_problem example
            cost = @(X) thomson_cost_exact(X);
            
            % EXACT Euclidean gradient from Manopt
            egrad = @(X) thomson_egrad_exact(X);
            
            % EXACT Riemannian gradient conversion
            grad = @(X) M.egrad2rgrad(X, egrad(X));
            
            problem_data.n_points = n_points;
            
        case 'stiffness' % SPD - EXACTLY as in Manopt sparse_PCA example style
            n = 3;
            M = sympositivedefinitefactory(n);
            rng(dataset_id); 
            X_data = randn(10, n); 
            Y_data = randn(10, n);
            
            % EXACT style from Manopt - linear regression on SPD
            cost = @(S) norm(X_data*S - Y_data, 'fro')^2 / 2;
            
            % EXACT Euclidean gradient
            egrad = @(S) X_data'*(X_data*S - Y_data);
            
            % EXACT Riemannian gradient conversion
            grad = @(S) M.egrad2rgrad(S, egrad(S));
            
            problem_data.X_data = X_data;
            problem_data.Y_data = Y_data;
    end
end

%-----------------------------------------------------------------------------------------
%% EXACT Thomson functions from Manopt examples
%-----------------------------------------------------------------------------------------
function f = svd_cost_manopt(X, A)
    U = X.U;
    V = X.V;
    % Cost function: f = -.5*norm(U'*A*V, 'fro')^2
    f = -.5*norm(U'*A*V, 'fro')^2;
end

function g = svd_egrad_manopt(X, A)
    U = X.U;
    V = X.V;
    AV = A*V;
    AtU = A'*U;
    % Euclidean gradient: g.U = -AV*(AV'*U), g.V = -AtU*(AtU'*V)
    g.U = -AV*(AV'*U);
    g.V = -AtU*(AtU'*V);
end

function h = svd_ehess_manopt(X, A, H)
    U = X.U;
    V = X.V;
    Udot = H.U;
    Vdot = H.V;
    AV = A*V;
    AtU = A'*U;
    AVdot = A*Vdot;
    AtUdot = A'*Udot;
    % Euclidean Hessian: h.U = -(AVdot*AV'*U + AV*AVdot'*U + AV*AV'*Udot)
    h.U = -(AVdot*AV'*U + AV*AVdot'*U + AV*AV'*Udot);
    h.V = -(AtUdot*AtU'*V + AtU*AtUdot'*V + AtU*AtU'*Vdot);
end

function f = procrustes_cost(X, A_measure)
    R = X.R;
    A = X.A;
    E = multiprod(R, A) - A_measure;
    N = size(A_measure, 3);
    f = (E(:)' * E(:)) / (2 * N);
end


function egrad = procrustes_grad(X, A_measure, M)
    R = X.R;
    A = X.A;
    N = size(A_measure, 3);

    E = multiprod(R, A) - A_measure;

    egrad.R = multiprod(E, A'/N);
    egrad.A = A - mean(multiprod(multitransp(R), A_measure), 3);

    % Convert Euclidean to Riemannian gradient
    egrad = M.egrad2rgrad(X, egrad);
end

function f = thomson_cost_exact(X)
    % Cost as in paper (page 6): sum_{i<j} 1/(1 - x_i'*x_j) divided by N^2
    [d, N] = size(X);
    I = eye(N);
    S = X' * X;                     % matrix of inner products s_ij = x_i' * x_j
    S = S - diag(diag(S));          % zero diagonal (not strictly needed for triu)
    % upper-triangular 1./(1 - s_ij), exclude diagonal
    M = triu(1 ./ ( max(1e-12, 1 - S) ), 1);   % small eps to avoid division by zero
    f = sum(M(:)) / (N^2);
end

function G = thomson_egrad_exact(X)
    % Euclidean gradient of paper cost: for each i, G(:,i) = sum_{j != i} x_j / (1 - x_i'*x_j)^2
    [d, N] = size(X);
    G = zeros(d, N);
    eps_small = 1e-12;
    for i = 1:N
        xi = X(:, i);
        for j = 1:N
            if j == i, continue; end
            inner = xi' * X(:, j);
            denom = (1 - inner);
            denom = max(denom, eps_small);
            G(:, i) = G(:, i) + X(:, j) / (denom^2);
        end
    end
    % match the cost scaling by N^2
    G = G / (N^2);
end

function problem = ensure_log_map(problem)
    M = problem.M;
    % If M.log missing, add fallback approximate log using projected difference
    if ~isfield(M, 'log') || isempty(M.log)
        % Use projection-based first-order approximate logarithm
        M.log = @(X, Y) M.proj(X, Y - X);
        problem.M = M;
    end
end
%-----------------------------------------------------------------------------------------
%% Remove old cost functions - we only use the exact ones above
%-----------------------------------------------------------------------------------------

%-----------------------------------------------------------------------------------------
%% Results processing functions (unchanged)
%-----------------------------------------------------------------------------------------
function process_results(results, problem_type, methods, method_names)
    fprintf('\n  RESULTS for %s:\n', upper(problem_type));
    fprintf('  Method\t\tMean Cost\tStd Cost\tMean Time\n');
    fprintf('  ------\t\t---------\t--------\t---------\n');
    
    for m = 1:length(methods)
        method = methods{m};
        if ~isfield(results.(problem_type), method)
            continue;
        end
        costs = results.(problem_type).(method).all_costs;
        times = results.(problem_type).(method).all_times;
        
        valid_costs = costs(~isnan(costs));
        valid_times = times(~isnan(times));
        
        if ~isempty(valid_costs)
            mean_cost = mean(valid_costs);
            std_cost = std(valid_costs);
            mean_time = mean(valid_times);
            
            results.(problem_type).(method).mean_cost = mean_cost;
            results.(problem_type).(method).std_cost = std_cost;
            results.(problem_type).(method).mean_time = mean_time;
            
            fprintf('  %-12s\t%.4e\t%.2e\t%.2fs\n', ...
                method_names{m}, mean_cost, std_cost, mean_time);
        else
            fprintf('  %-12s\tN/A\t\tN/A\t\tN/A\n', method_names{m});
        end
    end
end

function create_convergence_plots(results, problem_type, methods, method_names)
    figure('Position', [100, 100, 1200, 800]);
    colors = lines(length(methods));
    
    for m = 1:length(methods)
        method = methods{m};
        convergence_data = results.(problem_type).(method).convergence;
        
        valid_conv = [];
        for i = 1:numel(convergence_data)
            if ~isempty(convergence_data{i}) && isnumeric(convergence_data{i})
                valid_conv = convergence_data{i};
                break;
            elseif ~isempty(convergence_data{i}) && isstruct(convergence_data{i})
                if isfield(convergence_data{i}, 'cost')
                    valid_conv = [convergence_data{i}.cost];
                    break;
                end
            end
        end
        
        if ~isempty(valid_conv)
            plot(1:length(valid_conv), valid_conv, 'Color', colors(m,:), ...
                'LineWidth', 2, 'DisplayName', method_names{m});
            hold on;
        end
    end
    
    xlabel('Iteration');
    ylabel('Cost');
    title(sprintf('Convergence - %s Problem', upper(problem_type)));
    legend('show', 'Location', 'best');
    grid on;
    
    if ~exist('results/plots', 'dir')
        mkdir('results/plots');
    end
    saveas(gcf, sprintf('results/plots/convergence_%s.png', problem_type));
    close(gcf);
end


function create_summary_tables(results, problems, methods, method_names)
    filename = 'results/summary_table.tex';
    if ~exist('results', 'dir'), mkdir('results'); end
    fid = fopen(filename, 'w');
    
    fprintf(fid, '\\begin{table}[h]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\begin{tabular}{l%s}\n', repmat('rr', 1, length(methods)));
    fprintf(fid, '\\hline\n');
    
    % Header Row
    header_line = 'Problem';
    for m = 1:length(methods)
        header_line = [header_line ' & \multicolumn{2}{c}{\bf ' method_names{m} '}'];
    end
    fprintf(fid, '%s \\\\\n', header_line);
    
    % Sub-header (Mean/Std)
    sub_header_line = ' & \small Mean & \small Std';
    sub_header_line = repmat(sub_header_line, 1, length(methods));
    fprintf(fid, '\\hline\n');
    fprintf(fid, ' & %s \\\\\n', sub_header_line(4:end));
    fprintf(fid, '\\hline\n');
    
    for prob_idx = 1:length(problems)
        problem_type = problems{prob_idx};
        fprintf(fid, '%s ', upper(problem_type));
        
        for m = 1:length(methods)
            method = methods{m};
            if isfield(results.(problem_type).(method), 'mean_cost') && ...
               ~isnan(results.(problem_type).(method).mean_cost)
                fprintf(fid, '& %.3f & (%.2e) ', ...
                    results.(problem_type).(method).mean_cost, ...
                    results.(problem_type).(method).std_cost);
            else
                fprintf(fid, '& - & - ');
            end
        end
        fprintf(fid, '\\\\\n');
    end
    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');
    fclose(fid);
end

function create_comparison_plots(results, problems, methods, method_names)
    figure('Position', [100, 100, 1400, 600]);
    
    for prob_idx = 1:length(problems)
        problem_type = problems{prob_idx};
        
        subplot(2, 4, prob_idx);
        means = [];
        stds = [];
        labels = {};
        
        for m = 1:length(methods)
            method = methods{m};
            if isfield(results.(problem_type).(method), 'mean_cost') && ...
               ~isnan(results.(problem_type).(method).mean_cost)
                means(end+1) = results.(problem_type).(method).mean_cost;
                stds(end+1) = results.(problem_type).(method).std_cost;
                labels{end+1} = method_names{m};
            end
        end
        
        if ~isempty(means)
            bar(1:length(means), means);
            hold on;
            errorbar(1:length(means), means, stds, 'k.', 'LineWidth', 1.5);
            set(gca, 'XTick', 1:length(means), 'XTickLabel', labels, 'XTickLabelRotation', 45);
            title(upper(problem_type), 'FontSize', 10);
            ylabel('Mean Cost');
            grid on;
        end
    end
    
    if ~exist('results/plots', 'dir')
        mkdir('results/plots');
    end
    saveas(gcf, 'results/plots/performance_comparison.png');
    close(gcf);
end

function save_comprehensive_results(results)
    if ~exist('results', 'dir')
        mkdir('results');
    end
    save('results/comprehensive_results.mat', 'results', '-v7.3');
    
    problems = fieldnames(results);
        methods = {'rica', 'mDTMA', 'mDE', 'mPSO', 'mSMANN'};

    
    filename = 'results/all_results.csv';
    fid = fopen(filename, 'w');
    fprintf(fid, 'Problem,Method,Mean_Cost,Std_Cost,Mean_Time\n');
    
    for p = 1:length(problems)
        problem = problems{p};
        for m = 1:length(methods)
            method = methods{m};
            if isfield(results.(problem).(method), 'mean_cost') && ...
               ~isnan(results.(problem).(method).mean_cost)
                fprintf(fid, '%s,%s,%.6e,%.6e,%.2f\n', ...
                    problem, method, ...
                    results.(problem).(method).mean_cost, ...
                    results.(problem).(method).std_cost, ...
                    results.(problem).(method).mean_time);
            end
        end
    end
    fclose(fid);
end
%% ------------------------------------------------------------------------
% Fancy Thomson Visualization (replaces old)
% ------------------------------------------------------------------------
%% ------------------------------------------------------------------------
% FANCY THOMSON VISUALIZATIONS
% -------------------------------------------------------------------------
function create_thomson_visualizations(results, problem_type, methods, method_names)
    if ~strcmp(problem_type, 'thomson'), return; end
    fprintf('Generating Thomson visualizations (side & top views)...\n');

    % pick the dataset you want (n = 100 recommended)
    dataset_id = 1;  

    % Ensure output folder
    if ~exist('results/plots', 'dir')
        mkdir('results/plots');
    end

    % === Define shared sphere mesh and colormap ===
    [sx, sy, sz] = sphere(120);
    cmap = parula;

    % === 1️⃣ SIDE VIEW ===
    figure('Color', 'w', 'Position', [100,100,1600,400]);
    t = tiledlayout(1, length(methods), 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, '\bf Thomson problem — Side view (n = 100)', 'FontSize', 14);

    for m = 1:length(methods)
        nexttile;
        sols = results.(problem_type).(methods{m}).all_solutions;
        valid_sol = [];
        for i = 1:numel(sols)
            if ~isempty(sols{i})
                valid_sol = sols{i};
                break;
            end
        end
        if isempty(valid_sol)
            text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
            continue;
        end
        plot_thomson_paper_single(valid_sol, method_names{m}, [sx, sy, sz], cmap, 'side');
    end
    exportgraphics(gcf, 'results/plots/Thomson_side.png', 'Resolution', 400);
    close(gcf);

    % === 2️⃣ TOP VIEW ===
    figure('Color', 'w', 'Position', [100,100,1600,400]);
    t = tiledlayout(1, length(methods), 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, '\bf Thomson problem — Top view (n = 100)', 'FontSize', 14);

    for m = 1:length(methods)
        nexttile;
        sols = results.(problem_type).(methods{m}).all_solutions;
        valid_sol = [];
        for i = 1:numel(sols)
            if ~isempty(sols{i})
                valid_sol = sols{i};
                break;
            end
        end
        if isempty(valid_sol)
            text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
            continue;
        end
        plot_thomson_paper_single(valid_sol, method_names{m}, [sx, sy, sz], cmap, 'top');
    end
    exportgraphics(gcf, 'results/plots/Thomson_top.png', 'Resolution', 400);
    close(gcf);
end

function plot_thomson_paper_style(X, method_name, view_mode)
% Matches the paper's Thomson visualization style (side/top view)
% X: 3×N matrix of points on sphere
% view_mode: 'side' or 'top'

    % Sphere mesh
    [sx, sy, sz] = sphere(100);

    figure('Color', 'w');
    hold on;

    % === Sphere surface ===
    s = surf(sx, sy, sz);
    s.FaceAlpha = 1.0;
    s.FaceColor = 'interp';
    s.EdgeColor = [0.2 0.2 0.2];
    s.LineStyle = '-';
    s.LineWidth = 0.5;
    s.EdgeAlpha = 0.8;
    colormap(parula);
    shading interp;

    % === Thomson points ===
    scatter3(X(1,:), X(2,:), X(3,:), 25, 'filled', ...
        'MarkerFaceColor', [0 0.447 0.741], ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.3);

    % === Lighting and camera ===
    axis equal off;
    camlight headlight;
    lighting gouraud;
    material dull;

    switch lower(view_mode)
        case 'side'
            view(0, 0);   % side view
        case 'top'
            view(0, 90);  % top view
        otherwise
            view(35, 30); % fallback angled view
    end

    % === Aesthetic ===
    title(sprintf('%s (%s view)', method_name, view_mode), ...
        'FontWeight', 'bold', 'FontSize', 11);
    axis([-1 1 -1 1 -1 1]);
end




%% ------------------------------------------------------------------------
% STIFFNESS VISUALIZATIONS (Figs. 7–9)
% -------------------------------------------------------------------------
function create_stiffness_visualizations(results, problem_type, methods, method_names)
    if ~strcmp(problem_type, 'stiffness'), return; end
    fprintf('Generating stiffness visualizations (Figs. 7–9)...\n');
    
    % === Figure 7: Ellipsoid representation of SPD solutions ===
    figure('Position', [100, 100, 1500, 600], 'Color', 'w');
    t = tiledlayout(1, length(methods), 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, '\bf Stiffness Problem — SPD Ellipsoids (Fig. 7)', 'FontSize', 14);
    colors = lines(length(methods));
    
    [x, y, z] = sphere(40);
    for m = 1:length(methods)
        sols = results.(problem_type).(methods{m}).all_solutions;
        valid_sol = [];
        for i = 1:numel(sols)
            if ~isempty(sols{i})
                valid_sol = sols{i};
                break;
            end
        end
        if isempty(valid_sol), continue; end
        
        [V, D] = eig((valid_sol + valid_sol')/2); % Ensure SPD, symmetrize
        radii = sqrt(abs(diag(D)));
        ellipsoid_pts = V * diag(radii) * [x(:)'; y(:)'; z(:)'];
        Xp = reshape(ellipsoid_pts(1,:), size(x));
        Yp = reshape(ellipsoid_pts(2,:), size(x));
        Zp = reshape(ellipsoid_pts(3,:), size(x));
        
        nexttile;
        surf(Xp, Yp, Zp, 'FaceColor', colors(m,:), 'FaceAlpha', 0.7, ...
             'EdgeColor', 'none', 'SpecularStrength', 0.3);
        hold on;
        camlight('headlight'); lighting phong;
        axis equal off; title(method_names{m}, 'FontWeight', 'bold');
    end
    exportgraphics(gcf, 'results/plots/stiffness_fig7_ellipsoids.png', 'ContentType', 'vector');
    close(gcf);

    % === Figure 8: Input-output mapping (X*S vs Y) ===
    figure('Position', [100, 100, 1200, 500], 'Color', 'w');
    t = tiledlayout(1, length(methods), 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, '\bf Stiffness Problem — Linear Mapping X*S ≈ Y (Fig. 8)', 'FontSize', 14);
    
    rng(42);
    X_data = randn(10, 3);
    Y_data = randn(10, 3);
    
    for m = 1:length(methods)
        sols = results.(problem_type).(methods{m}).all_solutions;
        valid_sol = [];
        for i = 1:numel(sols)
            if ~isempty(sols{i})
                valid_sol = sols{i};
                break;
            end
        end
        if isempty(valid_sol), continue; end
        
        Y_pred = X_data * valid_sol;
        nexttile;
        scatter3(Y_data(:,1), Y_data(:,2), Y_data(:,3), 40, 'k', 'filled'); hold on;
        scatter3(Y_pred(:,1), Y_pred(:,2), Y_pred(:,3), 40, colors(m,:), 'filled', 'MarkerEdgeColor', 'k');
        grid on; axis equal;
        xlabel('Y_1'); ylabel('Y_2'); zlabel('Y_3');
        title(method_names{m}, 'FontWeight', 'bold');
        legend({'True $Y$', 'Predicted $X S$'}, 'Interpreter', 'latex', 'FontSize', 9);
    end
    exportgraphics(gcf, 'results/plots/stiffness_fig8_mapping.png', 'ContentType', 'vector');
    close(gcf);

    % === Figure 9: Eigenvalue spectrum of SPD matrices ===
    figure('Position', [100, 100, 1000, 500], 'Color', 'w');
    hold on;
    title('\bf Stiffness Problem — Eigenvalue Spectrum (Fig. 9)', 'FontSize', 14);
    colors = lines(length(methods));
    
    for m = 1:length(methods)
        sols = results.(problem_type).(methods{m}).all_solutions;
        valid_sol = [];
        for i = 1:numel(sols)
            if ~isempty(sols{i})
                valid_sol = sols{i};
                break;
            end
        end
        if isempty(valid_sol), continue; end
        
        eigvals = sort(real(eig((valid_sol + valid_sol')/2)), 'descend');
        plot(eigvals, '-o', 'Color', colors(m,:), 'LineWidth', 2, ...
             'DisplayName', method_names{m});
    end
    xlabel('Index'); ylabel('Eigenvalue'); grid on;
    legend('show', 'Location', 'best');
    exportgraphics(gcf, 'results/plots/stiffness_fig9_eigspectrum.png', 'ContentType', 'vector');
    close(gcf);
end
