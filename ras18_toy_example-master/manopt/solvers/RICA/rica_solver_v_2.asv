function [best_sol, best_cost, conv_history, all_countries_out] = rica_solver_v_2(M, cost_func, opts)
% Riemannian Imperialist Competitive Algorithm (RICA)
% Enhanced with flexible selection mechanisms:
%   1. Empire Assignment
%   2. Empire Competition
%   3. Colony Update Mechanism
%
% M:        Manifold structure (Manopt-style)
% cost_func:Cost function handle
% opts:     struct with algorithm parameters

    % --- Default parameters ---
    if nargin < 3
        opts = struct();
    end
    
    default_opts = struct(...
        'num_countries', 80, ...
        'num_imperialists', 10, ...
        'assimilation_coef', 0.1, ...
        'revolution_rate', 0.7, ...
        'assimilation_gamma', pi/4, ...
        'num_iterations', 800, ...
        'seed', 42, ...
        'verbosity', 1, ...
        'empire_assignment', 'random', ...     % 'random', 'power', 'roulette', 'round_robin'
        'empire_competition', 'classic', ...   % 'none', 'classic', 'tournament', 'probabilistic'
        'assimilation_mode', 'geodesic' ...    % 'geodesic', 'stochastic', 'momentum'
    );
    
    opt_fields = fieldnames(default_opts);
    for i = 1:length(opt_fields)
        field = opt_fields{i};
        if ~isfield(opts, field)
            opts.(field) = default_opts.(field);
        end
    end
    
    rng(opts.seed, 'twister');

    % --- 1. Initialize countries ---
    countries = cell(1, opts.num_countries);
    country_costs = zeros(1, opts.num_countries);
    for i = 1:opts.num_countries
        countries{i} = M.rand();
        country_costs(i) = cost_func(countries{i});
    end

    % --- 2. Sort and assign imperialists and colonies ---
    [country_costs, idx] = sort(country_costs);
    countries = countries(idx);
    
    num_imp = opts.num_imperialists;
    imperialists = countries(1:num_imp);
    imperialist_costs = country_costs(1:num_imp);
    colonies = countries(num_imp+1:end);
    
    empires = cell(1, num_imp);
    num_colonies = length(colonies);
    
    % ---- Empire Assignment Mechanism ----
    for i = 1:num_colonies
        switch lower(opts.empire_assignment)
            case 'power' % proportional to inverse cost
                inv_cost = max(imperialist_costs) - imperialist_costs + eps;
                probs = inv_cost / sum(inv_cost);
                empire_idx = randsample(num_imp, 1, true, probs);
            case 'roulette'
                fitness = 1 ./ (imperialist_costs + eps);
                probs = fitness / sum(fitness);
                empire_idx = randsample(num_imp, 1, true, probs);
            case 'round_robin'
                empire_idx = mod(i-1, num_imp) + 1;
            otherwise % 'random'
                empire_idx = randi(num_imp);
        end
        empires{empire_idx} = [empires{empire_idx}, colonies(i)];
    end

    % --- 3. Initialize convergence history ---
    conv_history = zeros(1, opts.num_iterations+1);
    conv_history(1) = min(imperialist_costs);

    if opts.verbosity >= 1
        fprintf('RICA: Initial best cost: %.6f\n', conv_history(1));
    end

    % --- 4. Main Optimization Loop ---
    for iter = 1:opts.num_iterations
        for imp_idx = 1:num_imp
            imp = imperialists{imp_idx};
            imp_cost = imperialist_costs(imp_idx);
            empire = empires{imp_idx};
            if isempty(empire)
                continue;
            end

            N_col = length(empire);
            new_empire = cell(1, N_col);
            new_empire_costs = zeros(1, N_col);
            
            for col_idx = 1:N_col
                col = empire{col_idx};
                old_col_cost = cost_func(col);

                % ---- Assimilation (Colony Update Mechanism) ----
                switch lower(opts.assimilation_mode)
                    case 'geodesic' % (current)
                        [new_col, new_col_cost] = geodesic_assimilation(M, col, imp, cost_func, opts);
                    case 'stochastic'
                        [new_col, new_col_cost] = stochastic_assimilation(M, col, imp, cost_func, opts);
                    case 'momentum'
                        [new_col, new_col_cost] = momentum_assimilation(M, col, imp, cost_func, opts, iter);
                    otherwise
                        [new_col, new_col_cost] = geodesic_assimilation(M, col, imp, cost_func, opts);
                end

                % ---- Acceptance ----
                if new_col_cost < old_col_cost
                    new_empire{col_idx} = new_col;
                    new_empire_costs(col_idx) = new_col_cost;
                else
                    new_empire{col_idx} = col;
                    new_empire_costs(col_idx) = old_col_cost;
                end
            end

            empires{imp_idx} = new_empire;

            % ---- Imperialist Swap ----
            [best_col_cost, best_col_idx] = min(new_empire_costs);
            if best_col_cost < imp_cost
                tmp = imperialists{imp_idx};
                imperialists{imp_idx} = empires{imp_idx}{best_col_idx};
                empires{imp_idx}{best_col_idx} = tmp;
                imperialist_costs(imp_idx) = best_col_cost;
                new_empire_costs(best_col_idx) = imp_cost;
            end
        end

        % ---- Empire Competition Mechanism ----
        switch lower(opts.empire_competition)
            case 'classic'
                empires = classic_competition(empires, imperialists, imperialist_costs, M, cost_func);
            case 'tournament'
                empires = tournament_competition(empires, imperialists, imperialist_costs);
            case 'probabilistic'
                empires = probabilistic_competition(empires, imperialists, imperialist_costs);
            otherwise
                % 'none' — do nothing
        end

        conv_history(iter+1) = min(imperialist_costs);

        if opts.verbosity >= 1 && mod(iter, 20) == 0
            fprintf('RICA: Iter %d/%d, Best cost: %.6e\n', iter, opts.num_iterations, conv_history(iter+1));
        end
    end

    % --- 5. Best Result ---
    [best_cost, best_idx] = min(imperialist_costs);
    best_sol = imperialists{best_idx};
    all_countries_out = [imperialists, horzcat(empires{:})];

    if opts.verbosity >= 1
        fprintf('RICA: Final best cost: %.6f after %d iterations\n', best_cost, opts.num_iterations);
    end
end


%% ===== Helper Functions =====

function [new_col, new_cost] = geodesic_assimilation(M, col, imp, cost_func, opts)
    try
        v_log = M.log(col, imp);
    catch
        v_log = M.lincomb(col, 1, imp, -1, col);
    end
    norm_v = M.norm(col, v_log);
    if norm_v > 1e-12
        rand_dir = M.randvec(col);
        inner_prod = M.inner(col, rand_dir, v_log);
        proj_val = inner_prod / (norm_v^2);
        rand_dir_ortho = M.lincomb(col, 1, rand_dir, -proj_val, v_log);
        norm_rand_ortho = M.norm(col, rand_dir_ortho);
        if norm_rand_ortho > 1e-12
            u_dev = M.lincomb(col, 1/norm_rand_ortho, rand_dir_ortho);
            theta = (2*rand - 1) * opts.assimilation_gamma;
            dev_scale = norm_v * tan(theta);
            tangent_move = M.lincomb(col, opts.assimilation_coef, v_log, dev_scale, u_dev);
        else
            tangent_move = M.lincomb(col, opts.assimilation_coef, v_log);
        end
    else
        tangent_move = M.zerovec(col);
    end
    new_col = M.retr(col, tangent_move);
    if rand < opts.revolution_rate
        rev_vec = M.randvec(new_col);
        rev_scale = 0.1 / (M.norm(new_col, rev_vec) + eps);
        new_col = M.retr(new_col, M.lincomb(new_col, rev_scale, rev_vec));
    end
    new_cost = cost_func(new_col);
end

function [new_col, new_cost] = stochastic_assimilation(M, col, imp, cost_func, opts)
    [new_col, new_cost] = geodesic_assimilation(M, col, imp, cost_func, opts);
    if rand < 0.3 % random jitter in tangent space
        noise = M.randvec(new_col);
        new_col = M.retr(new_col, M.lincomb(new_col, 0.05, noise));
        new_cost = cost_func(new_col);
    end
end

function [new_col, new_cost] = momentum_assimilation(M, col, imp, cost_func, opts, iter)
    persistent prev_tangent
    if isempty(prev_tangent)
        prev_tangent = M.zerovec(col);
    end
    [base_col, base_cost] = geodesic_assimilation(M, col, imp, cost_func, opts);
    v_log = M.log(col, imp);
    momentum = 0.9 * prev_tangent + 0.1 * v_log;
    new_col = M.retr(base_col, M.lincomb(base_col, 0.1, momentum));
    new_cost = cost_func(new_col);
    prev_tangent = momentum;
end

function empires = classic_competition(empires, imperialists, imp_costs, M, cost_func)
    num_imp = length(imperialists);
    total_costs = zeros(1, num_imp);
    for i = 1:num_imp
        if isempty(empires{i})
            total_costs(i) = imp_costs(i);
        else
            col_costs = zeros(1, length(empires{i}));
            for j = 1:length(empires{i})
                col_costs(j) = cost_func(empires{i}{j});
            end
            total_costs(i) = imp_costs(i) + mean(col_costs);
        end
    end
    [~, weakest] = max(total_costs);
    [~, strongest] = min(total_costs);
    if ~isempty(empires{weakest})
        losing_col = empires{weakest}{randi(length(empires{weakest}))};
        empires{strongest} = [empires{strongest}, {losing_col}];
        empires{weakest}(randi(length(empires{weakest}))) = [];
    end
end

function empires = tournament_competition(empires, imperialists, imp_costs)
    num_imp = length(imperialists);
    k = max(2, ceil(num_imp/4));
    chosen = randperm(num_imp, k);
    [~, idx] = sort(imp_costs(chosen));
    winner = chosen(1);
    loser = chosen(end);
    if ~isempty(empires{loser})
        losing_col = empires{loser}{randi(length(empires{loser}))};
        empires{winner} = [empires{winner}, {losing_col}];
        empires{loser}(randi(length(empires{loser}))) = [];
    end
end

function empires = probabilistic_competition(empires, imperialists, imp_costs)
    num_imp = length(imperialists);
    probs = exp(-imp_costs / max(imp_costs));
    probs = probs / sum(probs);
    strongest = randsample(num_imp, 1, true, probs);
    weakest = randsample(num_imp, 1, true, 1 - probs);
    if ~isempty(empires{weakest})
        losing_col = empires{weakest}{randi(length(empires{weakest}))};
        empires{strongest} = [empires{strongest}, {losing_col}];
        empires{weakest}(randi(length(empires{weakest}))) = [];
    end
end
