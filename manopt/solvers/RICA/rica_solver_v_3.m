function [best_sol, best_cost, conv_history, all_countries_out] = rica_solver_v_3(M, cost_func, opts)
% RICA_SOLVER_V_3  Riemannian Imperialist Competitive Algorithm (improved)
%
% [best_sol, best_cost, conv_history, all_countries_out] =
%     rica_solver_v_3(M, cost_func, opts)
%
% M: Manifold structure (Manopt style)
% cost_func: function handle cost_func(X)
% opts: struct with options -- many defaults are provided
%
% Main features:
%  - Empire assignment strategies: 'random','power','roulette','round_robin'
%  - Empire competition: 'none','classic','tournament','probabilistic','aggressive'
%  - Assimilation modes: 'geodesic','stochastic','momentum'
%  - Adaptive assimilation coefficient and annealed revolution
%  - Levy-style long-jump revolution (escapes local minima)
%  - DE-style recombination (cross-empire mixing)
%  - Aggressive multi-colony transfers
%  - Gradient-free local refinement (randomized tangent search)
%  - Optional parallel colony updates (requires parpool active)
%
% Returns:
%  - best_sol: best imperialist point found
%  - best_cost: cost at best_sol
%  - conv_history: vector of best-cost per iteration (length opts.num_iterations+1)
%  - all_countries_out: cell array of final imperialists + colonies

%% ----------------------- Defaults and option merging -----------------------
if nargin < 3, opts = struct(); end

defaults = struct(...
    'num_countries', 80, ...
    'num_imperialists', max(1, round(0.12 * 80)), ...
    'assimilation_coef', [], ...            % legacy single value (kept if provided)
    'assimilation_coef_base', 0.45, ...     % adaptive base (larger initial moves)
    'assimilation_alpha', 1.0, ...          % annealing exponent (lower -> longer big steps)
    'assimilation_gamma', pi/4, ...
    'revolution_rate0', 0.25, ...           % initial revolution probability
    'revolution_tau', [], ...               % will set based on iterations if empty
    'num_iterations', 800, ...
    'seed', 42, ...
    'verbosity', 1, ...
    'empire_assignment', 'roulette', ...    % 'random','power','roulette','round_robin'
    'empire_competition', 'aggressive', ... % 'none','classic','tournament','probabilistic','aggressive'
    'assimilation_mode', 'geodesic', ...    % 'geodesic','stochastic','momentum'
    'use_momentum', false, ...
    'momentum_beta', 0.85, ...
    'prob_swap', false, ...                 % probabilistic imperialist swap (simulated annealing)
    'swap_T0', 1.0, ...
    'swap_tau', [], ...
    'local_refine', true, ...               % enable gradient-free refinement
    'refine_every', 100, ...                % frequency of refinement (iterations)
    'refine_iters', 10, ...                  % refinement rounds per call
    'refine_dirs', 8, ...                   % tangent directions sampled per round
    'refine_step', 0.12, ...                % initial retraction step size
    'refine_decay', 0.6, ...                % shrink factor if no improvement
    'refine_topk', 3, ...                   % how many top imperialists to refine
    'use_parallel', false, ...              % requires active parpool
    'levy_revolution_prob', 0.15, ...       % chance for long jump inside levy_revolution
    'de_recombine_prob', 0, ...
    'de_F', 0.6, ...
    'de_CR', 0.6, ...
    'aggressive_transfer_count', 2, ...     % number of colonies to transfer in aggressive competition
    'empire_competition_period', 5, ...     % every k iterations run competition
    'max_imperialists', 40 ...              % safety cap
);
%'de_recombine_prob', 0.18, ...          % chance to attempt DE recombination per colony update
% merge defaults
fields = fieldnames(defaults);
for i = 1:numel(fields)
    f = fields{i};
    if ~isfield(opts, f) || isempty(opts.(f))
        opts.(f) = defaults.(f);
    end
end

% sensible derived defaults
if isempty(opts.revolution_tau)
    opts.revolution_tau = max(1, round(opts.num_iterations / 5));
end
if isempty(opts.swap_tau)
    opts.swap_tau = max(1, round(opts.num_iterations / 4));
end

% clamp imperialists
opts.num_imperialists = min(opts.num_imperialists, opts.num_countries - 1);
opts.num_imperialists = max(1, min(opts.num_imperialists, opts.max_imperialists));

% RNG
rng(opts.seed, 'twister');

%% ----------------------- Initialize population -----------------------------
% Countries: cell array of manifold points
countries = cell(1, opts.num_countries);
country_costs = zeros(1, opts.num_countries);
for i = 1:opts.num_countries
    countries{i} = M.rand();
    country_costs(i) = cost_func(countries{i});
end

% sort by cost (ascending)
[country_costs, idx] = sort(country_costs);
countries = countries(idx);

num_imp = opts.num_imperialists;
imperialists = countries(1:num_imp);
imperialist_costs = country_costs(1:num_imp);
colonies = countries(num_imp+1:end);

% Initialize empires: cell per imperialist, containing colonies (cell arrays)
empires = cell(1, num_imp);
num_colonies = length(colonies);

% Empire assignment (initial)
for i = 1:num_colonies
    switch lower(opts.empire_assignment)
        case 'power'
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

% convergence history
conv_history = zeros(1, opts.num_iterations + 1);
conv_history(1) = min(imperialist_costs);

if opts.verbosity >= 1
    fprintf('RICA v3: Initial best cost: %.6f\n', conv_history(1));
end

% momentum storage (one per empire)
emp_momentum = cell(1, num_imp);
for i = 1:num_imp, emp_momentum{i} = []; end

% check parallel pool
use_par = false;
if opts.use_parallel
    try
        pool = gcp('nocreate');
        if ~isempty(pool)
            use_par = true;
        end
    catch
        use_par = false;
    end
end

% build pool each iteration for DE
% prepare optional local refinement (doesn't require opts.problem)
has_refinement = opts.local_refine;

% stats counters
swap_accept_count = 0;
swap_attempt_count = 0;

%% ------------------------- Main optimization loop -------------------------
for iter = 1:opts.num_iterations
    % adaptive coefficients
    frac = (iter - 1) / max(1, opts.num_iterations - 1);
    assimilation_coef = opts.assimilation_coef_base * (1 - frac)^opts.assimilation_alpha;
    if ~isempty(opts.assimilation_coef) % legacy override if user set scalar assimilation_coef
        assimilation_coef = opts.assimilation_coef;
    end
    revolution_rate = opts.revolution_rate0 * exp(- (iter - 1) / opts.revolution_tau);
    swap_T = opts.swap_T0 * exp(- (iter - 1) / opts.swap_tau);

    % build pool for DE recombination (cells)
    pop_pool = [imperialists, horzcat(empires{:})];

    % iterate over empires
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

        % Decide whether to use parallel updates for this empire
        if use_par && N_col > 1
            % parfor-friendly containers
            tmp_new = cell(1, N_col);
            tmp_costs = zeros(1, N_col);

            parfor col_idx = 1:N_col
                local_col = empire{col_idx};
                old_col_cost = cost_func(local_col);

                % Choose assimilation mechanism
                switch lower(opts.assimilation_mode)
                    case 'geodesic'
                        candidate = apply_geodesic_move(M, local_col, imp, assimilation_coef, opts);
                    case 'stochastic'
                        candidate = apply_geodesic_move(M, local_col, imp, assimilation_coef, opts);
                        if rand < 0.3
                            n = M.randvec(candidate);
                            candidate = M.retr(candidate, M.lincomb(candidate, 0.05, n));
                        end
                    case 'momentum'
                        % In parfor, we cannot use shared emp_momentum reliably. Use small bias.
                        candidate = apply_geodesic_move(M, local_col, imp, assimilation_coef, opts);
                    otherwise
                        candidate = apply_geodesic_move(M, local_col, imp, assimilation_coef, opts);
                end

                % DE recombination (chance)
                if ~isempty(pop_pool) && rand < opts.de_recombine_prob && numel(pop_pool) >= 3
                    % pick two random donors different from this colony (best-effort)
                    indices = randperm(numel(pop_pool), 2);
                    Xa = pop_pool{indices(1)};
                    Xb = pop_pool{indices(2)};
                    try
                        candidate = de_recombine(M, candidate, Xa, Xb, opts.de_F, opts.de_CR);
                    catch
                        % ignore recombination failure
                    end
                end

                % Revolution: Levy-style long jumps (annealed chance overall)
                if rand < revolution_rate
                    candidate = levy_revolution(M, candidate, 0.6, opts.levy_revolution_prob);
                end

                % Evaluate
                new_cost_local = cost_func(candidate);

                % Acceptance greedy
                if new_cost_local < old_col_cost
                    tmp_new{col_idx} = candidate;
                    tmp_costs(col_idx) = new_cost_local;
                else
                    tmp_new{col_idx} = local_col;
                    tmp_costs(col_idx) = old_col_cost;
                end
            end

            % copy back
            for col_idx = 1:N_col
                new_empire{col_idx} = tmp_new{col_idx};
                new_empire_costs(col_idx) = tmp_costs(col_idx);
            end

        else
            % Serial update (supports per-empire momentum)
            for col_idx = 1:N_col
                col = empire{col_idx};
                old_col_cost = cost_func(col);

                % compute log map safely
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
                    nr = M.norm(col, rand_dir_ortho);

                    if nr > 1e-12
                        u_dev = M.lincomb(col, 1/nr, rand_dir_ortho);
                        theta = (2*rand - 1) * opts.assimilation_gamma;
                        dev_scale = norm_v * tan(theta);
                        base_tangent = M.lincomb(col, assimilation_coef, v_log, dev_scale, u_dev);
                    else
                        base_tangent = M.lincomb(col, assimilation_coef, v_log);
                    end
                else
                    base_tangent = M.zerovec(col);
                end

                % momentum blending
                if opts.use_momentum
                    if isempty(emp_momentum{imp_idx})
                        emp_momentum{imp_idx} = M.zerovec(col);
                    end
                    base_tangent = M.lincomb(col, 1, base_tangent, opts.momentum_beta, emp_momentum{imp_idx});
                    emp_momentum{imp_idx} = M.lincomb(col, 1, base_tangent);
                end

                % move (retraction)
                candidate = M.retr(col, base_tangent);

                % DE recombination (probabilistic)
                if ~isempty(pop_pool) && rand < opts.de_recombine_prob && numel(pop_pool) >= 3
                    % select two donors not equal to candidate (best effort)
                    donorsIdx = randperm(numel(pop_pool), 2);
                    Xa = pop_pool{donorsIdx(1)};
                    Xb = pop_pool{donorsIdx(2)};
                    try
                        candidate = de_recombine(M, candidate, Xa, Xb, opts.de_F, opts.de_CR);
                    catch
                        % ignore
                    end
                end

                % Revolution (Levy)
                if rand < revolution_rate
                    candidate = levy_revolution(M, candidate, 0.6, opts.levy_revolution_prob);
                end

                % Evaluate
                new_col_cost = cost_func(candidate);

                % Acceptance
                if new_col_cost < old_col_cost
                    new_empire{col_idx} = candidate;
                    new_empire_costs(col_idx) = new_col_cost;
                else
                    new_empire{col_idx} = col;
                    new_empire_costs(col_idx) = old_col_cost;
                end
            end
        end

        % update empire
        empires{imp_idx} = new_empire;

        % Imperialist swap (either greedy or probabilistic)
        [best_col_cost, best_col_idx] = min(new_empire_costs);
        if ~isempty(best_col_cost) && best_col_cost < imp_cost
            if opts.prob_swap
                swap_attempt_count = swap_attempt_count + 1;
                delta = imp_cost - best_col_cost;
                p_accept = 1 - exp(-max(0, delta) / (swap_T + eps));
                if rand < p_accept
                    % perform swap
                    tmp = imperialists{imp_idx};
                    imperialists{imp_idx} = empires{imp_idx}{best_col_idx};
                    empires{imp_idx}{best_col_idx} = tmp;
                    imperialist_costs(imp_idx) = best_col_cost;
                    new_empire_costs(best_col_idx) = imp_cost;
                    swap_accept_count = swap_accept_count + 1;
                end
            else
                tmp = imperialists{imp_idx};
                imperialists{imp_idx} = empires{imp_idx}{best_col_idx};
                empires{imp_idx}{best_col_idx} = tmp;
                imperialist_costs(imp_idx) = best_col_cost;
                new_empire_costs(best_col_idx) = imp_cost;
            end
        end
    end % end per-empire loop

    % Empire competition (periodic)
    if mod(iter, opts.empire_competition_period) == 0
        switch lower(opts.empire_competition)
            case 'classic'
                empires = classic_competition(empires, imperialists, imperialist_costs, M, cost_func);
            case 'tournament'
                empires = tournament_competition(empires, imperialists, imperialist_costs, cost_func);
            case 'probabilistic'
                empires = probabilistic_competition(empires, imperialists, imperialist_costs, cost_func);
            case 'aggressive'
                empires = aggressive_competition(empires, imperialists, imperialist_costs, cost_func, opts.aggressive_transfer_count);
            otherwise
                % none
        end
    end

    % ---------------- Gradient-free local refinement ----------------
    if has_refinement && mod(iter, opts.refine_every) == 0
        try
            % refine top-k imperialists (lowest costs)
            [~, rank_idx] = sort(imperialist_costs);
            topk = min(opts.refine_topk, numel(rank_idx));
            for r = 1:topk
                ii = rank_idx(r);
                x0 = imperialists{ii};
                c0 = imperialist_costs(ii);
                [x_ref, c_ref] = refine_random_search(M, x0, cost_func, ...
                    'max_rounds', opts.refine_iters, ...
                    'num_dirs', opts.refine_dirs, ...
                    'step', opts.refine_step, ...
                    'decay', opts.refine_decay);
                if c_ref < c0
                    imperialists{ii} = x_ref;
                    imperialist_costs(ii) = c_ref;
                end
            end
        catch ME
            % don't fail the whole run on refinement errors
            if opts.verbosity >= 2
                fprintf('Refinement error: %s\n', ME.message);
            end
        end
    end

    % record convergence
    conv_history(iter + 1) = min(imperialist_costs);

    % logging
    if opts.verbosity >= 1 && mod(iter, 20) == 0
        fprintf('RICA v3: Iter %d/%d, Best cost: %.6e\n', iter, opts.num_iterations, conv_history(iter + 1));
    end
end % main iterations

% final best
[best_cost, best_idx] = min(imperialist_costs);
best_sol = imperialists{best_idx};
all_countries_out = [imperialists, horzcat(empires{:})];

% print summary
if opts.verbosity >= 1
    fprintf('RICA v3: Final best cost: %.6f after %d iterations\n', best_cost, opts.num_iterations);
    if swap_attempt_count > 0
        fprintf('  Swap acceptance rate: %.2f%% (%d/%d)\n', 100*swap_accept_count/max(1,swap_attempt_count), swap_accept_count, swap_attempt_count);
    end
end

end % end main function

%% ---------------------- Helper functions ----------------------

function Xnew = apply_geodesic_move(M, X, imp, assimilation_coef, opts)
% Compute a combined geodesic-like move from X towards imp with small orthogonal dev
    try
        v_log = M.log(X, imp);
    catch
        v_log = M.lincomb(X, 1, imp, -1, X);
    end
    norm_v = M.norm(X, v_log);
    if norm_v > 1e-12
        rand_dir = M.randvec(X);
        inner_prod = M.inner(X, rand_dir, v_log);
        proj_val = inner_prod / (norm_v^2);
        rand_dir_ortho = M.lincomb(X, 1, rand_dir, -proj_val, v_log);
        norm_rand_ortho = M.norm(X, rand_dir_ortho);
        if norm_rand_ortho > 1e-12
            u_dev = M.lincomb(X, 1/norm_rand_ortho, rand_dir_ortho);
            theta = (2*rand - 1) * opts.assimilation_gamma;
            dev_scale = norm_v * tan(theta);
            tangent_move = M.lincomb(X, assimilation_coef, v_log, dev_scale, u_dev);
        else
            tangent_move = M.lincomb(X, assimilation_coef, v_log);
        end
    else
        tangent_move = M.zerovec(X);
    end
    Xnew = M.retr(X, tangent_move);
end

function Xnew = levy_revolution(M, X, scale, levy_prob)
% Levy-style revolution: small jitter + occasional heavy-tailed jump
% levy_prob: probability of doing the long jump component (default from opts)
    if nargin < 4 || isempty(levy_prob), levy_prob = 0.15; end
    % small jitter
    noise = M.randvec(X);
    nrm = M.norm(X, noise) + eps;
    step_small = M.lincomb(X, 0.05 / nrm, noise);
    Y = M.retr(X, step_small);

    % heavy-tailed jump
    if rand < levy_prob
        s = abs(randn() * 0.5 + randn() * 1.2); % heavy-ish tail
        jump = M.randvec(Y);
        jump_norm = M.norm(Y, jump) + eps;
        jump_step = M.lincomb(Y, scale * s / jump_norm, jump);
        Xnew = M.retr(Y, jump_step);
    else
        Xnew = Y;
    end
end

function Xnew = de_recombine(M, target, Xa, Xb, F, CR)
% Differential evolution style recombination on manifold (approximate)
% target: point being mutated (as base)
% Xa, Xb: donor points
% F: mutation factor
% CR: crossover probability
    if nargin < 5 || isempty(F), F = 0.6; end
    if nargin < 6 || isempty(CR), CR = 0.6; end

    try
        v1 = M.log(target, Xa);
    catch
        v1 = M.lincomb(target, 1, Xa, -1, target);
    end
    try
        v2 = M.log(target, Xb);
    catch
        v2 = M.lincomb(target, 1, Xb, -1, target);
    end

    % mutation: v1 + F*(v1 - v2) as heuristic in tangent at target
    diff = M.lincomb(target, 1, v1, -1, v2);
    mut = M.lincomb(target, 1, v1, F, diff);

    % crossover decision
    if rand < CR
        Xmut = M.retr(target, mut);
    else
        Xmut = target;
    end

    % small stabilization retraction
    stabil = M.randvec(Xmut);
    Xnew = M.retr(Xmut, M.lincomb(Xmut, 0.01, stabil));
end

function empires = classic_competition(empires, imperialists, imp_costs, M, cost_func)
% Classic ICA competition: compute total cost (imperialist + mean colony cost) and move a colony
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
        rand_idx = randi(length(empires{weakest}));
        losing_col = empires{weakest}{rand_idx};
        empires{strongest} = [empires{strongest}, {losing_col}];
        empires{weakest}(rand_idx) = [];
    end
end

function empires = tournament_competition(empires, imperialists, imp_costs, cost_func)
% Tournament-style: choose k empires, winner takes one colony from loser
    num_imp = length(imperialists);
    k = max(2, ceil(num_imp/4));
    chosen = randperm(num_imp, k);
    [~, order] = sort(imp_costs(chosen));
    winner = chosen(order(1));
    loser = chosen(order(end));
    if ~isempty(empires{loser})
        rand_idx = randi(length(empires{loser}));
        losing_col = empires{loser}{rand_idx};
        empires{winner} = [empires{winner}, {losing_col}];
        empires{loser}(rand_idx) = [];
    end
end

function empires = probabilistic_competition(empires, imperialists, imp_costs, cost_func)
% Probabilistic: stronger empires probabilistically take colonies from weaker ones
    num_imp = length(imperialists);
    vals = exp(-imp_costs / (max(imp_costs) + eps));
    probs = vals / sum(vals);
    strongest = randsample(num_imp, 1, true, probs);
    weakest = randsample(num_imp, 1, true, 1 - probs);
    if ~isempty(empires{weakest})
        rand_idx = randi(length(empires{weakest}));
        losing_col = empires{weakest}{rand_idx};
        empires{strongest} = [empires{strongest}, {losing_col}];
        empires{weakest}(rand_idx) = [];
    end
end

function empires = aggressive_competition(empires, imperialists, imp_costs, cost_func, num_transfer)
% Aggressive: compute total costs and transfer up to num_transfer colonies from weakest to strongest
    if nargin < 5 || isempty(num_transfer), num_transfer = 2; end
    num_imp = length(imperialists);
    total_costs = zeros(1, num_imp);
    for i = 1:num_imp
        if isempty(empires{i})
            total_costs(i) = imp_costs(i);
        else
            cc = zeros(1, length(empires{i}));
            for j = 1:length(empires{i})
                cc(j) = cost_func(empires{i}{j});
            end
            total_costs(i) = imp_costs(i) + mean(cc);
        end
    end
    [~, weakest] = max(total_costs);
    [~, strongest] = min(total_costs);

    if ~isempty(empires{weakest})
        cnt = min(num_transfer, length(empires{weakest}));
        idxs = randperm(length(empires{weakest}), cnt);
        for k = 1:cnt
            empires{strongest} = [empires{strongest}, empires{weakest}(idxs(k))];
        end
        idxs = sort(idxs, 'descend');
        for k = 1:length(idxs)
            empires{weakest}(idxs(k)) = [];
        end
    end
end

%% ---------------- Gradient-free refinement helper ----------------
function [x_best, cost_best] = refine_random_search(M, x0, cost_func, varargin)
% Gradient-free refinement: randomized search on tangent directions
% Options (passed as name/value):
%   'max_rounds' (default 6) : number of refinement rounds
%   'num_dirs'   (default 8) : number of directions per round
%   'step'       (default 0.12): initial step size (retraction scale)
%   'decay'      (default 0.6) : shrink factor when no improvement
%
    p = inputParser;
    addParameter(p, 'max_rounds', 6);
    addParameter(p, 'num_dirs', 8);
    addParameter(p, 'step', 0.12);
    addParameter(p, 'decay', 0.6);
    parse(p, varargin{:});
    max_rounds = p.Results.max_rounds;
    num_dirs = p.Results.num_dirs;
    step = p.Results.step;
    decay = p.Results.decay;

    x_best = x0;
    cost_best = cost_func(x_best);

    for r = 1:max_rounds
        improved = false;
        local_best = x_best;
        local_cost = cost_best;

        % sample directions
        for d = 1:num_dirs
            v = M.randvec(x_best);
            % normalize and scale
            vnorm = M.norm(x_best, v) + eps;
            v_scaled = M.lincomb(x_best, step / vnorm, v);

            x_candidate = M.retr(x_best, v_scaled);
            try
                c_candidate = cost_func(x_candidate);
            catch
                c_candidate = inf;
            end

            if c_candidate < local_cost
                local_cost = c_candidate;
                local_best = x_candidate;
                improved = true;
            end
        end

        if improved
            % accept best local move and slightly increase step
            x_best = local_best;
            cost_best = local_cost;
            step = min(0.5, step * 1.15);
        else
            % no improvement: shrink step and continue
            step = step * decay;
            if step < 1e-6
                break;
            end
        end
    end
end

% end of file
