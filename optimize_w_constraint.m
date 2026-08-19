function Weight = optimize_w_constraint(data, f, Weight, Para)
% OPTIMIZE_W_CONSTRAINT: Constrained weight optimization with L1 ball
%
% ============================================================================
% IMPORTANT: This function uses PROJECTED GRADIENT DESCENT with L1 constraint
% ============================================================================
%
% OPTIMIZATION PROBLEM:
%   minimize: L(W) = λ₁·L_smooth(W)
%   subject to: W ≥ 0, ||W||₁ ≤ τ
%
% HOW L1 CONSTRAINT IS ENFORCED:
%   Step 1: Gradient step on smooth objective
%           W_temp = W - α·∇L_smooth(W)
%
%   Step 2: Project onto feasible set {W ≥ 0, ||W||₁ ≤ τ}
%           W_new = project_L1_ball_nonneg(W_temp, τ)
%
% Key advantages over L1 penalty:
% 1. τ has clear interpretation: max sum of weights
% 2. GUARANTEES ||W||₁ ≤ τ (cannot collapse all weights to zero)
% 3. More stable: weights always remain in feasible set
% 4. Easier to tune: τ has natural scale (e.g., τ = 100 means max sum = 100)
%
% Comparison with L1 penalty version:
% - L1 PENALTY:  min L(W) + λ₂·||W||₁     → weights can collapse to zero
% - L1 CONSTRAINT: min L(W) s.t. ||W||₁ ≤ τ → weights bounded away from zero
%
% Inputs:
%   data   - [d x N] feature matrix
%   f      - [N x 1] instance labels
%   Weight - [d x 1] initial weights (must be >= 0)
%   Para   - parameter struct with fields:
%            .Weight_iter: number of iterations (or [min, max] for adaptive)
%            .lambda1: smoothness penalty weight
%            .tau: L1 constraint (max L1 norm, replaces lambda2)
%            .sigma: kernel bandwidth
%            .truncate_k: number of nearest neighbors (default: 25)
%            .distance_metric: 'cityblock' or 'squaredeuclidean'
%
% Outputs:
%   Weight - [d x 1] optimized weights satisfying ||Weight||₁ ≤ τ
%
% Reference: Duchi et al. (2008). "Efficient Projections onto the L1-Ball"

%% Initialize
Weight0 = Weight;
N_iter = Para.Weight_iter;
lambda1 = Para.lambda1;
tau = Para.tau;  % L1 constraint (replaces lambda2)
sigma = Para.sigma;
[dim, N_patterns] = size(data);

if ~isfield(Para, 'truncate_k')
    truncate_k = 25;
else
    truncate_k = Para.truncate_k;
end

if ~isfield(Para, 'distance_metric')
    distance_metric = 'cityblock';
else
    distance_metric = Para.distance_metric;
end

% Ensure Weight is non-negative and satisfies constraint
Weight = max(Weight, 0);
if sum(Weight) > tau
    % Project initial weights onto L1 ball
    Weight = project_L1_ball_nonneg(Weight, tau);
end

fprintf('  Initial: ||W||₁ = %.6f, nnz = %d/%d\n', ...
        sum(Weight), sum(Weight/max(Weight) >= 0.01), dim);

%% Compute initial cost
% Note: no lambda2 term in cost (constraint is handled by projection)
Para_eval = Para;
Para_eval.lambda2 = 0;  % No L1 penalty
cost = loss_weight_constraint(data, f, Weight, Para_eval);

%% Optimization loop
iter = 0;
progress = 1;

while iter < N_iter && progress >= 1e-6
    iter = iter + 1;

    %% Build symmetric k-nearest neighbor graph
    % Compute pairwise distances in weighted space
    data_weighted = diag(sqrt(Weight)) * data;
    dist_weighted = pdist2(data_weighted', data_weighted', 'squaredeuclidean');

    % Find k nearest neighbors for each point
    neighbors = cell(N_patterns, 1);
    for i = 1:N_patterns
        [~, idx] = mink(dist_weighted(i, :), min(truncate_k, N_patterns));
        neighbors{i} = idx;
    end

    % Build symmetric neighbor set (i~j if i∈N(j) OR j∈N(i))
    neighbor_pairs = [];
    for i = 1:N_patterns
        for j = neighbors{i}
            if i < j  % Only add each pair once
                neighbor_pairs = [neighbor_pairs; i, j];
            end
        end
    end

    % Ensure symmetry
    symmetric_pairs = unique(sort(neighbor_pairs, 2), 'rows');

    %% ========================================================================
    %% STEP 1: Compute gradient of SMOOTH objective (no L1 term!)
    %% ========================================================================
    % We only minimize the smooth part: λ₁·L_smooth(W)
    % The L1 constraint is NOT in the objective - it's handled by PROJECTION
    %
    % Gradient of smooth term:
    %   ∂L_smooth/∂W = λ₁ * Σ_pairs K(i,j) * (f_i - f_j)² * (1/σ) * Z_ij

    grad_smooth = zeros(dim, 1);

    % Check if we have valid neighbor pairs
    if size(symmetric_pairs, 1) == 0
        fprintf('    WARNING: No symmetric pairs found! Weights cannot be optimized.\n');
        break;
    end

    for pair_idx = 1:size(symmetric_pairs, 1)
        i = symmetric_pairs(pair_idx, 1);
        j = symmetric_pairs(pair_idx, 2);

        % Label difference squared
        Fij = (f(i) - f(j))^2;

        % Feature differences
        if strcmp(distance_metric, 'cityblock')
            Zij = abs(data(:, i) - data(:, j));
        else  % squaredeuclidean
            Zij = (data(:, i) - data(:, j)).^2;
        end

        % Weighted distance
        dist_ij = Weight' * Zij;

        % Kernel value
        K_ij = exp(-dist_ij / sigma);

        % Gradient contribution (negative for minimization)
        grad_smooth = grad_smooth - Fij * K_ij * (1/sigma) * Zij;
    end

    % Normalize by number of patterns
    grad_smooth = lambda1 * grad_smooth / N_patterns;

    %% Diagnostic: Check gradient magnitude
    grad_norm = norm(grad_smooth);
    

    %% ========================================================================
    %% FAST LINE SEARCH: Armijo backtracking with projection
    %% ========================================================================
    % SPEED OPTIMIZATION: Use backtracking instead of fminbnd
    % - fminbnd evaluates ~10-20 step sizes (slow!)
    % - Armijo backtracking evaluates ~1-5 step sizes (fast!)
    %
    % Armijo rule (sufficient decrease condition):
    %   f(x + α·d) ≤ f(x) + c₁·α·∇f'·d
    % where c₁ = 0.0001 (very loose condition for faster acceptance)

    % Armijo parameters
    alpha = 1.0;              % Initial step size (optimistic)
    beta = 0.5;               % Backtracking factor
    c1 = 1e-4;                % Armijo constant (loose for speed)
    max_backtrack = 20;       % Max backtracking iterations

    % Prepare parameter struct for loss evaluation
    Para_ls = Para_eval;
    Para_ls.symmetric_pairs = symmetric_pairs;

    % Directional derivative: ∇f' · (-grad) = -||grad||²
    % (Negative because descent direction is -grad)
    directional_deriv = -sum(grad_smooth.^2);

    % Armijo backtracking loop
    for bt_iter = 1:max_backtrack
        % Projected gradient step
        Weight_temp = Weight - alpha * grad_smooth;
        Weight_new = project_L1_ball_nonneg(Weight_temp, tau);

        % Evaluate cost at projected point
        newcost = loss_weight_constraint(data, f, Weight_new, Para_ls);

        % Check Armijo condition (sufficient decrease)
        armijo_threshold = cost + c1 * alpha * directional_deriv;
        if newcost <= armijo_threshold
            % Accept step size
            break;
        end

        % Reduce step size
        alpha = beta * alpha;

        if bt_iter == max_backtrack
            % Line search failed - use tiny step
            fprintf('    WARNING: Line search failed after %d backtracks, using alpha=%.2e\n', ...
                    max_backtrack, alpha);
        end
    end

    %% Weight_new is already computed from last backtracking iteration
    % No need to recompute - already have it!

    %% Check convergence
    progress = (cost - newcost) / max(abs(newcost), 1e-10);

    if progress < 0
        % Cost increased - reduce step size or stop
        if alpha < 1e-8
            fprintf('    Iter %d: converged (alpha too small)\n', iter);
            break;
        else
            fprintf('    Iter %d: cost increased (%.6e -> %.6e), stopping\n', ...
                    iter, cost, newcost);
            break;
        end
    end

    % Compute relative change in weights
    rel_change = norm(Weight_new - Weight) / (norm(Weight) + 1e-10);

    % Update
    Weight = Weight_new;
    cost = newcost;

    % Print progress every 5 iterations or at end
    if mod(iter, 10) == 0 || iter == N_iter || progress < 1e-6
        nnz_weights = sum(Weight/max(Weight) >= 0.01);
        weight_l1 = sum(Weight);
        fprintf('    Iter %d: cost=%.6e, progress=%.2e, rel_change=%.2e, alpha=%.2e, ||W||₁=%.2f/%.2f, nnz=%d/%d\n', ...
                iter, cost, progress, rel_change, alpha, weight_l1, tau, nnz_weights, dim);
    end

    % Additional convergence check on relative weight change
    if rel_change < 1e-6
        fprintf('    Converged: relative weight change < 1e-6\n');
        break;
    end
end

fprintf('  Weight optimization completed: %d iterations\n', iter);


end
