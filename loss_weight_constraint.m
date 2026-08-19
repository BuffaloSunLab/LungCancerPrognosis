function loss_weight = loss_weight_constraint(data, f, Weight, Para)
% LOSS_WEIGHT_CONSTRAINT Compute weight optimization loss (constrained version)
%
% This is the CONSTRAINED version for use with projected gradient descent.
% It computes ONLY the smoothness loss, NOT the L1 penalty.
%
% COMPARISON WITH PENALTY VERSION:
%   - PENALTY:    loss = λ₁·L_smooth + λ₂·||W||₁  (L1 in objective)
%   - CONSTRAINT: loss = λ₁·L_smooth               (L1 as constraint, not in objective)
%
% The L1 constraint ||W||₁ ≤ τ is enforced by PROJECTION, not by penalty.
%
% IMPORTANT: If Para.truncate_k is set, uses TRUNCATED kernel (k-NN only)
%            to match the gradient computation in optimize_w_constraint.m
%
% Inputs:
%   data   - [d x N] feature matrix
%   f      - [N x 1] instance labels
%   Weight - [d x 1] feature weights (DIRECT, not squared!)
%   Para   - parameters with fields:
%            .lambda1: smoothness penalty weight
%            .sigma: kernel bandwidth
%            .distance_metric: 'cityblock' or 'squaredeuclidean'
%            .truncate_k: number of nearest neighbors (optional, default: 25)
%            .tau: L1 constraint (NOT used in this function, only for reference)
%
% Outputs:
%   loss_weight - smoothness loss ONLY (no L1 penalty term)

lambda1 = Para.lambda1;
sigma = Para.sigma;
distance_metric = Para.distance_metric;

N_patterns = length(f);

% Check if truncated kernel should be used
if isfield(Para, 'truncate_k') && Para.truncate_k > 0
    use_truncated = true;
    truncate_k = Para.truncate_k;
else
    use_truncated = false;
end

%% Smoothness loss
if use_truncated
    % ========================================================================
    % TRUNCATED KERNEL: Use k-NN pairs only (MUST MATCH GRADIENT COMPUTATION!)
    % ========================================================================
    % CRITICAL: Use pre-computed pairs if available to avoid rebuilding k-NN graph

    if isfield(Para, 'symmetric_pairs')
        % ====================================================================
        % FAST PATH: Use pre-computed k-NN pairs from caller
        % ====================================================================
        % This ensures gradient and loss use the EXACT SAME pairs throughout
        % the iteration, even as weights change during line search.
        %
        % WHY THIS IS CRITICAL:
        % - Gradient is computed once at iteration start using k-NN graph based on current Weight
        % - Line search evaluates this loss function many times with different Weight candidates
        % - If we rebuild k-NN graph here, each evaluation uses DIFFERENT pairs
        % - Result: gradient ≠ ∂L/∂W, line search minimizes WRONG objective → cost increases
        % - Solution: Use FIXED k-NN pairs throughout the iteration

        symmetric_pairs = Para.symmetric_pairs;
    else
        % ====================================================================
        % SLOW PATH: Build k-NN graph (used when called standalone, not during optimization)
        % ====================================================================

        % Build symmetric k-nearest neighbor graph (same as in gradient)
        data_weighted = diag(sqrt(Weight)) * data;
        dist_weighted = pdist2(data_weighted', data_weighted', 'squaredeuclidean');

        % Find k nearest neighbors for each point
        neighbors = cell(N_patterns, 1);
        for i = 1:N_patterns
            [~, idx] = mink(dist_weighted(i, :), min(truncate_k, N_patterns));
            neighbors{i} = idx;
        end

        % Build symmetric neighbor pairs (i~j if i∈N(j) OR j∈N(i))
        neighbor_pairs = [];
        for i = 1:N_patterns
            for j = neighbors{i}
                if i < j  % Only add each pair once
                    neighbor_pairs = [neighbor_pairs; i, j];
                end
            end
        end
        symmetric_pairs = unique(sort(neighbor_pairs, 2), 'rows');
    end

    % Compute loss over symmetric pairs only
    loss_weight_smooth = 0;
    for pair_idx = 1:size(symmetric_pairs, 1)
        i = symmetric_pairs(pair_idx, 1);
        j = symmetric_pairs(pair_idx, 2);

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

        % Label difference squared
        Fij = (f(i) - f(j))^2;

        % Add contribution
        loss_weight_smooth = loss_weight_smooth + Fij * K_ij;
    end

elseif isfield(Para, 'precomputed_Z') && isfield(Para, 'precomputed_F')
    % ========================================================================
    % PRECOMPUTED: Fast path for full kernel
    % ========================================================================
    Z = Para.precomputed_Z;
    F = Para.precomputed_F;

    % Compute weighted distances
    dist_pair = Weight' * Z;  % [1 x N_pairs]
    A_pair = exp(-dist_pair / sigma);

    % Smoothness loss
    loss_weight_smooth = sum(F .* A_pair);
else
    % ========================================================================
    % FULL KERNEL: Compute over all pairs
    % ========================================================================
    loss_weight_smooth = 0;

    for i = 1:N_patterns
        if i < N_patterns
            % Vectorize inner loop for instance i paired with all j > i
            j_indices = (i+1):N_patterns;

            % Compute differences for all j > i
            if strcmp(distance_metric, 'cityblock')
                diff_i = abs(data(:, i) - data(:, j_indices));
            else  % squaredeuclidean
                diff_i = (data(:, i) - data(:, j_indices)).^2;
            end

            % Weighted distances
            dist_i = Weight' * diff_i;  % [1 x n_j]

            % Gaussian kernel
            A_i = exp(-dist_i / sigma);  % [1 x n_j]

            % Label differences squared
            F_i = (f(i) - f(j_indices)').^2;  % [1 x n_j]

            % Add contribution to smoothness loss
            loss_weight_smooth = loss_weight_smooth + sum(F_i .* A_i);
        end
    end
end

% Normalize by N_patterns (not N_pairs) for reasonable lambda scale
loss_weight_smooth = loss_weight_smooth / N_patterns;

%% ============================================================================
%% Total loss: ONLY smoothness term (NO L1 PENALTY!)
%% ============================================================================
% KEY DIFFERENCE from penalty version:
%   - Penalty version:    loss = λ₁·L_smooth + λ₂·||W||₁
%   - Constraint version: loss = λ₁·L_smooth
%
% The L1 constraint ||W||₁ ≤ τ is enforced by PROJECTION in the optimization,
% NOT by adding a penalty term to the objective.
%
% This is mathematically equivalent (by Lagrangian duality) but numerically
% more stable and easier to control.

loss_weight = lambda1 * loss_weight_smooth;

% No L1 penalty term!
% The constraint is handled by project_L1_ball_nonneg() in optimize_w_constraint.m

end
