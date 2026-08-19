function obj_values = get_objective(data, f, bag_label, bag_id, Weight, Para)
% GET_OBJECTIVE Compute all objective function components (CONSTRAINT VERSION)
%
% This version is for L1 CONSTRAINT formulation:
%   minimize: L_smooth + L_bag
%   subject to: W ≥ 0, ||W||₁ ≤ τ
%
% The L1 constraint is NOT in the objective - it's enforced by projection.
%
% Inputs:
%   data      - [d x N] feature matrix
%   f         - [N x 1] instance labels
%   bag_label - [M x 1] bag labels
%   bag_id    - [N x 1] bag assignment for each instance
%   Weight    - [d x 1] feature weights
%   Para      - parameters with fields:
%                 .lambda1: smoothness penalty weight
%                 .tau: L1 constraint (for tracking only, not in objective)
%                 .sigma: kernel bandwidth
%
% Outputs:
%   obj_values - struct with fields:
%                .obj_smooth   - smoothness objective
%                .obj_bag      - bag loss
%                .obj_sparsity - L1 norm (tracked but NOT in objective)
%                .total        - lambda1 * obj_smooth + obj_bag
lambda1 = Para.lambda1;
sigma = Para.sigma;
distance_metric = Para.distance_metric;
[dim, N_patterns] = size(data);

%% Compute smoothness objective
% Check if precomputed matrices are available from optimize_w
if isfield(Para, 'precomputed_Z') && isfield(Para, 'precomputed_F')
    % Use precomputed matrices (fastest path)
    Z = Para.precomputed_Z;
    F = Para.precomputed_F;

    dist_pair = Weight' * Z;
    A_pair = exp(-dist_pair / sigma);
    obj_smooth = sum(A_pair .* F);
else
    % Compute from scratch
    N_pairs = N_patterns * (N_patterns - 1) / 2;
    N_elements = N_pairs * dim;
    small_flag = (N_elements <= 1e9);

    if small_flag
        % Fast path: vectorized pre-computation
        F = zeros(1, N_pairs);
        Z = zeros(dim, N_pairs);
        k = 0;

        for i = 1:N_patterns
            for j = i+1:N_patterns
                k = k + 1;
                F(k) = (f(i) - f(j))^2;
                if strcmp(distance_metric, 'cityblock')
                    Z(:, k) = abs(data(:, i) - data(:, j));
                else  % squaredeuclidean
                    Z(:, k) = (data(:, i) - data(:, j)).^2;
                end
            end
        end

        dist_pair = Weight' * Z;
        A_pair = exp(-dist_pair / sigma);
        obj_smooth = sum(A_pair .* F);
    else
        % Slow path: vectorize inner loop
        obj_smooth = 0;

        for i = 1:N_patterns
            if i < N_patterns
                j_indices = (i+1):N_patterns;

                % Compute differences for all j > i
                if strcmp(distance_metric, 'cityblock')
                    Z_i = abs(data(:, i) - data(:, j_indices));
                else  % squaredeuclidean
                    Z_i = (data(:, i) - data(:, j_indices)).^2;
                end

                % Label differences squared
                F_i = (f(i) - f(j_indices)).^2;

                % Weighted distances and kernel
                dist_i = Weight' * Z_i;
                A_i = exp(-dist_i / sigma);

                obj_smooth = obj_smooth + sum(F_i .* A_i);
            end
        end
    end
end

%% Compute bag loss (use accumarray for vectorization)
bag_unique = unique(bag_id, 'stable');
N_bags = length(bag_unique);

% Remap bag_ids to consecutive integers for accumarray
[~, bag_id_remapped] = ismember(bag_id, bag_unique);

% Find max instance score per bag using accumarray
bag_max_scores = accumarray(bag_id_remapped, f, [], @max);

% Get bag labels for the unique bags present in this data
bag_label_subset = bag_label(bag_unique);

% Compute squared error for each bag
bag_errors = (bag_label_subset - bag_max_scores).^2;
obj_bag = sum(bag_errors);

%% L1 norm (tracked but NOT in objective for constraint version)
obj_sparsity = sum(abs(Weight));

%% Combine objectives
% CONSTRAINT VERSION: NO L1 penalty in objective
% The constraint ||W||₁ ≤ τ is enforced by projection, not by penalty
obj_values.obj_smooth = obj_smooth;
obj_values.obj_bag = obj_bag;
obj_values.obj_sparsity = obj_sparsity;  % Tracked for monitoring only
obj_values.total = lambda1 * obj_smooth + obj_bag;  % NO lambda2 term!

end
