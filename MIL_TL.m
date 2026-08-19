function [Weight, f, loss, f_history] = MIL_TL(data, bag_labels, weight_init, label_init, bag_ids, Para)
% MIL_TL Multiple Instance Learning with Transfer Learning
%
% Alternating optimization of instance labels (f) and feature weights (Weight)
%
% Inputs:
%   data        - [d x N] feature matrix (d features, N instances)
%   bag_labels  - [M x 1] binary bag labels (0 or 1)
%   weight_init - [d x 1] initial feature weights
%   label_init  - [N x 1] initial instance labels
%   bag_ids     - [N x 1] bag assignment for each instance (values 1 to M)
%   Para        - struct with fields:
%                 .Niter          - max iterations (default: 100)
%                 .lambda1        - smoothness penalty
%                 .tau            - L1 constraint
%                 .sigma          - kernel bandwidth (default: 1)
%                 .label_iter     - inner iterations for label optimization (can be scalar or [min, max])
%                 .Weight_iter    - inner iterations for weight optimization (can be scalar or [min, max])
%                 .distance_metric - distance metric: 'cityblock' (L1) or 'squaredeuclidean' (L2^2)
%                                    (default: 'cityblock')
%                 .truncate_k     - number of nearest neighbors for truncated kernel (default: 25)
%
% Note: For adaptive inner iterations, set label_iter and Weight_iter as [min, max].
%       The number of inner iterations will linearly increase from min (iteration 1)
%       to max (final iteration).
%
% Outputs:
%   Weight - [d x 1] optimized feature weights
%   f      - [N x 1] optimized instance labels (scaled to [0, 1])
%   loss   - [Niter x 1] objective values at each iteration
%   f_history - [N x (Niter+1)] matrix of f values at each iteration (optional)
%
% Algorithm:
%   Alternates between:
%   1. Optimize f (instance labels) given fixed Weight
%   2. Optimize Weight (feature weights) given fixed f
%   Converges when Weight changes less than 1e-4 relative difference

%% Initialize
close all
[dim, N_patterns] = size(data);
Weight = weight_init;
f = label_init;
loss = zeros(Para.Niter + 1, 1);
f_history = zeros(N_patterns, Para.Niter + 1);

% Set default distance metric if not specified
if ~isfield(Para, 'distance_metric')
    Para.distance_metric = 'cityblock';
end

% Set default truncation window size if not specified
if ~isfield(Para, 'truncate_k')
    Para.truncate_k = 25;
end

if ~isfield(Para, 'sigma')
    Para.sigma = 1;
end


% Parse inner iteration settings (can be scalar or [min, max])
if isscalar(Para.label_iter)
    label_iter_min = Para.label_iter;
    label_iter_max = Para.label_iter;
    adaptive_label = false;
else
    label_iter_min = Para.label_iter(1);
    label_iter_max = Para.label_iter(2);
    adaptive_label = true;
end

if isscalar(Para.Weight_iter)
    weight_iter_min = Para.Weight_iter;
    weight_iter_max = Para.Weight_iter;
    adaptive_weight = false;
else
    weight_iter_min = Para.Weight_iter(1);
    weight_iter_max = Para.Weight_iter(2);
    adaptive_weight = true;
end

%% Pre-compute pairwise differences if dataset is small enough
N_pairs = N_patterns * (N_patterns - 1) / 2;
N_elements = N_pairs * dim;
small_flag = (N_elements <= 1e9);

if small_flag

    % Pre-compute Z (data differences) - only depends on data (constant)
    Z = zeros(dim, N_pairs);
    k = 0;
    for i = 1:N_patterns
        for j = i+1:N_patterns
            k = k + 1;
            if strcmp(Para.distance_metric, 'cityblock')
                Z(:, k) = abs(data(:, i) - data(:, j));
            else  % squaredeuclidean
                Z(:, k) = (data(:, i) - data(:, j)).^2;
            end
        end
    end

    % Store in Para for reuse
    Para.precomputed_Z = Z;
end

% Compute initial objective and store initial f
obj_init = get_objective(data, f, bag_labels, bag_ids, Weight, Para);
loss(1) = obj_init.total;
f_history(:, 1) = f;



%% Alternating optimization
iter = 0;
Diff = 1;

while iter < Para.Niter && Diff >= 1e-4
    iter = iter + 1;

    % === Compute adaptive inner iterations ===
    if adaptive_label
        % Linear increase from min to max
        current_label_iter = round(label_iter_min + ...
            (label_iter_max - label_iter_min) * (iter - 1) / (Para.Niter - 1));
        Para_iter = Para;
        Para_iter.label_iter = current_label_iter;
    else
        Para_iter = Para;
    end

    if adaptive_weight
        % Linear increase from min to max
        current_weight_iter = round(weight_iter_min + ...
            (weight_iter_max - weight_iter_min) * (iter - 1) / (Para.Niter - 1));
        Para_iter.Weight_iter = current_weight_iter;
    end

    % === Step 1: Optimize instance labels (f) ===
    f = optimize_label_truncate(data, f, Weight, bag_labels, bag_ids, Para_iter);

    % Update precomputed F after f changes (only if precomputed matrices exist)
    if small_flag
        F = zeros(1, N_pairs);
        k = 0;
        for i = 1:N_patterns
            for j = i+1:N_patterns
                k = k + 1;
                F(k) = (f(i) - f(j))^2;
            end
        end
        Para_iter.precomputed_F = F;
    end

    % === Step 2: Optimize feature weights (Weight) ===
    Weight_new = optimize_w_constraint(data, f, Weight, Para_iter);

    % === Check convergence ===
    % Relative change in weights
    weight_sum = sum(Weight);
    if weight_sum > 0
        Diff = sum(abs(Weight_new - Weight)) / weight_sum;
    else
        Diff = sum(abs(Weight_new - Weight));
    end

    Weight = Weight_new;

    % === Track objective and f history ===
    obj = get_objective(data, f, bag_labels, bag_ids, Weight, Para);
    loss(iter + 1) = obj.total;
    f_history(:, iter + 1) = f;

end

% Trim loss and f_history to actual iterations
loss = loss(1:iter+1);
f_history = f_history(:, 1:iter+1);

% Normalize f to [0, 1]
f_min = min(f);
f_range = range(f);
if f_range > 0
    f = (f - f_min) / f_range;
else
    f = zeros(size(f));  % All labels are the same
end

fprintf('MIL_TL: Converged after %d iterations (Diff = %.6e)\n', iter, Diff);

end


