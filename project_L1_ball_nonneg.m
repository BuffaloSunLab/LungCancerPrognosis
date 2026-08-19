function W_proj = project_L1_ball_nonneg(W, tau)
% PROJECT_L1_BALL_NONNEG Project onto non-negative L1 ball
%
% Projects W onto the set {W ≥ 0, ||W||₁ ≤ τ}
%
% This is used for CONSTRAINED optimization:
%   minimize: L(W)
%   subject to: W ≥ 0, ||W||₁ ≤ τ
%
% Algorithm: Duchi et al. (2008) "Efficient Projections onto the L1-Ball
%            for Learning in High Dimensions"
%
% Inputs:
%   W   - [d x 1] weight vector to project
%   tau - L1 constraint (max L1 norm)
%
% Outputs:
%   W_proj - [d x 1] projected weights
%
% Example:
%   W = [5; 3; -2; 1];
%   tau = 6;
%   W_proj = project_L1_ball_nonneg(W, tau);
%   % Result: W_proj = [4; 2; 0; 0], sum = 6, all ≥ 0

W = W(:);  % Ensure column vector

%% Step 1: Enforce non-negativity first
W = max(W, 0);

%% Step 2: Check if already in L1 ball
W_sum = sum(W);
if W_sum <= tau
    % Already satisfies constraint
    W_proj = W;
    return;
end

%% Step 3: Project onto L1 ball using soft thresholding
% We need to find threshold θ such that:
%   sum(max(W - θ, 0)) = τ
%
% Algorithm:
% 1. Sort W in descending order
% 2. Find largest k such that W_sorted(k) > (cumsum(W_sorted(1:k)) - τ) / k
% 3. Set θ = (cumsum(W_sorted(1:k)) - τ) / k
% 4. Apply soft thresholding: W_proj = max(W - θ, 0)

% Sort in descending order
W_sorted = sort(W, 'descend');
cumsum_W = cumsum(W_sorted);

% Find the threshold
% We're looking for the largest k where:
%   W_sorted(k) > (cumsum_W(k) - tau) / k
k_max = 1;
for k = 1:length(W_sorted)
    theta_k = (cumsum_W(k) - tau) / k;
    if W_sorted(k) > theta_k
        k_max = k;
    else
        break;
    end
end

% Compute optimal threshold
theta = (cumsum_W(k_max) - tau) / k_max;

% Apply soft thresholding
W_proj = max(W - theta, 0);

%% Verification (optional, can be removed for speed)
W_proj_sum = sum(W_proj);
if abs(W_proj_sum - tau) > 1e-8 && abs(W_proj_sum - tau) / tau > 1e-6
    % Only warn if relative error is significant
    if W_proj_sum < tau * 0.99
        fprintf('    Warning: Projection resulted in ||W||₁ = %.6f < τ = %.6f (relative error: %.2e)\n', ...
                W_proj_sum, tau, abs(W_proj_sum - tau) / tau);
    end
end

end
