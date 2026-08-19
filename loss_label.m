function [loss_label_smooth, loss_label_y] = loss_label(A_mat, f, bag_labels, bag_ids, t)
% LOSS_LABEL Compute label optimization loss
%
% Inputs:
%   A_mat      - [N x N] kernel similarity matrix
%   f          - [N x 1] instance labels
%   bag_labels - [N x 1] bag-level labels (replicated for each instance)
%   bag_ids    - [N x 1] bag assignment for each instance
%   t          - temperature parameter for LogSumExp
%
% Outputs:
%   loss_label_smooth - smoothness loss (encourages similar instances to have similar labels)
%   loss_label_y      - bag loss (encourages max(instance labels) = bag label)

f = f(:);
N_patterns = length(f);

%% Smoothness loss
% Note: A_mat might be asymmetric (after truncation), so we compute full sum
% Normalize by N_patterns to match weight optimization normalization
% Since we sum over full NxN matrix (which double-counts off-diagonal pairs),
% dividing by N gives roughly the same scale as summing upper triangle / N
Fij = repmat(f', [N_patterns, 1]) - repmat(f, [1, N_patterns]);
Fij_square = Fij.^2;
loss_label_smooth = sum(sum(A_mat .* Fij_square)) / N_patterns;

%% Bag loss using LogSumExp (semi-vectorized)
% Get unique bags and their labels
[bags_unique, id] = unique(bag_ids, 'stable');
bag_label_unique = bag_labels(id);
f_scaled = f * t;

loss_label_y = 0;
for i = 1:length(bags_unique)
    id = find(bag_ids == bags_unique(i));
    f_bag = f_scaled(id);

    % LogSumExp with numerical stability
    m = min(f_bag);
    f_bag = f_bag - m;
    f_summed = sum(exp(f_bag), 1);
    LSE = (log(f_summed) + m) / t;

    % Squared error between LSE and bag label
    err_bag = (LSE - bag_label_unique(i))^2;
    loss_label_y = loss_label_y + err_bag;
end

% Normalize by number of bags
loss_label_y = loss_label_y / length(bags_unique);

end
