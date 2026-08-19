function cost = TL_cost(alpha, P)
% keyboard
v = P.v;
d = P.d;
Z = P.Z;
F = P.F;
sigma = P.sigma;
lambda = P.lambda;
N_patterns = P.N_patterns;

v = v-alpha*d;

Weight = v.^2;
dist_pair = Weight'*Z;
A_pair = exp(-dist_pair/(sigma));

cost = sum(F.*A_pair, 2) + lambda*sum(Weight);

end