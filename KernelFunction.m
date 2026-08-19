function kernel_coef = KernelFunction(distance_sorted, K, kernel)

u = distance_sorted(1:K)/distance_sorted(K); 
switch lower(kernel)
    case {'epanechnikov'}
        kernel_coef = (1-u.^2);
    case {'cosine'}
        kernel_coef = cos(u*pi/2);
    case {'quartic'}
        kernel_coef = (1-u.^2).^2;
    case {'tricube'}
        kernel_coef = (1-u.^3).^3;
    case {'triweight'}
         kernel_coef = (1-u.^2).^3;
end

