function [f] = diffpoisson(l,y,diffy,mask)
nonzeroy = max(y,1e-14);
mask = ~(y<0 | isnan(y));
nonzeroy(~mask) = y(~mask);
f = @(varargin)diff_func(l,mask,nonzeroy,diffy, varargin{:});


function [v,x] = diff_func(l, mask, y, diffy, x, t)
x1 = x(mask);
x2 = x(mask);

if nargin > 4 && t > 0
    t = 1./t * 0.5;
    % Solving the quadratic system for the derivative of the proximal
    % operator definition being 0.
    acbcone = t * (diffy(mask) - x1) + l(mask);
    factor = -((diffy(mask) .* x1 .* t + (y(mask) - diffy(mask).*l(mask))));
    
    x1 = -(acbcone + sign(acbcone) .* (sqrt(-4 .* t .* factor + acbcone.^2)))/(2*t);
    x2 = factor ./ (t .* x1);
else
    'Warning'
end

xb = [x1' ; x2'];
[~,b] = max(xb);

index = (b - 1) + (1:length(b)) * 2 - 1;

x(mask) = xb(index)';

% Avoid singularities by moving away from zero
% This step will also ensure that those x values that are not in the mask
% are still positive.
x = max(x, -diffy+2*eps(diffy));

vals = -(y(mask) .* (log((x(mask) + diffy(mask)) ./ diffy(mask))) - l(mask).*x(mask));
v = sum(vals);