function [f] = diffpoisson(l,y,diffy,minval)
nonzeroy = max(y,1e-14);
mask = ~(y<0 | isnan(y));
%minval(~mask) = 0;
f = @(varargin)diff_func(l,mask,nonzeroy,diffy, minval, varargin{:});


function [v,x,vals] = diff_func(l, mask, y, diffy, minval, x, t)
x1 = x(mask);
x2 = x(mask);
origx = x;

epsval = max(eps(diffy));

if nargin > 6 && t > 0
    t = 1./t;
    % Solving the quadratic system for the derivative of the proximal
    % operator definition being 0.
    % - k * b/(ab+bx) + b + d * (x-e)
    % where k is y (true photon count), a is diffy, b is l, d is t-inverse, 
    % e is the input
    % x and x is the output x
    %term = -l(mask) + t .* (x1 - diffy(mask))
    %factor = ((- y(mask) + diffy(mask) .* l(mask)) - x1 .* diffy(mask) .* t)
    
    acbcone = t * (diffy(mask) - x1) + l(mask);
    factor = -((diffy(mask) .* x1 .* t + (y(mask) - diffy(mask).*l(mask))));
    nowsign = sign(acbcone);
    nowsign(nowsign == 0) = 1;
    x1 = -(acbcone + nowsign .* (sqrt(-4 .* t .* factor + acbcone.^2)))/(2*t);
    %x1 = (term + sign(term) .* sqrt(term.^2 - 4 .* t .* factor)) ./ 2*t;
    x2 = factor ./ (t .* x1);
    %x1
    %x2
%     mask2 = abs(imag(x1)) > 0;
%     dy2 = diffy(mask2);
%     x1(mask2) = -dy2 + 2 * epsval;
%     x2(mask2) = 0;
else
    %'Warning'
end

xb = [x1' ; x2'];
[~,b] = max(xb);
% mask2 = abs(imag(x1)) > 0;
% 'Data'
% x3 = x(mask);
% x3 = x3(mask2);
% x3(1)
% x3 = x1(mask2);
% x3(1)
% y3 = y(mask);
% y3 = y3(mask2);
% y3(1)
% y3 = diffy(mask);
% y3 = y3(mask2);
% y3(1)
% y3 = l(mask);
% y3 = y3(mask2);
% y3(1)

index = (b - 1) + (1:length(b)) * 2 - 1;

x(mask) = xb(index)';

% Avoid singularities by moving away from zero
% This step will also ensure that those x values that are not in the mask
% are still positive.
oldx = x;
global x98; 
x98 = oldx;
lim = 0.01;
xbase = -diffy+2*epsval+minval-lim/2;
x = max(x, xbase);

upperlim = xbase + lim;
mask3 = (x > xbase) & (x < upperlim);
mask4 = mask3 & (origx < xbase);
x(mask4) = xbase(mask4);
mask4 = mask3 & (origx > upperlim);
x(mask4) = upperlim(mask4);
mask4 = mask3 & (origx > xbase) & (origx < upperlim);
x(mask4) = origx(mask4);

mask5 = x < upperlim;
xupperlim = x;
xupperlim(mask5) = upperlim(mask5);


vals = -(y(mask) .* (log((xupperlim(mask) + diffy(mask)) ./ diffy(mask))) - l(mask).*xupperlim(mask));
if nargout > 2
    vals2 = x;
    vals2(:) = 0;
    vals2(mask) = vals;
    vals = vals2;
end
v = sum(vals);
if (nargin == 6 || t == 0) && sum(abs(oldx - x)) > 0
    v = Inf;
end
if nargin == 6 && nargout > 1
    g = y(mask)./(x(mask) + diffy(mask)) - l(mask) .* (x(mask) + diffy(mask));
    x(:) = 0;
    x(mask) = -g;
end
