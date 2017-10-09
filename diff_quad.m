function [f] = diff_quad(l,y,diffy)
nonzeroy = max(y,1e-14);
mask = ~(y<0 | isnan(y));
nonzeroy(~mask) = y(~mask);
f = @(varargin)diff_quad_func(l,nonzeroy,diffy, mask, varargin{:});


function [v,x] = diff_quad_func(l, y, diffy, mask, x, t)
lnew = l;
q = smooth_quad(lnew(mask).^2);

origx = x;
x = (x(mask) - (y(mask) - diffy(mask).*lnew(mask)));
origx2 = x;
[v,x] = q(x,t);
diffx2 = x - origx2;
x3 = (x).^2.*lnew(mask).^2 - ((diffy(mask).*lnew(mask) - y(mask))) .^2;
v = 0.5 * sum(x3);
x = origx;
origx(mask) = origx(mask) + diffx2;
