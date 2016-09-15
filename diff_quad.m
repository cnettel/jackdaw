function [f] = diff_quad(l,y,diffy)
f = @(varargin)diff_quad_func(l,y,diffy, varargin{:});


function [v,x] = diff_quad_func(l, y, diffy, x, t)
lnew = l;
q = smooth_quad(lnew.^2);

origx = x;
x = (x - (y - diffy));
origx2 = x;
[v,x] = q(x,t);
diffx2 = x - origx2;
x3 = (x).^2.*lnew.^2 - ((diffy - y)) .^2 .* lnew.^2;
v = 0.5 * sum(x3);
x = origx + diffx2;