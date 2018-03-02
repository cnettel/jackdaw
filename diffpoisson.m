function [f] = diffpoisson(l,y,basey,minval,baseyz10)
mask = ~(y<0 | isnan(y));
f = @(varargin)diff_func(l,mask,basey, minval, baseyz10, varargin{:});
epsval = max(eps(basey));

function [v,x,vals] = diff_func(l, mask, y, basey, minval, baseyz10, x, epsval)
x1 = x(mask);
x2 = x(mask);
origx = x;
invnow = 0;

% Avoid singularities by moving away from zero
% This step will also ensure that those x values that are not in the mask
% are still positive.
oldx = x;
global f2;
lim = 1e-14./ l ./f2(:) ./ f2(:);
xbase = -basey+2*epsval+minval - lim ./ 2;
upperlim = xbase + lim;


%mask6 = y(:) >= lim*0.5;
%mask6 = y;
%mask6(:) = 1;
%upperlim(mask6) = -basey(mask6) + 2 * epsval + minval(mask6);
%xbase(mask6) = upperlim(mask6);

mask5 = x < upperlim; %& x >= xbase;
mask8 = mask5;
xupperlim = x;
xupperlim(mask5) = upperlim(mask5);
%mask6 = xupperlim < -basey + 2 * epsval;
%xupperlim(mask6) = -basey(mask6) + 2 * epsval;

%%baseyupperlim = basey;
%%mask5 = upperlim > 0;
%%add_basey = 0 * basey;
%%add_basey(mask5) = upperlim(mask5);
%%baseyupperlim = baseyupperlim + add_basey;

global x99;
if ~any(isnan(x))
  x99 = x;
end
vals = 0 * x;
baseyz10upperlim = baseyz10 - basey;
baseyz10upperlim(baseyz10upperlim<upperlim) = upperlim(baseyz10upperlim<upperlim);
baseyz10upperlim = baseyz10upperlim + basey;

vals(mask) = -(y(mask) .* (log((xupperlim(mask) + basey(mask)) ./ max(baseyz10upperlim(mask),0.5e-9))) - l(mask).*(x(mask)-1*(baseyz10(mask)-basey(mask)))) + (xupperlim(mask) - x(mask)) .* (y(mask) ./ max(xupperlim(mask)+basey(mask),1e-15)) - (baseyz10upperlim(mask) - baseyz10(mask)) .* (y(mask) ./ max(baseyz10upperlim(mask),1e-15));
% Constant part could xupperlim(mask) or x(mask) depending on desired gradient
if nargout > 2
    vals2 = x;
    vals2(:) = 0;
    vals2(mask) = vals;
    vals = vals2;
end
if invnow > 0
    resx = x;
    x = (xbyt - x) .* 1./t;
    vals = resx .* x - vals;
    x = -x;
end
lim2 = lim * 1;%0.5167;
% Has to be 1
limfac = 1./l;%./(f2(:).^1);
subs = x < xbase + lim2;

vals(subs) = vals(subs) + ((x(subs) - xbase(subs) - lim2(subs)).^2.*1./lim2(subs) .* limfac(subs));

subs2 = baseyz10 - basey < xbase + lim2;
vals(subs2) = vals(subs2) - ((baseyz10(subs2) - basey(subs2) - xbase(subs2) - lim2(subs2)).^2.*1./lim2(subs2) .* limfac(subs2));

v = sum(vals);%.*(f2(:)));

if nargin == 7% && nargout > 1
    g = y(mask)./max(xupperlim(mask) + basey(mask),1e-15) - 1;%l(mask) .* (x(mask) + basey(mask));
    oldx = x;
    x(:) = 0;
    x(mask) = -g;
%    x(mask & mask8) = 1;
    if any(subs)
      x(subs) = x(subs) + 2 * (oldx(subs) - xbase(subs) - lim2(subs)).^1 .* (1./lim2(subs).^1) .* limfac(subs);
    end
    x = x;% .*f2 ;
end

%endx = x(1)
%endxbase = xbase(1)
