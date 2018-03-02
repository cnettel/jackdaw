function [f] = diffpoisson(l,y,basey,minval,absrefpoint,filter)
% TODO: l support might be stale
mask = ~(y<0 | isnan(y));
f = @(varargin)diff_func(l,mask,basey, minval, absrefpoint, 1 ./ (filter.^2), varargin{:});

function [v,x,vals] = diff_func(l, mask, y, basey, minval, absrefpoint, filterrsq, x)
% Avoid singularities by moving away from zero
% This step will also ensure that those x values that are not in the mask
% are still positive.
lim = 1e-14./ l .* filterrsq;

% Special treatment occurs between xbase and upperlim
% At one point, values below xbase were subjected to hard cap
xbase = -basey + minval - lim ./ 2;
upperlim = xbase + lim;

subupper = x < upperlim;
xupperlim = x;
xupperlim(subupper) = upperlim(subupper);

% Expose intermediate steps slightly adhoc in this way
global x99;
if ~any(isnan(x))
  x99 = x;
end
vals = 0 * x;
absrefpointupperlim = absrefpoint - basey;
absrefpointupperlim(absrefpointupperlim<upperlim) = upperlim(absrefpointupperlim<upperlim);
absrefpointupperlim = absrefpointupperlim + basey;

% Compute log-poisson difference compared to absrefpoint, and with the log-lambda part capped at xupperlim, rather than true x (which might be less than xupperlim)
% Beyond xupperlim, extend linearly with the general 1 gradient, and a linear extrapolation of the y * ln(x) term from xupperlim
vals(mask) = -(y(mask) .* (log((xupperlim(mask) + basey(mask)) ./ max(absrefpointupperlim(mask),0.5e-9))) - l(mask).*(x(mask)-1*(absrefpoint(mask)-basey(mask)))) + (xupperlim(mask) - x(mask)) .* (y(mask) ./ max(xupperlim(mask)+basey(mask),1e-15)) - (absrefpointupperlim(mask) - absrefpoint(mask)) .* (y(mask) ./ max(absrefpointupperlim(mask),1e-15));

% Extra debug output
if nargout > 2
    vals2 = x;
    vals2(:) = 0;
    vals2(mask) = vals;
    vals = vals2;
end

% In theory, we might want the limit for the quadratic completion to be different than lim, but not in practice
lim2 = lim;

limfac = 1./l;
% Add quadratic for all low-value elements
subs = x < xbase + lim2;
vals(subs) = vals(subs) + ((x(subs) - xbase(subs) - lim2(subs)).^2.*1./lim2(subs) .* limfac(subs));

% Compensate by quadratic from absrefpoint position, if any
subs2 = absrefpoint - basey < xbase + lim2;
vals(subs2) = vals(subs2) - ((absrefpoint(subs2) - basey(subs2) - xbase(subs2) - lim2(subs2)).^2.*1./lim2(subs2) .* limfac(subs2));

v = sum(vals);

if nargout > 1
    g = y(mask)./max(xupperlim(mask) + basey(mask),1e-15) - 1;
    oldx = x;
    x(:) = 0;
    x(mask) = -g;
    if any(subs)
      x(subs) = x(subs) + 2 * (oldx(subs) - xbase(subs) - lim2(subs)).^1 .* (1./lim2(subs).^1) .* limfac(subs);
    end
    x = x;
end

