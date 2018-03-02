function [f] = diffpoisson(l,y,diffy,minval,diffyz10)
nonzeroy = max(y,1e-14);
mask = ~(y<0 | isnan(y));
%minval(~mask) = 0;
%diffy = max(diffy, 0.01/2);
f = @(varargin)diff_func(l,mask,nonzeroy,diffy, minval, diffyz10, varargin{:});


function [v,x,vals] = diff_func(l, mask, y, diffy, minval, diffyz10, x, t)
% l199 = l(199)
% m199 = mask(199)
% y199 = y(199)
% diffy199 = diffy(199)
% minval199 = minval(199)
% diffyz10199 = diffyz10(199)
%global inputx;
%inputx = x;
%inputx(1)

x1 = x(mask);
x2 = x(mask);
origx = x;
invnow = 0;

if nargin <= 7
    t = 0;
end

if invnow > 0
    if t > 0
        t = 1/t;
    else
        t = 1e15;        
    end
    x = -x ./ t;
    xbyt = x;
end


epsval = max(eps(diffy));

if t > 0
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
    t = 0;
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
global f2;
lim = 1e-14./ l ./f2(:) ./ f2(:);
xbase = -diffy+2*epsval+minval - lim ./ 2;
upperlim = xbase + lim;


%mask6 = y(:) >= lim*0.5;
%mask6 = y;
%mask6(:) = 1;
%upperlim(mask6) = -diffy(mask6) + 2 * epsval + minval(mask6);
%xbase(mask6) = upperlim(mask6);


% Taken care of below
%if t > 0
if nargin > 7
  asdas
  x = max(x, xbase);
  mask3 = (x >= xbase) & (x <= upperlim);
  mask4 = mask3 & (origx < xbase);
  x(mask4) = xbase(mask4);
  mask4 = mask3 & (origx > upperlim);
  x(mask4) = upperlim(mask4);
  mask4 = mask3 & (origx >= xbase) & (origx <= upperlim);
  x(mask4) = origx(mask4);
end

%mask4 = x > 1e2;
%x(mask4) = 1e2;
%end
mask5 = x < upperlim; %& x >= xbase;
mask8 = mask5;
xupperlim = x;
xupperlim(mask5) = upperlim(mask5);
%mask6 = xupperlim < -diffy + 2 * epsval;
%xupperlim(mask6) = -diffy(mask6) + 2 * epsval;

%%diffyupperlim = diffy;
%%mask5 = upperlim > 0;
%%add_diffy = 0 * diffy;
%%add_diffy(mask5) = upperlim(mask5);
%%diffyupperlim = diffyupperlim + add_diffy;

global x99;
if ~any(isnan(x))
  x99 = x;
end
vals = 0 * x;
diffyz10upperlim = diffyz10 - diffy;
diffyz10upperlim(diffyz10upperlim<upperlim) = upperlim(diffyz10upperlim<upperlim);
diffyz10upperlim = diffyz10upperlim + diffy;

vals(mask) = -(y(mask) .* (log((xupperlim(mask) + diffy(mask)) ./ max(diffyz10upperlim(mask),0.5e-9))) - l(mask).*(x(mask)-1*(diffyz10(mask)-diffy(mask)))) + (xupperlim(mask) - x(mask)) .* (y(mask) ./ max(xupperlim(mask)+diffy(mask),1e-15)) - (diffyz10upperlim(mask) - diffyz10(mask)) .* (y(mask) ./ max(diffyz10upperlim(mask),1e-15));
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

subs2 = diffyz10 - diffy < xbase + lim2;
vals(subs2) = vals(subs2) - ((diffyz10(subs2) - diffy(subs2) - xbase(subs2) - lim2(subs2)).^2.*1./lim2(subs2) .* limfac(subs2));

v = sum(vals);%.*(f2(:)));


if (nargin == 7 || t == 0) && (sum(abs(oldx - x)) > 0 || any(x < xbase))
    %sum(abs(oldx-x))
    %t;
    %v = Inf;
end

if any(subs)
end
if nargin == 7% && nargout > 1
    g = y(mask)./max(xupperlim(mask) + diffy(mask),1e-15) - 1;%l(mask) .* (x(mask) + diffy(mask));
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
