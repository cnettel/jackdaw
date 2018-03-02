function [outpattern, details] = healer(pattern, support, lambdas, initguess, maxdiff, relmumasked, z0, alg, betanow)
numrounds = 1;
% Everything needs to be square and the same dimension
side2 = size(pattern,1);
pattern = reshape(pattern, side2*side2, 1);

opts = tfocs;
opts.alg = alg;
opts.tol = 1e-300;
opts.maxmin = 1;
opts.debug = 1;
opts.restart = 5e5;
opts.maxIts = 5e5;
opts.countOps = 1;
%opts.L0 = 1e5;
%opts.cntr_reset = 0;
%%opts.stopCrit = 3;
opts.printStopCrit = 1;
opts.continuation = 1;
opts.printEvery = 1e3; 

copts = continuation;
copts.maxIts = 1;
copts.muDecrement = 1;
copts.innerTol = 1e-11;
copts.tol = 1e-11;
copts.betaTol = 1e-8;
copts.accel = 1;
copts.innerMaxIts = 10000;

x2 = reshape(support, side2*side2,1);
% Identical for real and imaginary
% Or purely real
x2 = [x2;x2 * 0];
onefilter = ones(side2, side2);

global ourlinpflat
ourlinpflat = @(x, mode) (jackdawlinop(x,mode,side2,side2,onefilter));

x = reshape(initguess, side2 * side2, 1);

global ourlinp;

for outerround=1:numrounds
    outerround
    diffx = x;
    x = max(x, 1e-14);
    
    
    patternmask = (pattern < 0 | isnan(pattern));
    truebild = pattern;
    truebild(patternmask) = x(patternmask) + lambdas(patternmask);
    filter = hann(side2, 'periodic');
    filter = filter * filter';
    filter = filter + 1e-7;
    filter = fftshift(filter);
    filter = reshape(filter,side2*side2,1);
    filterorig = filter;
    global f2;
    f2 = filterorig;
    %filter(:) = 1;
    
    %smoothop = diffpoisson(filter, pattern(:), (diffx(:) + lambdas(:)).* 1 ./ filter, lambdas(:) * 1./filter, (z12 + diffx(:) + lambdas(:)) .* 1./filter);
    global ourlinp;
    %filter2 = 1./filterorig(:);
    rfilterorig = 1./filterorig;
    ourlinp = @(x, mode) (jackdawlinop(x,mode,side2,side2,rfilterorig));
    diffxt = ourlinpflat(diffx .* filterorig(:), 2);

    diffxtorig = diffxt;
    %diffxt = diffxt - (x2 == 0) .* 1e-3 .* (0.5 - 0*rand(size(diffxt)));
    diffxorig = diffx;
    %diffx = ourlinpflat(diffxt, 1);

    size(z0)
    %[x,out] = tfocs(smoothop, {ourlinp}, proj_box(l, u), z0, opts);
    x22 = x2;% & (diffxt >= 0);
    u = -diffxt;
    u(x22 > 0) = maxdiff;
    l = -diffxt;
    l(x22 > 0) = -maxdiff;
    l2 = l;
    %l = l - 1e-1;
    %u = u + 1e-1;
          
    l2(isinf(l2)) = 0;
    addlevel = ourlinp(l2,1);

    factor = 1;
    smoothop = diffpoisson(factor, pattern(:), (diffx(:) + lambdas(:)).* 1 ./ factor, lambdas(:) * 1./factor, diffx ./ factor);
    [x,out] = tfocs({smoothop}, {ourlinp,addlevel}, smooth_quad_hack2(1e18 .* (x22 <= 0), -diffxt-l2), -l2 * 1, opts);
    x = ourlinp(x,1) + addlevel;
    x = x + diffx(:);
end

outpattern = reshape(x,side2,side2);
details = out;