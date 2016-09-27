function [outpattern, details] = healer(pattern, support, lambdas, initguess, maxdiff, relmumasked, z0)

numrounds = 1;
indices = 1:numel(pattern)';
% Everything needs to be square and the same dimension
side2 = size(pattern,1);
pattern = reshape(pattern, side2*side2, 1);

opts = tfocs_SCD;
opts.alg = 'AT';
opts.tol = 1e-11;
opts.maxmin = -1;
opts.debug = 0;
opts.restart = 200000;
opts.maxIts = 200000;
opts.countOps = 1;
%opts.cntr_reset = 0;
opts.stopCrit = 3;
opts.printStopCrit = 1;
opts.continuation = 1;
opts.printEvery = 10; 

copts = continuation;
copts.maxIts = 1;
copts.muDecrement = 1;
copts.innerTol = 1e-11;
copts.tol = 1e-11;
copts.betaTol = 1e3;
copts.accel = 1;
copts.innerMaxIts = 10000;

x2 = reshape(support, side2*side2,1);
% Identical for real and imaginary
x2 = [x2; x2];
onefilter = ones(side2, side2);

ourlinpflat = @(x, mode) (jackdawlinop(x,mode,side2,side2,indices,onefilter));
x = reshape(initguess, side2 * side2, 1);
betanow = 1;

for outerround=1:numrounds
    outerround
    diffx = x;
    
    patternmask = (pattern < 0 | isnan(pattern));
    truebild = pattern;
    truebild(patternmask) = x(patternmask);
    filter = 1./sqrt(max(truebild(:)+1,1e-15)) + 0.00001;
    filter = 1./filter;
    %filter = filter + sqrt(0.1);
    %filter(:) = 1;
    filter = filter .* (1 - (1 - 1./sqrt(relmumasked)) * patternmask);
    filter = filter .* side2 .* side2 ./ sum(filter(:));
    filter = reshape(filter,side2*side2,1);
    %filter(:) = 1;
    smoothop = diffpoisson(filter, pattern(:), (diffx(:) + lambdas(:)).* 1 ./ filter, lambdas(:) * 1./filter);    

    ourlinp = @(x, mode) (jackdawlinop(x,mode,side2,side2,indices,filter));
    diffxt = ourlinpflat(diffx, 2);

    u = -diffxt;
    u(x2 > 0) = maxdiff;
    l = -diffxt;
    l(x2 > 0) = -maxdiff;

    [x,out] = tfocs_SCD(smoothop, {linop_adjoint(ourlinp),0*diffxt+1e-300}, {prox_boxDual(l, u, -1)}, betanow, 0, [], opts, copts);
    z0 = out.dual;

    x = x .* filter;

    % Return translation
    x = x + diffx(:);
    betanow = betanow * copts.muDecrement;
end

outpattern = reshape(x,side2,side2);
details = out;