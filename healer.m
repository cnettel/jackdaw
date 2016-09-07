function [outpattern, details] = healer(pattern, support, lambdas, initguess, maxdiff)

numrounds = 3;
indices = 1:numel(pattern);

opts = tfocs_SCD;
opts.alg = 'AT';
opts.tol = 1e-11;
opts.maxmin = -1;
opts.debug = 0;
opts.restart = 5000;
opts.maxIts = 10000;
opts.countOps = 1;
%opts.cntr_reset = 0;
opts.stopCrit = 3;
opts.printStopCrit = 1;
opts.continuation = 1;
opts.printEvery = 1000;

copts = continuation;
copts.maxIts = 2;
copts.muDecrement = 0.75;
copts.innerTol = 1e-11;
copts.tol = 1e-11;
copts.betaTol = 1;
copts.accel = 1;
copts.innerMaxIts = 10000;

x2 = reshape(support, side2*side2,1);
% Identical for real and imaginary
x2 = [x2; x2];
onefilter = ones(1024, 1024);

ourlinpflat = @(x, mode) (jackdawlinop(x,mode,side2,side2,indices,onefilter));
z0 = zeros(numel(pattern),1);
x = initguess;

for outerround=1:numrounds
    diffx = x;
    
    scaling = 1./sqrt(max(x,1e-15)) + 0.00001;
    filter = 1./scaling;
    filter = filter + sqrt(0.1);
    filter = filter .* 1048576 ./ sum(filter);
    smoothop = diffpoisson(filter, pattern(:), (diffx(:) + lambdas(:)).* 1 ./ filter);
    filter = reshape(filter,1024,1024);

    ourlinp = @(x, mode) (jackdawlinop(x,mode,side2,side2,indices,filter));
    diffxt = ourlinpflat(diffx, 2);

    u = -diffxt;
    u(x2 > 0) = maxdiff;
    l = -diffxt;
    l(x2 > 0) = -maxdiff;

    [x,out] = tfocs_SCD(smoothop, {linop_adjoint(ourlinp),0*origdiffxt+1e-300}, {prox_boxDual(l, u, -1)}, betanow, 0, z0, opts, copts);
    z0 = out.dual;

    x = x .* filter;

    % Return translation
    x = x + diffx(:);
    betanow = betanow * copts.muDecrement;
end

outpattern = x;
details = out;