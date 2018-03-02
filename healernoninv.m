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

mask = reshape(support, side2*side2,1);
% Identical for real and imaginary
% Or purely real
mask = [mask; mask * 0];
ourlinpflat = @(x, mode) (jackdawlinop(x,mode,side2,side2,1));

x = reshape(initguess, side2 * side2, 1);

for outerround=1:numrounds
    diffx = x;    
    
    filter = hann(side2, 'periodic');
    filter = filter * filter';
    filter = filter + 1e-7;
    filter = fftshift(filter);
    filter = reshape(filter,side2*side2,1);
    filterorig = filter;
    
    rfilterorig = 1./filterorig;
    ourlinp = @(x, mode) (jackdawlinop(x,mode,side2,side2,rfilterorig));
    diffxt = ourlinpflat(diffx .* filterorig(:), 2);

    level = -diffxt;
    level(mask > 0) = 0;
          
    xlevel = ourlinp(level, 1);

    factor = 1;
    smoothop = diffpoisson(factor, pattern(:), (diffx(:) + lambdas(:)).* 1 ./ factor, lambdas(:) * 1./factor, diffx ./ factor);
    [x,out] = tfocs({smoothop}, {ourlinp,xlevel}, smooth_quad_hack2(1e18 .* (mask <= 0), -diffxt-level), -level * 1, opts);
    x = ourlinp(x,1) + xlevel;
    x = x + diffx(:);
end

outpattern = reshape(x,side2,side2);
details = out;