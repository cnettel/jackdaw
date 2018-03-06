function [outpattern, details] = healer(pattern, support, bkg, initguess, alg, numrounds, qbarrier, nzpenalty, iters, tols)

% Handle scalars
nzpenalty = ones(1, numrounds) .* nzpenalty;
qbarrier = ones(1, numrounds) .* qbarrier;

% Everything needs to be square and the same dimension
side2 = size(pattern,1);
pattern = reshape(pattern, side2*side2, 1);

opts = tfocs;
opts.alg = alg;

% Note, the tolerance will be dependent on the accuracy of the (double precision) FFT
% which essentially sums side2*side2 entries (succession of two side2 sums)
opts.maxmin = 1;
opts.restart = 5e5;
opts.countOps = 1;
opts.printStopCrit = 1;
opts.printEvery = 100; 

mask = reshape(support, side2*side2,1);
% Identical for real and imaginary
% Or purely real
mask = [mask; mask * 0];
ourlinpflat = @(x, mode) (jackdawlinop(x,mode,side2,side2,1));

x = reshape(initguess, side2 * side2, 1);

for outerround=1:numrounds
    opts.maxIts = iters(outerround);
    opts.tol = tols(outerround);
    diffx = x;
    %opts.Lexact = 2 / qbarrier(outerround);
    opts.Lexact = max(pattern) / qbarrier(outerround).^2;
    
    filter = hann(side2, 'periodic');
    filter = filter * filter';
    filter = filter + 1e-6;
    filter = fftshift(filter);
    filter = reshape(filter,side2*side2,1);
    rfilter = 1./filter;

    ourlinp = @(x, mode) (jackdawlinop(x,mode,side2,side2,rfilter));
    diffxt = ourlinpflat(diffx .* filter(:), 2);

    level = -diffxt;
    level(mask > 0) = 0;
          
    xlevel = ourlinp(level, 1);
    factor = ones(side2*side2,1);
    smoothop = diffpoisson(factor, pattern(:), (diffx(:) + bkg(:)).* 1 ./ factor, bkg(:) * 1./factor, diffx ./ factor, filter, qbarrier(outerround));
    [x,out] = tfocs({smoothop}, {ourlinp,xlevel}, smooth_quad_hack2(nzpenalty(outerround) .* (mask <= 0), -diffxt-level), -level * 1, opts);
    x = ourlinp(x,1) + xlevel;
    x = x + diffx(:);
end

outpattern = reshape(x,side2,side2);
details = out;