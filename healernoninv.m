function [outpattern, details, factor] = healer(pattern, support, bkg, initguess, alg, numrounds, qbarrier, nzpenalty, iters, tols)

% Main COACS function.

% pattern, the pattern to phase
% support, the support mask (in autocorrelation space)
% bkg, background signal, support for non-zero values here basically stale at this point
% initguess, start guess for the pattern
% alg, the TFOCS algorithm to use
% numrounds, the number of outermost iterations
% qbarrier, the qbarrier (2 * l) in each round
% nzpenalty, the penalty constant outside of the support
% iters, number of TFOCS iterations within the round
% tols, the tolerance used to determine end of iteration in TFOCS

% Acceleration and continuation adapted to COACS is also used, but hardcoded.

% Handle scalars
nzpenalty = ones(1, numrounds) .* nzpenalty;
qbarrier = ones(1, numrounds) .* qbarrier;

% Everything needs to be square and the same dimension
side2 = size(pattern,1);
fullsize = numel(pattern);
pattern = reshape(pattern, fullsize, 1);

opts = tfocs;
opts.alg = alg;

% Note, the tolerance will be dependent on the accuracy of the (double precision) FFT
% which essentially sums side2*side2 entries (succession of two side2 sums)
opts.maxmin = 1;
opts.restart = 5e5;
opts.countOps = 1;
opts.printStopCrit = 1;
opts.printEvery = 100;
% Use "no regress" restart option
opts.restart = -100000;
% Special hack in our version of TFOCS to accept mode where both
% objective function and gradient are checked when determining no-regress option
% Change to only 'fun' if your version does not support this.
opts.autoRestart = 'fun,gra';

mask = reshape(support, fullsize,1);
% Purely real, i.e. zero mask in imaginary space
mask = [mask; mask * 0];
ourlinpflat = jackdawlinop(side2,1);

global filter
filter = hann(side2, 'periodic');
%filter = filter * filter';
filter = fftshift(filter);

side3 = side2

filter1 = repmat(filter, [1 side2 side2]);
filter2 = ones(1,side2,1);
filter2(:) = filter(:);
filter2 = repmat(filter2, [side2 1 side2]);

filter3 = ones(1,1,side2);
filter3(:) = filter(:);
filter3 = repmat(filter3, [side2 side2 1]);



filter = filter1 .* filter2 .* filter3;
filter = filter + 1e-3;

global factor
factor = reshape(filter, fullsize, 1);
factor = factor .* factor;

if isempty(initguess)
    initguess = pattern(:) .* factor;
    initguess(initguess < 0) = 0;
end

x = reshape(initguess, fullsize, 1);
xprev = x;
y = x;
jval = 0;

for outerround=1:numrounds
    % Acceleration scheme based on assumption of linear steps in response to decreasing qbarrier
    if outerround > 1 && (qbarrier(outerround) ~= qbarrier(outerround - 1))
        if (jval > 0)
            diffx = x + (x - xprev) .* (qbarrier(outerround) / qbarrier(outerround - 1));
            diffxt = ourlinpflat(diffx .* filter(:), 2); 
            smoothop = diffpoisson(factor, pattern(:), diffx(:), bkg(:), diffx, filter, qbarrier(outerround));
            proxop = zero_tolerant_quad(penalty, -diffxt-level);
            y = x + halfboundedlinesearch((x - xprev), @(z) (smoothop(z + (x-diffx)) + proxop(ourlinp(z + (x-diffx), 2))));
        end
        step = norm(y - x)
        xprev = x;
        jval = jval + 1;
    end
    
    xprevinner = y;
    jvalinner = -1;
    opts.maxIts = ceil(iters(outerround) / 1.01);
    while true
      % Static .9 inner acceleration scheme for repeated iterations at the same qbarrier level
      if jvalinner >= 0
        %x = y + (y - xprevinner) * 0.9;
        x = y + halfboundedlinesearch((y - xprevinner), @(x) (smoothop(x + (y-diffx)) + proxop(ourlinp(x + (y-diffx), 2))));
      else
        x = y;
      end
    xprevinner = y;
	jvalinner = jvalinner + 1;
	opts.maxIts = ceil(opts.maxIts * 1.01);
    opts.tol = tols(outerround);
    opts.L0 = 1 ./ qbarrier(outerround);
    diffx = x;
    
    % No actual filter windowing used, window integrated in the scale in diffpoisson instead
    filter(:) = 1;
    filter = reshape(filter,fullsize,1);    
    rfilter = 1./filter;       

    global ourlinp
    ourlinp = jackdawlinop(side2,rfilter);
  
    diffxt = ourlinpflat(diffx .* filter(:), 2);

    % Perform Hann windowing on our penalty matrix
    global penalty;
    penalty = (reshape((mask(1:fullsize) <= 0) + j * (mask(fullsize + 1:2*fullsize) <= 0),side2,side2,side2));
    penalty = fftshift(fftn(fftn(fftshift(penalty)) .* reshape(factor,side2,side2,side2))) / fullsize;
    penalty = [real(penalty) imag(penalty)];

    % Filter out numerical inaccuracies
    penalty(penalty < 1e-8) = 0;

    penalty = reshape(penalty, 2 * fullsize, 1);
    penalty = penalty * nzpenalty(outerround);

    % Translate those portions that are not penalized
    level = -diffxt * 0;
    level(penalty == 0) = 0;
          
    xlevel = ourlinp(level, 1);
    % TODO: Should bkg(:) be included in diffx???
    smoothop = diffpoisson(factor, pattern(:), diffx(:), bkg(:), diffx, filter, qbarrier(outerround));
    proxop = zero_tolerant_quad(penalty, -diffxt-level);

    [x,out] = tfocs({smoothop}, {ourlinp,xlevel}, proxop, -level * 1, opts);
    
    xtupdate = x;
    x = ourlinp(x,1) + xlevel;
    xupdate = x;
    oldy = y;
    prevstep = xprevinner - diffx;
    levelprevdiff = norm(prevstep)
    y = x(:) + diffx(:);
    y = y + halfboundedlinesearch(x, @(x) (smoothop(x + (y-diffx)) + proxop(ourlinp(x + (y-diffx), 2))));
    y = y + halfboundedlinesearch(prevstep, @(x) (smoothop(x + (y-diffx)) + proxop(ourlinp(x + (y-diffx), 2))));
    levelxdiff = norm(xprevinner - y)
    
    % Is the distance to the new point from the previous end point, shorter than from the previous end point to the starting point?
    % If so, the acceleration step was too large, change to previous starting point and redo.
    % Otherwise, divergence is a real risk.
    if levelprevdiff > levelxdiff
        % Reset acceleration
        %y = xprevinner;
        'Do we need to reset acceleration?'
        %continue
    end
    
    x = y;

    global x2;
    x2 = x;
    save x3 x2

    % Our step was very small for desired accuracy level, break
    if abs(smoothop(y - diffx) - smoothop(xprevinner(:) - diffx(:)) + proxop(ourlinp(y - diffx, 2) - level(:)) - proxop(ourlinp(xprevinner(:), 2) - diffxt(:) - level(:))) /opts.maxIts < 1e-7
      'Next outer iteration'
        break
    end

    % Our change in function value was so small that the numerical accuracy can be in jeopardy
    % Within iteration, we are using translation to increase accuracy
    % Increase number of steps in order to possibly achieve a large enough cnhange
    %if norm(xupdate) / norm(x) < max(1e-4, sqrt(eps(1) * side2 * side2))
    if levelxdiff < max(1e-7 * norm(x), sqrt(eps(1) * fullsize))
        opts.maxIts = opts.maxIts * 2;
        continue
    end
    end
end

outpattern = reshape(x,side2,side2);
details = out;