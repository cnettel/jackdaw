function [outpattern, details, factor] = healer(pattern, support, bkg, initguess, alg, numrounds, qbarrier, nzpenalty, iters, tols, nowindow)

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

if nargin < 11
  nowindow = []
end

if isempty(nowindow)
  nowindow = 0;
end

iterfactor = 1.1;

% Handle scalars
nzpenalty = ones(1, numrounds) .* nzpenalty;
qbarrier = ones(1, numrounds) .* qbarrier;

% Everything needs to be square and the same dimension
[dims, side2, fullsize, pshape, cshape] = getdims(pattern);

origpattern = pattern;
pattern = reshape(pattern, fullsize, 1);

opts = tfocs;
opts.alg = alg;

% Note, the tolerance will be dependent on the accuracy of the (double precision) FFT
% which essentially sums side2^dims entries (succession of two side2 sums)
opts.maxmin = 1;
opts.restart = 5e5;
opts.countOps = 1;
opts.printStopCrit = 1;
opts.printEvery = 2500;
% Use "no regress" restart option
opts.restart = -10000000;
% Special hack in our version of TFOCS to accept mode where both
% objective function and gradient are checked when determining no-regress option
% Change to only 'fun' if your version does not support this.
opts.autoRestart = 'fun';

mask = [reshape(support, pshape)  0 * reshape(support, pshape)];
mask = reshape(mask, fullsize * 2,1);
% Purely real, i.e. zero mask in imaginary space
ourlinpflat = jackdawlinop(origpattern,1);
% No windowing used within linop [for now]
ourlinp = ourlinpflat;

global factor
global basepenalty



if isempty(initguess)
    initguess = pattern(:);
    initguess(initguess < 0) = 0;
end

x = reshape(initguess, fullsize, 1);
xprev = x;
y = x;
jval = 0;

				% No actual filter windowing used, window integrated in the scale in diffpoisson instead              
filter = ones(fullsize, 1);   
rfilter = 1./filter;  



for outerround=1:numrounds
    if nowindow == 1
      factor = ones(65536, 1);
      basepenalty = 1 - mask;
    else
    [factor, basepenalty] = createwindows(origpattern, mask, qbarrier(outerround));
    end
    
    y = y .* factor;
    x = x .* factor;

    penalty = basepenalty * nzpenalty(outerround);

    % Acceleration scheme based on assumption of linear steps in response to decreasing qbarrier
    if outerround > 1 && (qbarrier(outerround) ~= qbarrier(outerround - 1))
        xprev = xprev .* factor;
        if (jval > 0)
            diffx = x + (x - xprev) .* (qbarrier(outerround) / qbarrier(outerround - 1));

            smoothop = diffpoisson(factor, pattern(:), diffx(:), bkg(:), diffx, filter, qbarrier(outerround));
            [proxop, diffxt, level, xlevel] = createproxop(diffx, penalty, ourlinp);

            	    newstep = ourlinp(x - xprev, 2);
        negstep = -newstep;
        negstep(penalty == 0) = 0;
	    newstep(penalty > 0) = 0;
        negstep = ourlinp(negstep, 1);
	    newstep = ourlinp(newstep, 1);
        y = x;        
	    y = y + halfboundedlinesearch((newstep), @(z) (smoothop(z + (y-diffx))));
	    step = norm(y - x)
        
        y = y + halfboundedlinesearch(negstep, @(z) (smoothop(z + (y-diffx)) + proxop(ourlinp(z + (y-diffx-xlevel), 2))));
        step2 = norm(y - x)
        
        y = y + halfboundedlinesearch(-negstep, @(z) (smoothop(z + (y-diffx)) + proxop(ourlinp(z + (y-diffx-xlevel), 2))));
        step3 = norm(y - x)
        
            y = y + halfboundedlinesearch((x - xprev), @(z) (smoothop(z + (y-diffx)) + proxop(ourlinp(z + (y-diffx-xlevel), 2))));
            step4 = norm(y - x)
            
             y = y + halfboundedlinesearch(negstep, @(z) (smoothop(z + (y-diffx)) + proxop(ourlinp(z + (y-diffx-xlevel), 2))));
        step5 = norm(y - x)
            
           y = y + halfboundedlinesearch(-negstep, @(z) (smoothop(z + (y-diffx)) + proxop(ourlinp(z + (y-diffx-xlevel), 2))));
        step6 = norm(y - x)
        
        y = y + halfboundedlinesearch((newstep), @(z) (smoothop(z + (y-diffx))));
	    step7 = norm(y - x)
        
        %smoothop2 = diffpoisson(factor, pattern(:), diffx(:), bkg(:), diffx, filter, qbarrier(outerround));
        %y = y + halfboundedlinesearch((x - xprev), @(z) (smoothop2(z + (y-diffx))));
        y = y + halfboundedlinesearch((x - xprev), @(z) (smoothop(z + (y-diffx)) + proxop(ourlinp(z + (y-diffx-xlevel), 2))));
	    step8 = norm(y - x)


        end
        xprev = x ./ factor;
        jval = jval + 1;
    end
    
    xprevinner = y;
    jvalinner = -1;
    opts.maxIts = ceil(iters(outerround) / iterfactor);
    while true
      % Inner acceleration scheme based on overall difference to previous pre-acceleration start
      if jvalinner >= 0
        diffx = y;

        % TODO: Should bkg(:) be included in diffx???
        smoothop = diffpoisson(factor, pattern(:), diffx(:), bkg(:), diffx, filter, qbarrier(outerround));
	[proxop, diffxt, level, xlevel] = createproxop(diffx, penalty, ourlinp);
        x = y + halfboundedlinesearch((y - xprevinner), @(x) (smoothop(x) + proxop(ourlinp(x - xlevel, 2))));
      else
        x = y;
      end
    xprevinner = y;
	jvalinner = jvalinner + 1;
	opts.maxIts = ceil(opts.maxIts * iterfactor);
    opts.tol = tols(outerround);
    opts.L0 = 2 ./ qbarrier(outerround);
    opts.Lexact = opts.L0 * sqrt(96*96*96);
    opts.alpha = 0.1;
    opts.beta = 0.1;
    diffx = x;
    
    % TODO: Should bkg(:) be included in diffx???
    smoothop = diffpoisson(factor, pattern(:), diffx(:), bkg(:), diffx, filter, qbarrier(outerround));
    [proxop, diffxt, level, xlevel] = createproxop(diffx, penalty, ourlinp);

    qbarrier(outerround)
    [x,out] = tfocs({smoothop}, {ourlinp,xlevel}, proxop, -level * 1, opts);
    
    xtupdate = x;
    x = ourlinp(x,1);
    xrtnorm = norm(xtupdate - ourlinp(x, 2))
    xstep = x;
    xupdate = x + xlevel;
    oldy = y;
    prevstep = xprevinner - diffx;
    levelprevdiff = norm(prevstep)
    y = (xupdate(:)) + diffx(:);
    % Acceleration step continuing in the same direction
    y = y + halfboundedlinesearch(xupdate, @(x) (smoothop(x + xupdate(:))  + proxop(ourlinp(x + xstep, 2))));
    smoothop(y - diffx(:))
    pstep = proxop(ourlinp(y - (diffx(:) + xlevel(:)), 2))
    
    smoothop(0 * diffx(:))
    proxop(ourlinp(-xlevel(:), 2))
    proxop(-level)       
    
    olddiffx = diffx;
    diffx = y;        
    
    normdiff1 = norm(ourlinp(level(:), 1) - xlevel(:))
    normdiff2 = norm(level(:) + ourlinp(-xlevel(:), 2))
    norm(level(:))
    smoothop = diffpoisson(factor, pattern(:), diffx(:), bkg(:), diffx, filter, qbarrier(outerround));
            [proxop, diffxt, level, xlevel] = createproxop(diffx, penalty, ourlinp);
    levelxdiff = norm(xprevinner - y)
    
    smoothop(olddiffx(:) - diffx(:))
    proxop(ourlinp(olddiffx(:) - (diffx(:) + xlevel(:)), 2))
    
    smoothop(0 * diffx(:))
    proxop(ourlinp(-xlevel(:), 2))
    
    % Is the distance to the new point from the previous end point, shorter than from the previous end point to the starting point?
    % If so, the acceleration step was too large, change to previous starting point and redo.
    % Otherwise, divergence is a real risk. This is believed to be a non-issue with the line search scheme now in place.
    if levelprevdiff > levelxdiff
        % Reset acceleration
        %y = xprevinner;
        'Do we need to reset acceleration?'
        %continue
    end           
    
    x = y;

    global x2;
    x2 = x ./ factor;
    save x26 x2

    % Our step was very small for desired accuracy level, break
    %if abs(smoothop(y - diffx) - smoothop(xprevinner(:) - diffx(:)) + proxop(ourlinp(y - (diffx + xlevel), 2)) - proxop(ourlinp(xprevinner(:), 2) - diffxt(:) - level(:)))  < 1e-7 / sqrt(qbarrier(outerround))
    diffstep = xprevinner - y;
    rchange = norm(diffstep(pattern >= 0), 1) / norm(pattern(pattern >= 0) .* factor(pattern >= 0), 1)
    if pstep > 0
        'Penalty too low, saddle point circus'      
        break;
    end
    
    if jvalinner >= 0 && rchange < 5e-9 * out.niter && ...
            abs(smoothop(y - diffx) - smoothop(xprevinner(:) - diffx(:)) + proxop(ourlinp(y - (diffx + xlevel), 2)) - proxop(ourlinp(xprevinner(:), 2) - diffxt(:) - level(:)))  < 1e-6 %* sqrt(qbarrier(outerround)) 
      'Next outer iteration'
      break        
    end
    if jvalinner > 20
        'Going next anyway'
        break
    end
   
    if out.niter < opts.maxIts
        'Reverting maxIts'
        % We did a lot of restarts, do not extend maxIts just yet.
        opts.maxIts = ceil(opts.maxIts / iterfactor);
    end
    if outerround < numrounds && qbarrier(outerround) == qbarrier(outerround + 1)
        'Not final iter, moving on'
        break
    end
    end
    y = y ./ factor;
    x = x ./ factor;
end

outpattern = reshape(x, pshape);
details = out;