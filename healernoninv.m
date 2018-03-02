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
    filter = 1./sqrt(max(truebild(:)+1,1e-15)) + 0.00001;
    filter = 1./filter;
    %filter = filter + sqrt(0.1);
    %filter(:) = 1;
    filter = filter .* (1 - (1 - 1./sqrt(relmumasked)) * patternmask);
    filter(:) = 1;
    filter = hann(side2, 'periodic');
    filter = filter * filter';
    filter = filter + 1e-7;
    filter(filter<0) = 0;
    filter = fftshift(filter);
    filter = filter + filter';
    filter = filter .* side2 .* side2 ./ sum(filter(:));
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
%% Positive real
    l(1:65536) = -diffxt(1:65536);
    
    
    
    global l99;
    l99 = l;
    global projector;
    %projector = proj_box(l, u);
    %projector = proj_l2(1, spdiags(diffxt,1:(side2*side2), side2*side2, side2*side2), x2 > 0);
    proxector = prox_boxDual(l, u, -1);
    
    
    %l2 = ourlinpflat(addlevel .* filterorig(:), 2);

    l2(isinf(l2)) = 0;
    global addlevel;

    addlevel = ourlinp(l2,1);
    addlevel2 = ourlinpflat(addlevel .* filterorig(:), 2);
    AAA = norm(l2)
    norm(addlevel2)

    l(:) = -0.1;
    u(:) =  0.1;
    u(x2 > 0) = maxdiff;
    l(x2 > 0) = -maxdiff;
    % Just tighten it a little bit
    l = min(-l2*(1-1e-6),l);
    u = max(-l2*(1-1e-6),u);
    projector2 = proj_box(l, u);
    
    %z2 = ourlinp(z0,2);
    %[val, z3] = projector2(z2, 0);
    %z0 = ourlinp(z3, 1) ./ filter;
    filter(:) = 1;
    global smoothop
    smoothop = diffpoisson(filter, pattern(:), (diffx(:) + lambdas(:)).* 1 ./ filter, lambdas(:) * 1./filter, diffx .* 1./filter);
    desired = max(0,diffxorig);
    %z0 = findfixpoint(smoothop, z0, betanow, 1e4, desired - diffx);
    %[val, z11] = projector(ourlinp(z0,2),0);
    %z0 = ourlinp(z11, 1) ./ filter;
    global z19;
    z19 = z0;
    global z12;
    %[throwaway,z12] = smoothop(-z0/betanow, 1/betanow);
    %throwaway
    
    %smoothop = diffpoisson(filter, pattern(:), (diffx(:) + lambdas(:)).* 1 ./ filter, lambdas(:) * 1./filter, z12 + diffx(:) .* 1./filter); 
    %global z13;
    %[throwaway,z13] = smoothop(-z0/betanow, 1/betanow);
    %throwaway
    
    global z14;
    z14=diffx;
    global z15;
    z15 = smoothop;
   
    
    
    'KORREKT29'
%    [x,out] = tfocs_SCD([], {ourlinp,0;linop_scale(1),0 }, {prox_dualize2(smoothop,'negative',-z0{1}), prox_dualize2(projector,'negative',-z0{1})}, betanow, 0*ourlinp(desired,2)-0*diffxt+0*ourlinp((truebild(:)-diffx(:)-lambdas(:))./filter,2), z0, opts, copts);
    %[x,out] = tfocs_SCD(smoothop, {linop_adjoint(ourlinp);0*zeros(1048576,1)}, {proxector}, betanow, zeros(1048576, 1), z0, opts, copts);
    %[x,out] = tfocs(smoothop, {ourlinp;0*zeros(1048576,1)}, projector, zeros(2097152,1), opts);
    size(addlevel)
    norm(addlevel)
  
    zzz = smooth_quad_hack2(1e8 .* (x22 <= 0), -diffxt-l2);
    ppp = zzz(-l2)
    %PPP = sum((-l2.^2).* (x2 <= 0))
    len2 = length(x2);
    
    
    %[x,out] = tfocs({smoothop; smooth_quad(1e-9)}, {ourlinp,addlevel; linop_compose(linop_dot(double(pattern>=0)), ourlinp), sum(addlevel(pattern>=0)-pattern(pattern>=0)+diffx(pattern>=0))}, smooth_quad_hack2(1e7 .* (x22 <= 0), -diffxt-l2), -l2 * 1, opts);
    [x,out] = tfocs({smoothop}, {ourlinp,addlevel}, smooth_quad_hack2(1e18 .* (x22 <= 0), -diffxt-l2), -l2 * 1, opts);
%[x,out] = tfocs(smoothop, {ourlinp,addlevel}, projector2, -l2, opts);




    %global z9;
    %z9 = ourlinpflat(x,1);% ./ filterorig;
    %global z10;
    %[throwaway, z10] = smoothop(-out.dual/1e-15,1/1e-15);
    %nz9 = norm(z9)
    %nzdiff = norm(z9-z10)
    %opts.maxIts = opts.maxIts * 4;
    %[x2,out2] = tfocs_SCD(proj_box(l, u), {ourlinp}, prox_dualize(smoothop,'negative'), betanow / 2, ourlinp((truebild(:)-diffx(:)-lambdas(:))./filter,2), z9, opts, copts);
    %z0 = out.dual;
    %z9 = out2.dual;

    %[val, x] = projector(ourlinp(z10,2), 0);
    %x = ourlinp(x,1) ./ filter;
    %global z11;
    %z11 = x;
    %x = z9;
    
    %x2 = ourlinp(x2,1) ./ filter;
    %x(patternmask) = x(patternmask) + (x2(patternmask) - x(patternmask) / 0.5);

    % Return translation
    global z99
    z99 = x;
    x = ourlinp(x,1) + addlevel;%./ filterorig(:) + addlevel;
    global z100
    z100 = x;
    size(x)
    size(diffx)
    newx = -1;
    %while any(newx <= -0.005 + 2*eps(max(abs(newx))))
    %  newx = x + diffx(:);
    %  x = x * 0.99 + 2 * eps(max(abs(newx)));
    %  'Dip'
    %end  
    %x = newx;
    %[skip, newx] = smoothop(x,1e-15);
    x = x + diffx(:);
    %x = x + 2 * eps(max(abs(x)));
    
    betanow = betanow * copts.muDecrement;
end

outpattern = reshape(x,side2,side2);
details = out;