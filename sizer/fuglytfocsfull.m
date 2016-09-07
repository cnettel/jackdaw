index = 2597;
%bild70 = bildrna3_1;
%lambda70 = lambdarna3_1;

bild = bild70(:,:,index);
lambda = lambda70(:,:,index);
lambda(240:246, 286:414) = 0;
lambda(125:295, 173:278) = 0;
lambda(175:247,1:207) = 0;
lambda(207:272,145:304) = 0;
bild59 = zeros(414,414);
lambda59 = zeros(414,414);
bild59(:) = bild(:);
lambda59(:) = lambda(:);


% bilden = fft2(bild59 - lambda59);
% bilden(7:408,:) = 0;
% bilden(:,7:408) = 0;
% bilden = ifft2(bilden);
% bilden = real(bilden) + lambda59;
bilden = bild59 + 1e-9;

mask3 = zeros(414,414);
mask3(1:end,1:end) = 1;
mask3 = mask3 .* (lambda > 0);
bilden = bilden .* mask3;
mask4 = find(1 - mask3);
step = 1e-0;
a= 0;

oldbilden2 = bilden;

send = 5;


indices = find(mask3(:) >= 0);
flatbild = bild59(:);
flatbild(mask4) = NaN;
flatlambdas = lambda59(:);

side1 = 20;
side2 = 414;
ourlinp = @(x, mode) (jackdawlinop(x,mode,side2,side2,indices,flatlambdas));
x0 = zeros(side2, side2);
x0(1, 1) = 1;
x0 = ifftshift(x0);
opts = tfocs;
opts.printEvery = 250;

opts.alg = 'GRA';
%opts.restart = -200;
%opts.L0 = 1e9;
%opts.L0 = 1e6;


opts.tol = -1;
opts.maxmin = 1;
opts.debug = 1;
%opts.restart = 30;
opts.maxIts = 3000;
%opts.cntr_reset = 0;
opts.printStopCrit = 1;

x2 = zeros(side2, side2);
hs = side1 * 0.5;
x2(1:hs+1, 1:hs+1) = 1;
x2(end-hs+1:end, 1:hs+1) = 1;

x2(1:hs+1, end-hs+1:end) = 1;
x2(end-hs+1:end, end-hs+1:end) = 1;

global dual
clear dual

x2 = ifftshift(x2);
x2(x2 > 0) = Inf;
x2 = reshape(x2,side2*side2,1);

while true
[x,out] = tfocs(smooth_logLPoisson(flatbild), {ourlinp, flatlambdas}, {proj_box(zeros(side2*side2,1), x2)}, x(:),opts);
imagesc(reshape(ourlinp(x,1),414,414));
end
%x10(:) * 100 + 9e-4, opts);
%ourlinp(ourlinp(x10(:) * 100,1)+1e-3,2), opts);
%[x,out] = tfocs(smooth_logLPoisson(flatbild), {ourlinp, flatlambdas}, proj_nested(ourlinp,x2), x0(:), opts);