function linop = jackdawlinop(pattern, filter)

[dims, side2, fullsize, pshape, cshape] = getdims(pattern);

if dims == 3
% 3D mode also implies half-pixel shift in centering, for now...
%range = linspace(0, -pi + (pi / side), side);
  range = fftshift(pi / 2 + ((0.25:(side2 - 0.75)) * pi/side2));
%range = fftshift(range);
%range = range(1:end-1);
%range(:) = 0;

%range = range(1:end-1);
  [Xs, Ys, Zs] = meshgrid(range, range, range);
  shifter = exp(sqrt(-1)*(Xs+Ys+Zs));
%shifter = fftshift(shifter);

  range = linspace(0, -pi + pi/(side2), side2);
  [Xs, Ys, Zs] = meshgrid(range, range, range);
  unshifter = exp(sqrt(-1)*(Xs+Ys+Zs));

%shifter = fftshift(shifter
else
  shifter = 1;
  unshifter = 1;
end
  linop = @(x, mode) helper(x, mode, dims, side2, fullsize, pshape, cshape, filter, unshifter, shifter);
end

function [y] = helper(x, mode, dims, side, fullsize, pshape, cshape, filter, unshifter, shifter)
% FFT2 for TFOCS with possible additional window and differently sized matrices
% Complex data expressed as stacked real data

switch mode,
case 0, y = [fullsize, 2 * fullsize];
case 1,
x = reshape(x, cshape);
if dims == 3
  x = fftshift(x(1:side,1:side,1:side) + 1j*x(1:side, side+1:side*2, 1:side));
else
  x = fftshift(x(1:side,1:side) + 1j*x(1:side, side+1:side*2));
end
x2 = x;
%x2 = x2 .* conj(shifter);
x2 = x2 .* conj(shifter);
x = fftn(x2);
x = x .* conj(shifter);
%x = (x + flipdim(flipdim(flipdim(x,1),2),3)) * 0.5;
y = side^(-dims/2) * real(x(:));
y = y(:) .* filter(:);

case 2,
x2 = zeros(pshape);
x2(:) = real(x(:)) .* filter(:);
%x2 = x2 + flipdim(flipdim(flipdim(x2,1),2),3);

x2 = x2 .* shifter;
x2 = ifftn(x2);
x2 = x2 .* (shifter);
x2 = ifftshift(x2);


y = reshape([real(side^(dims/2) * x2) imag(side^(dims/2) * x2)], 2 * fullsize, 1);
end

end
