function linop = jackdawlinop(side, filter)

%range = linspace(0, -pi + (pi / side), side);
range = fftshift(pi / 2 + ((0.25:(side - 0.75)) * pi/side));
%range = fftshift(range);
%range = range(1:end-1);
%range(:) = 0;

%range = range(1:end-1);
[Xs, Ys, Zs] = meshgrid(range, range, range);
shifter = exp(sqrt(-1)*(Xs+Ys+Zs));
%shifter = fftshift(shifter);

range = linspace(0, -pi + pi/(side), side);
[Xs, Ys, Zs] = meshgrid(range, range, range);
unshifter = exp(sqrt(-1)*(Xs+Ys+Zs));

%shifter = fftshift(shifter);

linop = @(x, mode) helper(x, mode, side, filter, unshifter, shifter);

end

function [y] = helper(x, mode, side, filter, unshifter, shifter)
% FFT2 for TFOCS with possible additional window and differently sized matrices
% Compelx data expressed as stacked real data

switch mode,
case 0, y = [side*side*side, 2*side*side*side];
case 1,
x = reshape(x, [side 2 * side side]);
x = fftshift(x(1:side,1:side,1:side) + j*x(1:side, side+1:side*2, 1:side));
x2 = x;
%x2 = x2 .* conj(shifter);
x2 = x2 .* conj(shifter);
x = fftn(x2);
x = x .* conj(shifter);
y = side^(-1.5) * real(x(:));
y = y(:) .* filter(:);

case 2,
x2 = zeros(side,side,side);
x2(:) = real(x(:)) .* filter(:);
x2 = x2 .* shifter;
x2 = ifftn(x2);
x2 = x2 .* (shifter);
x2 = ifftshift(x2);


y = reshape([real(side^(1.5) * x2) imag(side^1.5 * x2)], 2 * side * side * side, 1);
end

end
