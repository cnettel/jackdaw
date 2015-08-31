function [y] = jackdawlinop(x, mode, side1, side2, indices, b)

hs = side1 / 2;

switch mode,
case 0, y = [length(indices), side1*side1];
case 1,
x = fftshift(reshape(x, [side1 side1]));
x2 = zeros(side2);
x2(1:hs, 1:hs) = x(1:hs, 1:hs);
x2(end-hs+1:end, 1:hs) = x(end-hs+1:end, 1:hs);

x2(1:hs, end-hs+1:end) = x(1:hs, end-hs+1:end);
x2(end-hs+1:end, end-hs+1:end) = x(end-hs+1:end, end-hs+1:end);

x = fft2(x2);
y = real(1/side2 * x(indices));
%y = max(y, -b);
%'ISNAN1'
%sum(isnan(y))

case 2,
x2 = zeros(side2);
x2(indices) = real(x(:));
x2 = ifft2(x2);
origx = x;
x = x2;
x2 = zeros(side1, side1);
x2(1:hs, 1:hs) = x(1:hs, 1:hs);
x2(end-hs+1:end, 1:hs) = x(end-hs+1:end, 1:hs);

x2(1:hs, end-hs+1:end) = x(1:hs, end-hs+1:end);
x2(end-hs+1:end, end-hs+1:end) = x(end-hs+1:end, end-hs+1:end);
%x2 = x2 + x2.';
%x2 = x2 * 0.5;

x2 = ifftshift(x2);

y = side2 * x2(:);
%'ISNAN2'
%[sum(isnan(y)) max(origx) min(origx)]
end
