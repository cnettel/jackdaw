function [y] = jackdawlinop(x, mode, side1, side2, filter)

hs = side1 / 2;

switch mode,
case 0, y = [numel(x), 2*side1*side1];
case 1,
x = reshape(x, [side1 2*side1]);
x = fftshift(x(1:side1,1:side1) + j*x(1:side1,side1+1:side1*2));
x2 = zeros(side2);

x2(1:hs, 1:hs) = x(1:hs, 1:hs);
x2(end-hs+1:end, 1:hs) = x(side1-hs+1:side1, 1:hs);
x2(1:hs, end-hs+1:end) = x(1:hs, side1-hs+1:side1);
x2(end-hs+1:end, end-hs+1:end) = x(side1-hs+1:side1, side1-hs+1:side1);

x = fft2(x2);
if side2 / 2 == floor(side2 / 2)
  x(:,side2 / 2 + 1) = 0;
  x(side2 / 2 + 1,:) = 0;
end
y = 1/side2 * real(x(:));
y = y(:) .* filter(:);

case 2,
x2 = zeros(side2,side2);
x2(:) = real(x(:)) .* filter(:);
if side2 / 2 == floor(side2 / 2)
  x2(:,side2 / 2 + 1) = 0;
  x2(side2 / 2 + 1,:) = 0;
end
x2 = ifft2(x2);
x = x2;
x2 = zeros(side1, side1);

x2(1:hs, 1:hs) = x(1:hs, 1:hs);
x2(end-hs+1:end, 1:hs) = x(end-hs+1:end, 1:hs);
x2(1:hs, end-hs+1:end) = x(1:hs, end-hs+1:end);
x2(end-hs+1:end, end-hs+1:end) = x(end-hs+1:end, end-hs+1:end);

x2 = ifftshift(x2);

y = real(side2 * x2(:));
y = [y; imag(side2 * x2(:))];
end
