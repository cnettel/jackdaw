function [factor, basepenalty] = createwindows(side2, fullsize)
filter = hann(side2, 'periodic');
%filter = filter * filter';
filter = fftshift(filter);

side3 = side2;



filter1 = repmat(filter, [1 side2 side2]);
filter2 = ones(1,side2,1);
filter2(:) = filter(:);
filter2 = repmat(filter2, [side2 1 side2]);

filter3 = ones(1,1,side2);
filter3(:) = filter(:);
filter3 = repmat(filter3, [side2 side2 1]);
filter = filter1 .* filter2 .* filter3;

purefactor = reshape(filter, fullsize, 1);
purefactor = purefactor .* purefactor;

				% Perform Hann windowing on our penalty matrix
maskinshape = reshape(mask, side2, 2 * side2, side2);
basepenalty = double((maskinshape(1:side2,1:side2,1:side2 )> 0) + ...
                 1j * (maskinshape(1:side2,side2 + (1:side2), 1:side2) > 0));
				%basepenalty = double(reshape((mask(1:fullsize) > 0) + 1j * (mask(fullsize + 1:2*fullsize) > 0),side2,side2,side2));
				%basepenalty = double(mask > 0);
basepenalty = ifftshift(ifftn(fftn(fftshift(basepenalty)) .* reshape(purefactor,side2,side2,side2)));
				%basepenalty = ourlinp(ourlinp(basepenalty, 1) .* purefactor, 2);

				% Filter out numerical inaccuracies
basepenalty = [real(basepenalty) imag(basepenalty)];
basepenalty = basepenalty(:);
basepenalty(basepenalty < 1e-8) = 0;

basepenalty = 1 - basepenalty;

				% Filter out numerical inaccuracies
basepenalty(basepenalty < 1e-8) = 0;

basepenalty = reshape(basepenalty, 2 * fullsize, 1);




filter = hann(side2);
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

factor = reshape(filter, fullsize, 1);
factor = factor .* factor;

end