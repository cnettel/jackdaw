function [factor, basepenalty] = createwindows(pattern, mask)

[dims, side2, fullsize, pshape, cshape] = getdims(pattern);

function [factor] = createfilter(filter, pshape, side2, fullsize)
  shape1 = pshape;
  shape1(1) = 1
  filter1 = repmat(filter, shape1);

  shapeb = pshape;
  shapeb(:) = 1;
  shapeb(2) = pshape(2);
  filter2 = ones(shapeb);
  filter2(:) = filter(:);
  shape2 = pshape;
  shape2(2) = 1;
  filter2 = repmat(filter2, shape2);
  size(filter1)
  size(filter2)
  newfilter = filter1 .* filter2;

  if dims == 3
    filter3 = ones(1,1,side2);
    filter3(:) = filter(:);
    filter3 = repmat(filter3, [side2 side2 1]);
    newfilter = newfilter .* filter3;
  end
  factor = reshape(newfilter, fullsize, 1);
end

filter = hann(side2, 'periodic');
%filter = filter * filter';
filter = fftshift(filter);

purefactor = createfilter(filter, pshape, side2, fullsize);
purefactor = purefactor .* purefactor;
  


				% Perform Hann windowing on our penalty matrix
maskinshape = reshape(mask, cshape);
if dims == 3
  basepenalty = double((maskinshape(1:side2,1:side2,1:side2 )> 0) + ...
                 1j * (maskinshape(1:side2,side2 + (1:side2), 1:side2) > 0));
else
  basepenalty = double((maskinshape(1:side2,1:side2 )> 0) + ...
                 1j * (maskinshape(1:side2,side2 + (1:side2)) > 0));
end
				%basepenalty = double(reshape((mask(1:fullsize) > 0) + 1j * (mask(fullsize + 1:2*fullsize) > 0),side2,side2,side2));
				%basepenalty = double(mask > 0);
basepenalty = ifftshift(ifftn(fftn(fftshift(basepenalty)) .* reshape(purefactor, pshape)));
				%basepenalty = ourlinp(ourlinp(basepenalty, 1) .* purefactor, 2);

				% Filter out numerical inaccuracies
basepenalty = [real(basepenalty) imag(basepenalty)];
basepenalty = basepenalty(:);
basepenalty(basepenalty < 1e-8) = 0;

basepenalty = 1 - basepenalty;

				% Filter out numerical inaccuracies
basepenalty(basepenalty < 1e-8) = 0;

basepenalty = reshape(basepenalty, 2 * fullsize, 1);




filter = hann(side2, 'periodic');
%filter = filter * filter';
filter = fftshift(filter);

factor = createfilter(filter, pshape, side2, fullsize);

factor = factor + 1e-3;
factor = factor .* factor;

end