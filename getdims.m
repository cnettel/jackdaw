function [dims, side2, fullsize, pshape, cshape] = getdims(pattern)
side2 = size(pattern,1);
fullsize = numel(pattern);

if fullsize == side2^3
  dims = 3;
  pshape = [side2 side2 side2];
  cshape = [side2 side2 * 2 side2];
elseif fullsize == side2^2
  dims = 2;
  pshape = [side2 side2];
  cshape = [side2 side2 * 2];
else
  'Unknown dimensionality'
end

end