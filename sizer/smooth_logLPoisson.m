function op = smooth_logLPoisson(x)
% SMOOTH_LOGLPOISSON Log-likelihood of a Poisson: sum_i (-lambda_i + x_i * log( lambda_i) )
%   OP = SMOOTH_LOGLPOISSON( X )
%   returns a function that computes the log-likelihood function
%   of independent Poisson random variables with parameters lambda_i:
%
%       log-likelihood(lambda) = sum_i ( -lambda_i + x_i * log( lambda_i) )
%
%   where LAMBDA is the parameter of the distribution (this is unknown,
%    so it is the variable), and X is a vector of observations.
%
%   Note: the constant term in the log-likelihood is omitted.

error(nargchk(1,1,nargin));
op = tfocs_smooth( @smooth_llPoisson_impl );

function [ v, g ] = smooth_llPoisson_impl( lambda )

  realsize = size(x);
  indices = ~isnan(x);
  origlambda = lambda;
  lambda = lambda(indices);
  x2 = x(indices);

  if length(lambda) == 1, 
      lambda = lambda * ones(size(x2));
  elseif size(lambda) ~= size(x2),
      error('Parameters and data must be of the same size'),
  end
  
  if any( origlambda < realmin ) | any((lambda <= realmin) & (x2 > 0)) ,
      v = Inf;
      if nargout > 1,
          g = NaN * ones(realsize);
      end
  else
      loglambda = log(max(lambda,realmin));
      v = -(-tfocs_dot(lambda, ones(size(x2))) + tfocs_dot( x2, loglambda));
      vector = exp(1./(origlambda.^4 * 1e10));
      v = v + sum(vector) - length(vector(:));
      if nargout > 1,
          g = zeros(realsize);
          g(indices) = -(-1 + x2./lambda);          
                    g = g - vector ./ (origlambda.^4 * 1e10 / 4);

      end
  end
end

end

% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.
