function op = zero_tolerant_quad( P, p2, p3 )

  P = P(:); % make it a column vector
  op = @(varargin)smooth_quad_diag_matrix( P, p2, p3, varargin{:} );
  
  
  function [ v, g ] = smooth_quad_diag_matrix( p, p2, p3, origx, t )
    pm = max(p(:));
    x = origx;
    x = x - p2;
    
    % Can be added to make non-negativity constraint
    % Desn't jive well with Hann windowing
    %p(x<0) = pm;
    mask = p == 0;
    x(mask) = origx(mask);
    p2(mask) = 0;
    switch nargin
      case 4,
        g = p .* x;
        v = 0.5 *  sum( x.*g - p.*p3.*p3 );
      case 5,
        prevx = x;
        x = (1./(t*p+1)) .* (x);
        
        g = x + p2; % for this case, "g" is NOT the gradient        
        v = 0.5 * sum( p.*x .* x - p.*p3.*p3);
	end
	
% Part of Jackdaw by Carl Nettelblad.

% Loosely based on:
% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.

% Copyright (c) 2013, California Institute of Technology and CVX Research, Inc.
% All rights reserved.

% Contributors: Stephen Becker, Emmanuel Candes, and Michael Grant.
% Based partially upon research performed under the DARPA/MTO Analog-to- 
% Information Receiver Development Program, AFRL contract #FA8650-08-C-7853.

% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are
% met:

% 1. Redistributions of source code must retain the above copyright
% notice, this list of conditions and the following disclaimer.

% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
 
% 3. Neither the names of California Institute of Technology
% (Caltech), CVX Research, Inc., nor the names of its contributors may
% be used to endorse or promote products derived from this software
% without specific prior written permission.

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
