function [proxop, diffxt, level, xlevel] = createproxop(diffx, penalty, ourlinp)
  diffxt = ourlinp(diffx, 2); 
  
  level = -diffxt;
  level(penalty == 0) = 0;
  xlevel = ourlinp(level, 1);
  proxop = zero_tolerant_quad(penalty, -diffxt-level, -diffxt);
end