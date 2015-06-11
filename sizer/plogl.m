function [ logl ] = plogl( bilden, pc)
mask = bilden < 1e-9;
bilden = bilden .* (1-mask) + mask .* 1e-9; 
terms = bilden .* pc - bilden;
logl = sum(terms(:)) + sum(mask(:)) * 1e9 - bilden(mask);
end

