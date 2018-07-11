function [X] = halfboundedlinesearch(Y, f)

factor = 1;
prevval2 = 0;
prevval = 0;
lastval = f(Y * factor);
minoverall = 0;
minval = f(minoverall);
while true
factor = factor * 2;
newval = f(Y * factor);
if newval >= lastval
  break
end
if newval < minval
    minoverall = factor;
    minval = newval;
end

end

lo = 0;
hi = factor;

for i = 1:100
  diff = hi - lo;
  poses = [lo + diff/3 lo + 2 * diff/3];
  realvals = [f(Y * poses(1)) f(Y * poses(2))];
  [a,b] = min(realvals);
  if realvals(1) == realvals(2)
      break
  end
  if b == 1
    hi = poses(2);
  elseif b == 2
    lo = poses(1);
  end
  if realvals(b) < minval
      minval = realvals(b);
      minoverall = poses(b);
  end
  if minoverall < lo
      lo = minoverall;
      break
  end  
  hi = max(minoverall,hi);
lastval = realvals(b);
end
X = Y * lo;
lo
hi