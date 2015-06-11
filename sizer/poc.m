index = 1755;
bild = bild70(:,:,index);
lambda = lambda70(:,:,index);
bild59 = zeros(414,414);
lambda59 = zeros(414,414);
bild59(:) = bild(:);
lambda59(:) = lambda(:);


% bilden = fft2(bild59 - lambda59);
% bilden(7:408,:) = 0;
% bilden(:,7:408) = 0;
% bilden = ifft2(bilden);
% bilden = real(bilden) + lambda59;
bilden = bild59 + 1e-9;

mask3 = zeros(414,414);
mask3(1:end,1:end) = 1;
mask3 = mask3 .* (lambda > 0);
bilden = bilden .* mask3;
mask4 = find(1 - mask3);
step = 1e-0;
a= 0;

oldbilden2 = bilden;

send = 5;
while true
for i = 0:send
     g = 1./bilden .* bild59 - 1;
      
     g = (g .* (g > 0)) + (g .* (g <= 0) .* (bilden >= 1e-9));
     g(mask4) = 0;
     
%      j = 1;
%      while true
%      gfN = fft2(g);
%      gf2N = gfN;
%      gf2N(7:409,:) = 0;
%      gf2N(:,7:409) = 0;
%      val = ifft2(gf2N);
%      uncut = val;
%      val = real(val);
%      val = (val .* (val > 0)) + (val .* (val <= 0) .* (bilden > 1e-8));
%      gold = g;
%      g = val;
%      if (norm(uncut - g,1) * 1e8 < norm(g,1))
%          break
%      end
%      if mod(j,1000) == 0
%          [j/1000 log10(norm(uncut - g, 1)/norm(g,1))]
%      end
%      j = j + 1;
%      end
     val = g;
     oldbilden = bilden;
     bilden = bilden + val / max(max(abs(val)))*step;
     stepnow = bilden;
     
     j = 1;
     while true
         oldstep = bilden;
     bilden = fft2(bilden - lambda59);
     bilden(7:409,:) = 0;
     bilden(:,7:409) = 0;
     bilden = ifft2(bilden);
     %bilden = bilden;
     bilden = real(bilden) + lambda59;

     unmasked = bilden;
     bilden(mask3 > 0) = stepnow(mask3> 0);
     mask = bilden >= 1e-9;
     bilden = bilden .* mask + (1e-8) .* (1 - mask);
     
%      bilden = fft2(bilden - lambda59);
%      bilden(7:408,:) = 0;
%      bilden(:,7:408) = 0;
%      bilden = ifft2(bilden);
%      %bilden = bilden;
%      bilden = real(bilden) + lambda59;
%      
%      minval = min(bilden(:));
%      add = minval -1e-8;
%      
%      if add < 0
%          bilden = bilden - add;
%      end
     diff = norm(bilden - oldstep, 1);
     if diff * 1e8 < norm(bilden,1)
         break
     end
     if mod(j,1000) == 0
         [j / 1000 log10(diff/norm(bilden,1))]
     end
     j = j + 1;
     end
     
     j = 1
     while true
     bilden = fft2(bilden - lambda59);
     bilden(7:409,:) = 0;
     bilden(:,7:409) = 0;
     bilden = ifft2(bilden);
     %bilden = bilden;
     bilden = real(bilden) + lambda59;

     unmasked = bilden;
     mask = bilden >= 1e-9;
     bilden = bilden .* mask + (1e-8) .* (1 - mask);
     
%      bilden = fft2(bilden - lambda59);
%      bilden(7:408,:) = 0;
%      bilden(:,7:408) = 0;
%      bilden = ifft2(bilden);
%      %bilden = bilden;
%      bilden = real(bilden) + lambda59;
%      
%      minval = min(bilden(:));
%      add = minval -1e-8;
%      
%      if add < 0
%          bilden = bilden - add;
%      end
     diff = norm(bilden - unmasked, 1);
     if diff * 1e8 < norm(bilden,1)
         break
     end
     if mod(j,1000) == 0
         [j / 1000 log10(diff/norm(bilden,1))]
     end
     j = j + 1;
     end
end
diff = norm(bilden - unmasked, 1);
masked = bilden;
unmasked2 = unmasked;
diff

if mod(a,2) == 0
    imagesc(log(real(bilden)), [-3 5])
else
    imagesc(bilden -oldbilden2, [-0.1 0.1])
end

a = a + 1;

step1 = norm(bilden - oldbilden, 1);
step100 = norm(bilden - oldbilden2, 1);
% if step100 < step1 * send * 0.9
%    step = step / 2;
%    bilden = oldbilden2;
% end
% if step100 > step1 * (send + 0.9)
%     step = step * 1.1;
% else
%     step = step / 1.1;
%     bilden = oldbilden2;
% end
step
[step1 step100]
oldbilden2 = bilden;
pause(1)


end