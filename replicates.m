load invicosa68

rounds = 48;
qbarrier = [];
nzpenalty = [];
iters = [];
tols = [];

% Prepare settings for the different continuation levels
for i=0:rounds
  val = 2^-(i - 3);
  qbarrier = [qbarrier val];
  nzpval = 1e8 / val;
  nzpenalty = [nzpenalty nzpval];
  iters = [iters 3e2];
  tolval = val * 1e-14;
  tols = [tols tolval];
end

rs = []
vs = []

for qq2 = 1:50;
  
  banner = sprintf('##################### REPLICATE %d\n', qq2)
  r2 = poissrnd(r3b);
  r(r>=0) = r2(r>=0);
    
    rs = [rs r];
tic
    [v, b] = healernoninv(r, mask, zeros(256,256), [], 'AT', length(qbarrier), qbarrier, nzpenalty, iters, tols);
toc
    vs = [vs v];
end


save invicosa74 -v7.3
