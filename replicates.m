load reference

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
  iters = [iters 1e2];
  tolval = val * 1e-14;
  tols = [tols tolval];
end

numrep = 1;
rs = cell(50,256,256);
vs = cell(50,256,256);


for qq2 = 1:50      
  banner = sprintf('##################### PREP REPLICATE %d\n', qq2)
  r2 = poissrnd(r3b);
  r(r>=0) = r2(r>=0);
    
  rs{qq2} = r;
end


parfor qq2 = 1:numrep
  banner = sprintf('##################### REPLICATE %d\n', qq2)
  r = rs{qq2}; 
tic
    [v, b] = healernoninv(r, mask, zeros(256,256), [], 'AT', length(qbarrier), qbarrier, nzpenalty, iters, tols);
toc
    vs{qq2} = v;
end

rsold = rs;
vsold = vs;
rs = [];
vs = [];
for qq2 = 1:numrep
    rs = [rs rsold{qq2}];
    vs = [vs vsold{qq2}];
end

clear rsold vsold

save invicosa72orig -v7.3
