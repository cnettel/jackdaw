rounds = 24;
qbarrier = [];
nzpenalty = [];
iters = [];
tols = [];

% Prepare settings for the different continuation levels
for i=0:rounds
  val = 4^-(i - 2)
  qbarrier = [qbarrier val];
  nzpval = 1e8;% / val;
  nzpenalty = [nzpenalty nzpval];
  iters = [iters 3e2];
  tolval = val * 1e-14;
  tols = [tols tolval];
end

rs = []
vs = []
    rs = [rs r];
tic
    [v, b] = healernoninv(r, mask, zeros(96,96,96), [], 'AT', length(qbarrier), qbarrier, nzpenalty, iters, tols);
toc
    vs = [vs v];

save coacsemcres -v7.3
