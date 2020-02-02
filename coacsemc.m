rounds = 16;
qbarrier = [];
nzpenalty = [];
iters = [];
tols = [];

step = 1;

% Prepare settings for the different continuation levels
for i=0:rounds
  step = step + i;
  val = 2^-(step - 4);
  qbarrier = [qbarrier val val];
  nzpval = 1e4 / sqrt(val);
  nzpenalty = [nzpenalty nzpval nzpval];
  iters = [iters 250 1e5];
  tolval = 1e-7;
  tols = [tols tolval tolval];
end

nzpenalty(:) = 1e15;
nzpenalty(1) = 1000;


rs = []
vs = []
    rs = [rs r];
tic
    [v, b] = healernoninv(r, mask, zeros(96,96,96), [], 'AT', length(qbarrier), qbarrier, nzpenalty, iters, tols);
toc
    vs = [vs v];

save coacsemcres -v7.3
