vs72super = h5read('vs72/phasing.h5','/super_images');
rs72f2super = h5read('rs72f2/phasing.h5','/super_images');
%vs72nwsuper = h5read('vs72nw/phasing.h5','/super_images');
rs72super = h5read('rs72/phasing.h5','/super_images');

save -v7.3 superimages vs72super rs72f2super rs72super
%vs72nwsuper rs72super
