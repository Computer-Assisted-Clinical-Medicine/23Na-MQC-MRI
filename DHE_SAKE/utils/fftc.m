function res = fftc(x,dim)
%res = ifftc(x,dim)
res = sqrt(size(x,dim))*ifftshift(fft(fftshift(x,dim),[],dim),dim);