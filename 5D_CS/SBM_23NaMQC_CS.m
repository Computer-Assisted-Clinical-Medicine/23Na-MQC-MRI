%% DBM_23NaMQC_CS
% based on: 
%   1) Goldstein et al.,2009, https://doi.org/10.1137/080725891
%   2) Montesinos et al., 2014, https://doi.org/10.1002/mrm.24936
%   5D CS was proposed by Licht et al., 2023, doi: 10.1002/mrm.29902
function [u] = SBM_23NaMQC_CS(use_gpu, f, nInner, nBreg, alpha_xyz, alpha_TE, alpha_Rep)
% f: undersampled k-space
% nInner: # of inner iterations (enforcing sparsity) 
% nBreg: # of outer Bregman iterations (enforcing data fidelity)
% alpha_xyz: spatial sparsity threshold
% alpha_TE: multi-echo sparsity threshold
% alpha_Rep: phase-cycling sparsity threshold

dims = size(f); 
R = single(f ~= 0.0);   % generates mask with desired undersampling
norm_Factor = getNormFactor(squeeze(R(:,:,:,1,:)),squeeze(f(:,:,:,1,:)));
f=(f*norm_Factor);

% Reconstruction parameters
lambda =1;         % weighting parameter for sparsity
mu = 1;            %
gamma = 1;      %1e-4   
 
murf = ifft3c((mu*f));
data_type = 'single';

% Build Kernels
% 3D Kernel
uker(1,1,1) = 8; uker(1,2,1)=-1; uker(2,1,1)=-1;
uker(dims(1),1,1)=-1; uker(1,dims(2),1)=-1;
uker(1,1,2)=-1; uker(1,1,dims(3))=-1;
uker        =   (lambda)*fftn(uker);
uker        =   uker + gamma + mu*R;

    % Pre-allocation (gpu computing useed to speed up reconstruction)
if use_gpu == 1
    x = gpuArray(zeros(dims,data_type));
    y = gpuArray(zeros(dims,data_type));
    z = gpuArray(zeros(dims,data_type));
    TE = gpuArray(zeros(dims,data_type));
    Rep = gpuArray(zeros(dims,data_type));
    
    % Bregman updates
    bx = gpuArray(zeros(dims,data_type));
    by = gpuArray(zeros(dims,data_type));
    bz = gpuArray(zeros(dims,data_type));
    bTE = gpuArray(zeros(dims,data_type));
    bRep = gpuArray(zeros(dims,data_type));
else
    x = zeros(dims,data_type);
    y = zeros(dims,data_type);
    z = zeros(dims,data_type);
    TE = zeros(dims,data_type);
    Rep = zeros(dims,data_type);
    
    % Bregman updates
    bx = zeros(dims,data_type);
    by = zeros(dims,data_type);
    bz = zeros(dims,data_type);
    bTE = zeros(dims,data_type);
    bRep = zeros(dims,data_type);
end

u=zeros(dims);
f0 = f;

for outer = 1:nBreg
    for inner = 1:nInner
        Sparse_Comps= 1*lambda*Dxt(x-bx,dims)+1*lambda*Dyt(y-by,dims)+1*lambda*Dzt(z-bz,dims) +0.2*ifftc(TE-bTE,4)+1.9*ifftc(Rep-bRep,5);

        rhs = murf+gamma*(u)+ Sparse_Comps;% + u_LR;%+...

        u = gather(ifftn((fftn(rhs)./uker)));
        
        % Sparsity transforms
        dx = Dx(gpuArray(u),dims);
        dy = Dy(gpuArray(u),dims);
        dz  =Dz(gpuArray(u),dims);
        dTE = fftc(u,4);
        dRep = fftc((u),5);
                
        % Shrinkage operations
        [x,y,z] = (shrink3( (dx+bx), (dy+by),(dz+bz), (alpha_xyz)));
        [TE] = gather(shrink1( (dTE+bTE), alpha_TE));  
        [Rep] = gather(shrink1( (dRep+bRep), alpha_Rep));  
  
        % Update Bregman parameters
        bx = bx+dx-x;
        by = by+dy-y;
        bz = bz+dz-z;
        bTE = bTE+dTE-TE;
        bRep = bRep+dRep-Rep;
 
  
%  clear dx dy dz dTE dRep dxBM dyBM dzBM dTEBM dRepBM dChanBM dxBMObject dyBMObject dzBMObject      
        
    end% end inner loop

    % update solution (data fidelity)
    t1= R.*fft3c(u);
    f = f+f0-t1;
    murf = ifft3c((mu*R.*f));

end %end outer loop

  u=u/(norm_Factor);    % undo normalization

return;

% Shrinkage method
function [xs,ys,zs] = shrink3(x,y,z,lambda)
s = sqrt((x.*conj(x))+(y.*conj(y))+(z.*conj(z)));
ss = s-lambda;
ss = ss.*(ss>0);
s = s+(s<lambda);
ss = ss./s;
clear s
xs = ((ss.*x));
clear x
ys = ((ss.*y));
clear y
zs = ((ss.*z));
return;

function [TEs] = shrink1(TE,lambda)
s = sqrt(TE.*conj(TE));
ss = s-lambda;
ss = ss.*(ss>0);
s = s+(s<lambda);
ss = ss./s;
clear s
TEs = ((ss.*TE));

return;


function nF = getNormFactor(mask,data)
% keyboard
nF = 1/norm(data(:)/size(mask==1,1));    %*size(mask==1,3))

function d = Dx(u,n)    
d = gpuArray(zeros(n,'single'));
d(:,2:end,:,:,:) = (u(:,2:end,:,:,:)-u(:,1:end-1,:,:,:));
d(:,1,:,:,:) = (u(:,1,:,:,:)-u(:,end,:,:,:));
return

function d = Dxt(u,n)   
d = gpuArray(zeros(n,'single'));
d(:,1:end-1,:,:,:) = u(:,1:end-1,:,:,:)-u(:,2:end,:,:,:);
d(:,end,:,:,:) = u(:,end,:,:,:)-u(:,1,:,:,:);
return

function d = Dy(u,n)
d = gpuArray(zeros(n,'single'));
d(2:end,:,:,:,:) = u(2:end,:,:,:,:)-u(1:end-1,:,:,:,:);
d(1,:,:,:,:) = (u(1,:,:,:,:)-u(end,:,:,:,:));
return

function d = Dyt(u,n)
d = gpuArray(zeros(n,'single'));
d(1:end-1,:,:,:,:) = u(1:end-1,:,:,:,:)-u(2:end,:,:,:,:);
d(end,:,:,:,:) = u(end,:,:,:,:)-u(1,:,:,:,:);
return

function d = Dz(u,n)
d = gpuArray(zeros(n,'single'));
d(:,:,2:end,:,:) = u(:,:,2:end,:,:)-u(:,:,1:end-1,:,:);
d(:,:,1,:,:) = (u(:,:,1,:,:)-u(:,:,end,:,:));
return

function d = Dzt(u,n)
d = gpuArray(zeros(n,'single'));
d(:,:,1:end-1,:,:) = u(:,:,1:end-1,:,:)-u(:,:,2:end,:,:);
d(:,:,end,:,:) = u(:,:,end,:,:)-u(:,:,1,:,:);
return