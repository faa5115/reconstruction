
%%  Purpose:
% Illustrates that raw multi-channel data from MR systems are correlated.
% To treat them as independent receivers, they must be whitened. 
% This is useful for many reconstruction settings. 

%% Load Data:  has a single slice hip data and a noise scan.
load('hipSliceAndNoise.mat')

[Nx, Ny, Nz, Nc, Ns] = size(raw); 
[Nt, ~] = size(noise);
%% Show cross-channel correlation.
Rn = noise' * noise;
figure,
imagesc(abs(Rn))%imshow(abs(Rn), [])
%%  Demonstrate how to whiten.  (this is shown in func_whitenMatrix)
[V, D] = eig(Rn);% eigen value decomposition of noise correlation matrix
                                % noiseCorrMat * Vnc = Vnc * Dnc;

W = V * diag(diag(D).^(-0.5)) * V';
%% Whiten noise
noisew = noise * W;
Rnw = noisew'*noisew;

figure,
imagesc(abs(Rnw))
%% This section just demonstrates how we apply W to the raw data.
imRaw = zeros(size(raw));
imRaw_vect = zeros(Nx*Ny*Nz, Nc  );
imRaw_w = zeros(size(raw));
for chiter = 1 : Nc
    imRawiter = ifftnc(squeeze(raw(:, :, :, chiter, :)));
    imRaw(:, :, :, chiter, : ) = imRawiter;
    imRaw_vect(:, chiter) = imRawiter(:);
end

imRaw_w_vect(:, :) = imRaw_vect(:, :) * W;
for chiter = 1 : Nc
    imRaw_w(:, :, :, chiter) = ...
        reshape(imRaw_w_vect(:, chiter), [Nx, Ny, Nz]);

end

imRaw_w_sos = func_sqSOS(imRaw_w, []); 
% I put [] instead of the noise data because we already whitened it.

figure, imshow(flip(abs(imRaw_w_sos),1), [])