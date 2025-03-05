%% Purpose: 
% Demonstrate 

%% Load data
% load('hipSliceAndNoise.mat')
load('hipSlice.mat')
load('noiseForHipSlice.mat')

[Nx, Ny, Nz, Nc, Ns] = size(raw); 
[Nt, ~] = size(noise);
%%
imRaw = zeros(size(raw));

figure,
for chiter = 1 : Nc
    imRaw(:, :, :, chiter) = ifftnc(raw(:, :, :, chiter));
    subplot(5, 7, chiter)
    imshow(flip(abs(imRaw(:, :, :, chiter)), 1), [])
    title(strcat('Channel #', 32, num2str(chiter)))
end




imRawsos = func_sqSOS(imRaw, noise);

figure, 
imshow(flip(abs(imRawsos), 1), [])
title('Sq. SOS. Recon.')
%% Now Walsh Adaptive Coil Combine

figure,
for chiter = 1 : Nc
    subplot(5, 7, chiter)
    imshow(flip(angle(imRaw(:, :, :, chiter)), 1), [-pi, pi])
    title(strcat('Channel #', 32, num2str(chiter)))
end
%%  Walsh Recon.

% The input imRaw is of size Nx x Ny x Nz x Nc x Nm where Nm is the number
% of echoes.  Earlier i mentioned imRaw is Nx x Ny x Nz x Nc x Ns, so make
% sure you permute to place the echo dimension where the Ns (slice)
% dimension is and run the recon slice-by-slice. So for this section the
% fourth dimension of imRaw is "echo" (and that's not a big deal because
% this is a single echo and single slice dataset). 

patch = [5, 5, 1, size(imRaw, 4)]; 
[imRawcc] = func_WalshMethod(imRaw, noise, patch) ;

figure,
subplot(1, 2, 1)
imshow(flip(abs(imRawcc), 1), [])
subplot(1, 2, 2)
imshow(flip(angle(imRawcc), 1), [-pi, pi])
