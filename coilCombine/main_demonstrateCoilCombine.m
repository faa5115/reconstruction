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

calib =  raw(:, round(Ny/2) - 14 : round(Ny/2) + 13, :, :);
calibPad = zeros(size(raw));
calibPad(:, round(Ny/2) - 14 : round(Ny/2) + 13, :, :) = raw(:, round(Ny/2) - 14 : round(Ny/2) + 13, :, :);


% figure,
% for chiter = 1 : Nc
%     imRaw(:, :, :, chiter) = ifftnc(raw(:, :, :, chiter));
%     subplot(5, 7, chiter)
%     imshow(flip(abs(imRaw(:, :, :, chiter)), 1), [])
%     title(strcat('Channel #', 32, num2str(chiter)))
% end

imCalib = zeros(size(calib));
imCalibPad = zeros(size(calibPad));
for chiter = 1 : Nc
    imRaw(:, :, :, chiter) = ifftnc(raw(:, :, :, chiter));
    imCalib      (:, :, :, chiter) = ifftnc(calib      (:, :, :, chiter));
    imCalibPad   (:, :, :, chiter) = ifftnc(calibPad   (:, :, :, chiter));
end





imRawsos = func_sqSOS(imRaw, noise);

figure, 
imshow(flip(abs(imRawsos), 1), [])
title('Sq. SOS. Recon.')
%% Now Walsh Adaptive Coil Combine

figure,
for chiter = 1 : Nc
    subplot(5, 7, chiter)
    imshow(flip(abs(imRaw(:, :, :, chiter)), 1), [])
    title(strcat('Channel #', 32, num2str(chiter)))
end

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
%% SENSE recon:
senseMapsOption = 1; % I have two options in generating sensitivity maps.  
                     % Option 1 simply makes low resolution images from  
                     % the center k-space lines and divides each low
                     % resolution image by their square root sum of squares.
                     % Option 2 uses a method called E-SPIRiT which I will
                     % discuss in a later section. 
R = [1, 1];  % This is "acceleration factor" along ky and kz encoding which I will discuss in the parallel imaging section.  
             % We did not accelerate the acceleration in this example, so
             % the values for both is 1. 

[imRawSense, senseMaps] = func_SENSE(imRaw, imCalib, R, noise, senseMapsOption);

%%
figure,
subplot(1, 2, 1)
imshow(flip(abs(imRawSense), 1), [])
title('SENSE Recon. Magnitude')
subplot(1, 2, 2)
imshow(flip(angle(imRawSense), 1), [-pi, pi])
title('SENSE Recon. Phase')
%% show sense maps
figure,
for chiter = 1 : Nc
    subplot(5, 7, chiter)
    imshow(abs(senseMaps(:, :, :, chiter)), [])
end
%%

figure,
subplot(2, 2, 1)
imshow(flip(abs(imRawSense), 1), [])
title('SENSE Recon. Magnitude')
subplot(2, 2, 2)
imshow(flip(angle(imRawSense), 1), [-pi, pi])
title('SENSE Recon. Phase')
subplot(2, 2, 3)
imshow(flip(abs(imRawcc), 1), [])
title('Stochastic Matched Filter Magnitude')
subplot(2, 2, 4)
imshow(flip(angle(imRawcc), 1), [-pi, pi])
title('Stochastic Matched Filter Magnitude')
