%% Purpose:
% This is just to demonstrate how SENSE works. 
% This is just done for simple 2D Cartesian imaging.  therefore the
% unaliasing is just along 1D, the phase-encoding direction. 
%%
load('brain_8ch.mat'); NcSets = 1; % number of channel sets. 
% load('brain_alias_8ch.mat'); NcSets = 2; % number of channel sets. (for
% e-spirit ... see my "simplified E-SPIRit" code.  
raw = permute(DATA, [1, 2, 4, 3]); %x, y, z, ch
[Nx, Ny, Nz, Nc] = size(raw);

R = 2;
rawUs = zeros(size(raw));
rawUs(:, 1 : R : end, :, :) = raw(:, 1 : 2 : end, :, :);
%% Show the original (fully sampled) and undersampled (US) images. 
% We must coil combine them first.  I use Walsh's method.  See my walsh
% implementation in the "coilCombine" directory. 
Npatchx = 15; Npatchy = 15; Npatchz = 1; Npatchm = 1;

imRaw         = zeros(size(raw        ));
imRawUs       = zeros(size(raw        ));
% imDiff        = zeros(size(raw        ));
for channelIter = 1 : Nc
    imRaw        (:, :, :, channelIter) = ifftnc(raw        (:, :, :, channelIter));
    imRawUs      (:, :, :, channelIter) = ifftnc(rawUs      (:, :, :, channelIter));
end

[imRaw_cc  , ~, ~, ~, ~] = func_WalshMethod(imRaw  ,        [], [Npatchx, Npatchy, Npatchz, Npatchm]);
[imRawUs_cc, ~, ~, ~, ~] = func_WalshMethod(imRawUs,        [], [Npatchx, Npatchy, Npatchz, Npatchm]);

imRaw_sos   = func_sqSOS(imRaw  , []);
imRawUs_sos = func_sqSOS(imRawUs, []);

mask_im   = imRaw_cc   > 0.06 * max(imRaw_cc  (:));
mask_imus = imRawUs_cc > 0.06 * max(imRawUs_cc(:));


figure,
subplot(2, 3, 1)
imshow(abs(imRaw_cc), [])
title('original fully sampled image')
subplot(2, 3, 2)
imshow(angle(imRaw_cc), [-pi, pi])
title('original fully sampled phase')
subplot(2, 3, 3)
imshow(log(abs(raw(:, :, 1, 1))), [])
title('original fully sampled k-space')

subplot(2, 3, 4)
imshow(abs(imRawUs_cc), [])
title('original fully sampled image')
subplot(2, 3, 5)
imshow(angle(imRawUs_cc), [-pi, pi])
title('original fully sampled phase')
subplot(2, 3, 6)
imshow(log(abs(rawUs(:, :, 1, 1))), [])
title('original fully sampled k-space')

%% Generate Sensitivity maps.
% To unalias the signal, we must generate sensitivity maps. A common
% approach is to divide the channel images by the sq. sos. image.  I implemented a
% simple least-squares version of E-SPIRiT, which I call "Simplified
% E-SPIRiT."  I use this to generate the sensitivity maps. I will discuss
% this in a separate main file, along with all of the inputs that are discussed 
% in this section.  but for now just treat it as a funciton
% that gives sensitivity maps: 

% -------------------------------------------------------------------------
% Inputs for Simplified E-Spirit. I will explain them in a different main
% file. 
kSize = [7, 7, 1];
Nkx = kSize(1); Nky =  kSize(2); Nkz = kSize(3); numKernelShapes = 1;
kernelShape = ones(Nkx, Nky, Nkz, numKernelShapes);
% for kernelShapeIter = 1 : numKernelShapes
%     kernelShape(:, 2, :, kernelShapeIter) = 0;
% %     kernelShape(:, 1, :, kernelShapeIter) = 0;
% end
kSolveIndices = zeros(3, numKernelShapes);
kSolveIndices(:, 1) = [4, 4, 1];
kernelShape(kSolveIndices(1), kSolveIndices(2), kSolveIndices(3), 1) = 0;
[ kernels, kernels_ones, kernels_solve] = func_generate_kernels(kernelShape,kSolveIndices);

[weightsKernels, targetColumnIndex, sourceColumnIndex, ...
    weightsKernels_decompressed, weightsKernels_reformatted, ...
    Ac_sourceKernels_harmonicsx_reformatted, ...
          Ac_sourceKernels_harmonicsy_reformatted, ...
          Ac_sourceKernels_harmonicsz_reformatted]  = ...
    func_getWeights_II(raw, kernelShape,kSolveIndices);
[eigValuesM, eigVectorsM,mask_eig, mask_im] = ...
    func_simplified_ESPIRiT(raw, kernelShape, kSolveIndices);


% The eigenvectors corresponding to eigenvalue 1 is the sense map.  I will
% explain this in my main file discussion simplified e-spirit. 
eigValue = 1;
if NcSets == 1
    eigValue = 1;
    % sensemap = mask_eig(:, :, 1, eigValue) .* eigVectorsM(:, :, 1, :, eigValue);
    sensemap = eigVectorsM(:, :, 1, :, eigValue);
elseif NcSets == 2
    sensemap = zeros(Nx, Ny, Nz, Nc * NcSets);
    sensemap(:, :, :, 1 : Nc     ) = eigVectorsM(:, :, 1, :, 1);
    sensemap(:, :, :, Nc+1 : 2*Nc) = eigVectorsM(:, :, 1, :, 2);
end

%%
senseRecon = SENSE_fa1D(imRawUs, S_2, 2);

figure,
subplot(2, 2, 1)
imshow(abs(senseRecon), [])
title('sense recon magnitude')

subplot(2, 2, 2)
imshow(angle(senseRecon), [-pi, pi])
title('sense recon phase')

subplot(2, 2, 3)
imshow(abs(imRaw_cc), [])
title('magnitude of Walsh recon of fully sampled image ')

subplot(2, 2, 4)
imshow(angle(imRaw_cc), [-pi, pi])
title('phase of Walsh recon of fully sampled image')

