%% my edits after 12/09/2024. 
% Differences: 
%   1. removed zeros on the edges. 
%   2. included echoes.
%function [imRaw_cc, eigMatrix, eigValues, wgtMatrix, alpMatrix] = func_WalshMethod(imRaw, noiseVect, patch) 
function [imRaw_cc] = func_WalshMethod(imRaw, noiseVect, patch) 

% imRaw — (correlated) multi-channel images. Nx x Ny x Nz x Nc x Nm (m is
% for echo times).
% Nz is samples in a slab.  not slices. 
% noiseVect — noise vectors:  Nc x number of sampled timepoints of a noise scan.
% patch  — Npatchx x Npatchy x Npatchz x Nm.  Nm is the number of echoes. 


[Nx, Ny, Nz, Nc, Nm] = size(imRaw);  
% Npatchx = patch(1); Npatchy = patch(2); Npatchz = patch(3);
if isempty(patch)

    % what we had before 12/09/2024
    %{
    Npatchx = 15; Npatchy = 15;
    if Nz < 15
        Npatchz = Nz;
    end
    %}

    NminIndicesInPatch = 225;
    % First look along the echo dimension.
    Npatchm = Nm;
    % If you already have enough along the echo dimension, there is no need
    % to draw correlation of channel sensitivites from neighboring voxels. 
    if Npatchm >= NminIndicesInPatch 
        Npatchx = 1; Npatchy = 1; Npatchz = 1;
    else
        % if you do not have enough along the echo dimension, you will need
        % to pool from neighboring voxels. 
        % i am using ceil instead of round just for convenience. the more,
        % the merrier.
        Npatchxyz = ceil(NminIndicesInPatch / Npatchm);
        NpatchxyzCubeRoot = Npatchxyz^(0.33333333333);
        if Nz < ceil(NpatchxyzCubeRoot) 
            % often you will not have this many samples along your slab. 
            Npatchz = Nz;
            Npatchxy = ceil(NminIndicesInPatch / (Npatchm * Npatchz));
            Npatchx = round(sqrt(Npatchxy)); 
            Npatchy = round(sqrt(Npatchxy));
        else
            Npatchz = round(NpatchxyzCubeRoot);
            Npatchx = round(NpatchxyzCubeRoot);
            Npatchy = round(NpatchxyzCubeRoot);
        end
    end

else
    Npatchx = patch(1); 
    Npatchy = patch(2); 
    Npatchz = patch(3); 
    Npatchm = patch(4);
end

% Identify channel with the that has the maximum intensity value. 
sumEachChannel = zeros(1, Nc);
for channelIter = 1 : Nc
    imRawCh = imRaw(:, :, :, channelIter);
    sumEachChannel(1, channelIter) = sum(imRawCh(:));
end
[~,maxIntensityChannel] = max(sumEachChannel(:));
%% Prepare patch:  only in the image dimensions. 
patchShape = ones(Npatchx, Npatchy, Npatchz);
patchShape(round(Npatchx/2), round(Npatchy/2), 1) = 0;

patchSolveIndices = zeros(3, 1);
patchSolveIndices(:, 1) = [round(Npatchx/2), round(Npatchy/2), 1];

[ patches, patches_ones, patches_solve] = ...
    func_generate_kernels(patchShape,patchSolveIndices);
disp('created patches')
%% Pad image matrix (and noise for our case too ...)

patchSolveX = patchSolveIndices(1, 1);
patchSolveY = patchSolveIndices(2, 1);
patchSolveZ = patchSolveIndices(3, 1);

imRawPad     = padarray(imRaw       , [patchSolveX - 1  , patchSolveY - 1  , patchSolveZ - 1  , 0], 'pre'  );
imRawPad     = padarray(imRawPad    , [Npatchx - patchSolveX, Npatchy - patchSolveY, Npatchz - patchSolveZ, 0], 'post' );

% imNoisePad     = padarray(imNoise       , [patchSolveX - 1  , patchSolveY - 1  , patchSolveZ - 1  , 0], 'pre'  );
% imNoisePad     = padarray(imNoisePad    , [Npatchx - patchSolveX, Npatchy - patchSolveY, Npatchz - patchSolveZ, 0], 'post' );
elRawData = single(reshape((1 : 1 : prod(size(imRaw, 1, 2, 3))), [size(imRaw, 1, 2, 3)]));

elRawDataPad     = padarray(elRawData       , [patchSolveX - 1  , patchSolveY - 1  , patchSolveZ - 1  , 0], 'pre'  );
elRawDataPad     = padarray(elRawDataPad    , [Npatchx - patchSolveX, Npatchy - patchSolveY, Npatchz - patchSolveZ, 0], 'post' );
disp('reached the part of the code that deals with edges.')
%% Create matrices whose rows will be Rs or Rn. 
%{
% fadilali this is what we had before 12/009/2024 .... before we included
% the echo dimension. 
[AImpad_mc, AImpadel]   = func_create_A(imRawPad      , patches_ones); % padded image.
% [ANspad_mc, ANspadel]   = func_create_A(imNoisePad    , patches_ones); % padded noise.
%}
[AelPad , AelPadel]     = func_create_A(elRawDataPad  , patches_ones);
% targetColumn = zeros(1, size(AelPad, 1));
patch_target = patches_solve(:, :, :, :);
disp('constructed matrix')


AImpad_mc = zeros(...
    (size(imRawPad,1) - (Npatchx - 1)) * ...
    (size(imRawPad,2) - (Npatchy - 1)) * ...
    (size(imRawPad,3) - (Npatchz - 1)),  ...
    Npatchx*Npatchy*Npatchz * Nm, Nc);

startIndex = 1; 
for echoiter = 1 : Nm
    [AImpad_mc_echoiter, AImpadel]   = func_create_A(imRawPad(:, :, :, :, echoiter), patches_ones); % padded image.
    AImpad_mc(:, startIndex : startIndex + Npatchx * Npatchy * Npatchz - 1, :) = AImpad_mc_echoiter;
    startIndex = startIndex + Npatchx * Npatchy * Npatchz;
end

targetColumnIndex = round( (Npatchx * Npatchy * Npatchz)/2);
targetColumn = AelPad(:, targetColumnIndex);

% determine the rows of the AImpad_mc matrix that has zeros (edges):
[rowsWithZeros, ~] = ind2sub(size(AelPad), find(AelPad(:) == 0));
rowsWithZeros = unique(rowsWithZeros(:));
%% Noise Correlation Matrix

% if size(noiseVect, 1) == Nc
%     Rn = noiseVect * noiseVect';
% else
%     disp('no noise vector')
%     Rn = eye(Nc);
% end

if ~isempty(noiseVect)
    disp('with noise vector')
    Rn =  noiseVect' * noiseVect;
else
    disp('no noise vector')
    Rn = eye(Nc);
end
disp('created noise correlation matrix')
%% Create signal correlation matrix for each pixel ...

disp('starting to create signal covariance matrices')
RsArray = pagemtimes(permute(AImpad_mc, [2, 3, 1]), 'ctranspose', ...
    permute(AImpad_mc, [2, 3, 1]),'none');
% RsArray = permute(RsArray, [3, 1, 2]); % only use this format if you want
% to debug with what what you have below (before 12/09/2024).
disp('about to compute RNInvRs')
RnInvRs = pagemtimes(pinv(Rn), RsArray); clear RsArray
disp('computed RNInvRs')
[~, S, V] = pagesvd(RnInvRs);
wgtMatrix = squeeze(V(:, 1, :)); clear V
disp('found RninvRs subspace')
% calculate the normalization factor of the weights:
%  alpMatrix 1/(sqrt(weights' * pinv(Rn) * weights )); Break into the following
%  parts:
% 1. 1/pagemtimes pinv(Rn) and wgtMatrix.
RnInvWgt = pagemtimes(pinv(Rn),  wgtMatrix); % inverse Rn times largest eigenvector done for each location. 
disp('multiply by largest eigenvector')
% 2. alpMatrix = 1/pagemtimes wgtMatrix * RnInvWgt:
wgtMatrix = permute(wgtMatrix, [1, 3, 2]);
RnInvWgt  = permute(RnInvWgt , [1, 3, 2]);
alpMatrix = pagemtimes(wgtMatrix, 'ctranspose', RnInvWgt, 'none');
alpMatrix = pageinv(alpMatrix);
disp('determine new weights')
%Now normalize the weights: 
normWgtMatrix_pre = pagemtimes(alpMatrix, wgtMatrix);
% Now multiply by phase of signal with max intensity. 
normWgtMatrix = pagemtimes(  normWgtMatrix_pre, ...
                exp(     -1i * angle(normWgtMatrix_pre(maxIntensityChannel, 1, :))     )    );
disp('normalized the weights and multiplied by the channel having the largest signal power')

% imRawReshaped = permute(reshape(imRaw, [Nx * Ny * Nz, Nm, Nc]), [3, 2, 1]);
% imRaw_cc = pagemtimes(imRawReshaped, normWgtMatrix);
% imRaw_cc = reshape(imRaw_cc, [Nx, Ny, Nz, Nm]);

disp('about to combine the channels')
imRaw_cc = zeros(Nx,  Ny, Nz, 1, Nm);
for echoiter = 1 : Nm
    imRawiter = permute(reshape(imRaw(:, :, :, :, echoiter), ...
                [Nx * Ny * Nz, Nc]), [3, 2, 1]);

    imRaw_cciter =  pagemtimes(imRawiter, normWgtMatrix);
    imRaw_cc(:, :, :, 1, echoiter) = reshape(imRaw_cciter, [Nx, Ny, Nz]);
end
disp('finished adapive coil combine')
% below is what we had before 12/09/2024
%{
RsArray = zeros(size(AImpad_mc,1), Nc, Nc);

tic
for locIter = 1 : length(targetColumn)
    sM = squeeze(AImpad_mc(locIter, :, :));
    RsArray(locIter, :, :) = sM'*sM;
end
toc

eigMatrix = zeros(size(AImpad_mc,1), Nc, Nc);      % store eigenvectors if you want.  
eigValues = zeros(size(AImpad_mc,1), Nc, Nc);      % store eigenvalues  if you want.  
wgtMatrix = zeros(size(AImpad_mc,1), Nc, 1 );      % store eigvect of max eigvalue . 
alpMatrix = zeros(size(AImpad_mc,1), 1     );      % normalizing term              .
norm_wgtMatrix = zeros(size(AImpad_mc,1), Nc, 1 ); % normalized eigvect. 
tic
for locIter = 1 : length(targetColumn)
    Rs = squeeze(RsArray(locIter, :, :));
    [V, D] = eig(pinv(Rn) * Rs);
    eigMatrix(locIter, :, :) = V;
    eigValues(locIter, :, :) = D;
    listEigValuesMagnitude = abs(diag(D));
    maxIndex = find(listEigValuesMagnitude == max(listEigValuesMagnitude));
    wgtMatrix(locIter, :) = V(:, maxIndex);
    alpMatrix(locIter,:) = 1/(sqrt(squeeze(V(:, 1))' * pinv(Rn) * squeeze(V(:, 1)) ));

    norm_wgtMatrix(locIter,:) = wgtMatrix(locIter, :) * alpMatrix(locIter,:);
    vect = squeeze(norm_wgtMatrix(locIter,:));
    norm_wgtMatrix(locIter,:) = vect .* exp(-1i*angle(vect(maxIntensityChannel))); 
end
toc
%% reconstruct the coil combined image. 


% phases2 = repmat(exp(-1i * angle(imRaw(:, :, :, maxIntensityChannel))), [1, 1, 1, Nc]);
% imRaw_phases = imRaw .* phases2;

imRaw_cc = zeros(Nx, Ny, Nz);

for locIter = 1 : length(targetColumn)
    %[row, col] = find(elRawData == locIter);
    [row, col, sl] = ind2sub([Nx, Ny, Nz], locIter);
    channelVector = squeeze(imRaw(row, col, sl, :));
    imRaw_cc(row, col, sl) =  squeeze(norm_wgtMatrix(locIter, :)) * channelVector;

end
%}