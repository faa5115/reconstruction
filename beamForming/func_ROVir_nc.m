function [rawVs,W, Vs, ds, SIR_array] = func_ROVir_nc(raw, ref, Smask, Imask, eigNum) 
% This is a more generalized version of ROVir.  "_nc" refers to it being generalized 
% for non Cartesian k-space samples.  
% The weights will be applied to the non Cartesian data. 
%
% If you want whitened data, whiten before the reconstruction.
% raw is multichannel image: Nx x Ny x Nz x Nc.  imRaw is the image of the
% individual channels of raw.
% ref is the calibration data.  % Nxc x Nyc x Nzc x Nc. 
% Smask is a binary mask of the signal region you want. size is Nx x Ny x Nz.
% Imask (optional) binary mask of the signal region we want to suppress.
% if it has an empty input, it will simply be 1 - Smask.  
% eigNum is the number of virtual channels (eigenvectors of the matrix). 
% outputs:  imRawVs is the set of virtual channel images:  Nx x Ny x Nz x Nvc
% W the selected eigenvectos for virtual channel weights. (sorted on
% descending order of corresponding eigenvalues).
% Vs the total set of eigen vectors sorted on descending order of
% corresponding eigenvalues.
% ds sorted eigenvalues. 
if isempty(Imask)
    Imask = 1 - Smask;
    disp('empty Imask')

else
    disp('input Imask')
end

 [Nxc, Nyc, Nzc, Nc] = size(ref); % for this code, the size of ref and Smask should have the same satial dimensions).
 [Nxm, Nym, Nzm] = size(Smask);  
 % [Nx, Ny, Nz, Nc] = size(raw);
 Nro = size(raw, 1); Nspokes = size(raw, 2); Nz = size(raw, 3);
 
 % ------------------------------------------------------------------------------------------------
 % From previous version. 
 % % refPad = padarray(ref, [round((Nx-Nxc)/2) round((Ny-Nyc)/2) round((Nz-Nzc)/2) ], 0, 'both');
 % if (Nz > Nzc) % standard scenario...
 %    refPad = padarray(ref, [round((Nx-Nxc)/2) round((Ny-Nyc)/2) round((Nz-Nzc)/2) ], 0, 'both');
 %    NzMask = Nz;
 % else % for a slice-encoded calibration, the ref data already achieved
 % full SE. 
 %    refPad = padarray(ref, [round((Nx-Nxc)/2) round((Ny-Nyc)/2) 0 ], 0, 'both');
 %    NzMask = Nzc;
 % end
 % ------------------------------------------------------------------------------------------------


 % ------------------------------------------------------------------------------------------------
 % We had this in this current version but now decied to comment out
 % because we now consider the size of Smask. 
 % refPad = padarray(ref, [round((Nx-Nxc)/2) round((Ny-Nyc)/2) round((Nz-Nzc)/2) ], 0, 'both');
 % if (Nz > Nzc) % standard scenario... the mask should have the same number of slices as the image. 
 %    % refPad = padarray(ref, [round((Nx-Nxc)/2) round((Ny-Nyc)/2) round((Nz-Nzc)/2) ], 0, 'both');
 %    NzMask = Nz;
 % else % for a slice-encoded calibration. the mask should have the same number of slices as the calibration.
 %    % refPad = padarray(ref, [round((Nx-Nxc)/2) round((Ny-Nyc)/2) 0 ], 0, 'both');
 %    NzMask = Nzc;
 % end
 % ------------------------------------------------------------------------------------------------

 refPad = padarray(ref, [round((Nxm - Nxc)/2) round((Nym - Nyc)/2) round((Nzm - Nzc)/2)], 0, 'both');

 imRefPad = zeros(size(ref));
 imRaw    = zeros(size(raw));
 for chIter = 1 : Nc
    imRefPad(:, :, :, chIter) = ifftnc(ref(:, :, :, chIter));
    imRaw   (:, :, :, chIter) = ifftnc(raw   (:, :, :, chIter));
 end

SmaskInd = find(Smask); % indices of the mask of the desired roi. 
Imask = 1 - Smask; % "Inverse mask" ... entries outside of the mask. 
ImaskInd = find(Imask); % indices of the mask of regions outside of the desired roi.

%% Create A and B matrices.  

A = zeros(Nc, Nc);
covIterA = zeros(Nc, Nc, length(SmaskInd));
for indIter = 1 : length(SmaskInd)
    indLoc = SmaskInd(indIter);
    % [xLoc, yLoc, zLoc] = ind2sub([Nxc, Nyc, NzMask], indLoc);
    [xLoc, yLoc, zLoc] = ind2sub([Nxm, Nym, Nzm], indLoc);
    g = squeeze(imRefPad(xLoc, yLoc, zLoc, :)).'; 
    covIterA(:, :, indIter) = g'* g;
    A = A + g' * g;
end
% A = sum(covIterA, 3);

B = zeros(Nc, Nc);
covIterB = zeros(Nc, Nc, length(ImaskInd));
for indIter = 1 : length(ImaskInd)
    indLoc = ImaskInd(indIter);
    [xLoc, yLoc, zLoc] = ind2sub([Nxm, Nym, Nzm], indLoc);
    g = squeeze(imRefPad(xLoc, yLoc, zLoc, :)).'; 
    covIterB(:, :, indIter) = g'* g;
    B = B + g'*g;
end
% B = sum(covIterB, 3);

%% 
[V, D] = eig(pinv(B) * A);

[ds, order] = sort(diag(D), 'descend');
Vs = V(:, order);

% figure, plot(abs(ds))

SIR_array = zeros(1, Nc);
for chIter = 1 : Nc
    w = Vs(:, chIter);
    SIR_array(1, chIter) = w' *(A) * w / (w' * B * w);
end

% eigNum = 4;
W = Vs(:, 1 : eigNum);

%%
Nvc = size(W, 2);
% -------------------------------- Original (Cartesian) -----------------------------
% imRawVs = zeros(Nx, Ny, Nz, Nvc);
% for vchIter = 1 : Nvc
%     w_vect = squeeze(W(:, vchIter));
%     for locIter = 1 : Nx * Ny * Nz
%         [xLoc, yLoc, zLoc] = ind2sub([Nx, Ny, Nz], locIter);
%         locVect = squeeze(imRaw(xLoc, yLoc, zLoc, :)); 
%         imRawVs(xLoc, yLoc, zLoc, vchIter) = locVect.' * w_vect;
%     end
% end
% -----------------------------------------------------------------------------------

% ------------------------------- Generalized ---------------------------------------
rawVs = zeros(Nro, Nspokes, Nz, Nvc); % number of ro poins, spokes, partitions, virtual channels.
for vchIter = 1 : Nvc
    w_vect = squeeze(W(:, vchIter));
    for locIter = 1 : Nro * Nspokes * Nz
        [roLoc, spokeLoc, zLoc] = ind2sub([Nro, Nspokes, Nz], locIter);
        locVect = squeeze(raw(roLoc, spokeLoc, zLoc, :));
        rawVs(roLoc, spokeLoc, zLoc, vchIter) = locVect.' * w_vect;
    end
end
% -----------------------------------------------------------------------------------