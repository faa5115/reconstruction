%% First get the data.

% % % % delta_phi =  2 * pi / Nspokes; % linear increments (radians) 2.39996322972865332; % golden angle (radians) %
% % % % rotZ = zrot(delta_phi);

% First load the phantom.
N  = 150; % number of Cartesian points along one direction.
P = phantom('Modified Shepp-Logan',N);

% Then we must get the spokes.  The spokes will readout from 0 to kmax. 
% I will carry out the radon transform to get several projections, each at
% a different angle.  These projections are in the image domain.  The FT of
% each projection will then be a radial spoke in k-space.  Because Vahid
% wants the spoke to be from 0 to kmax, only the second half of each spoke
% will be kept.  The following will be the steps:
%           1. radon transform of P at multiple angles.
%           2. FFT of each projection.
%           3. store only the second half.
% These will be the k-space spokes that we will NUFFT.  
Nspokes =  610;
delta_phi =  2 * pi / Nspokes;
spokeAnglesArr = 0 : delta_phi : (Nspokes - 1) * delta_phi;

%First I must know the size of the output of each radon transform.  The
%number of points along a projection will be the number of readout points
%that we "sample."
[Proj_firstiter, xp] = radon(P, spokeAnglesArr(1).' .* 180 / pi);
NroFullSpoke = size(Proj_firstiter,1);
cXval = find(xp==0)
cYval = find(xp==0)
cZval = 1;

% Now that we know the number of readout points, we now can preallocate for
% the projections of the phantom. 

Projs = zeros(NroFullSpoke, Nspokes); % Stores the number of projections.
for spokeiter = 1 : Nspokes
    [Proj_iter, ~] = radon(P, spokeAnglesArr(spokeiter).' .* 180 / pi);
    Projs(:, spokeiter) = Proj_iter;
end

Nro = round(NroFullSpoke/2); % The number of readout points.
                           % it is half of the number of readouts 
                           % of "the full spoke."
% % ---------This subsection just shows the inv. radon transfrom -------
% ... not related to your work.-----------------
 [invRadon] = iradon(Projs, spokeAnglesArr .* 180 / pi);
% debug
% figure, imshow([abs(invRadon)], []), imcontrast()
%---------------------------------------------------------------

% get full k-space spoke (-kmax to kmax) and half spoke (0 to kmax)
% Prepare the readout raw data ---------------------------
muFullSpoke = zeros(size(Projs));
mu = zeros(Nro, Nspokes);
for spokeiter = 1 : Nspokes
    muFullSpoke(:, spokeiter) = fftnc(squeeze(Projs(:, spokeiter)));
    mu(:, spokeiter) = muFullSpoke(round(NroFullSpoke/2) : end, spokeiter);
end

% NOTE for Vahid!:  mu ( size Nro (readoutpoints) x Nspokes (spokes)) is
% your raw (non cartesian) k-space data. "mu_combined" (Nro * Nspoks x 1) just stacks them to
% one long vector because my function takes it as a long vector. 
mu_combined = mu(:);
%%  Create coord_Matrix, which stores all non-Cartesian acquired data.
% coord_Matrix holds the nonCartesian coordinates.  
% it is structured as 
% Nro points for a readout x number of readout spokes x 3 (kx,ky,kz
% coordinates).

% First I define the "nominal" number of pixels.  Because these are the nominal 
% pixel numbers, that means they are"pre" oversampling.  
Nx_pre = N; 
Ny_pre = N;
Nz_pre = 1;

% Define oversampling factor (osf).  We will do 2.
osf = 2; 
Nx = osf * Nx_pre; % Nro
Ny = osf * Ny_pre; % Nro
Nz = osf * Nz_pre;

% Initialize the coord_Matrix:  size Nro x Nspokes x 3 (kx,ky,kz).
% It stores the (kx,ky,kz) coordinate of each sampled readout of each
% spoke. k-space units are cycles / FOV. 
coord_Matrix = zeros(Nro, Nspokes, 3); % pre oversampling coordinates.

coord_Matrix(:, 1, 1) = 0 : (Nro-1);
coord_Matrix(:, 1, 2) = (0 : (Nro-1)) .* 0;
coord_Matrix(:, 1, 3) = (0 : (Nro-1)) .* 0;

for spokeiter = 2 : Nspokes
    rotZ = zrot(spokeAnglesArr(spokeiter));
    for roiter = 1 : Nro
        coord_Matrix(roiter, spokeiter, :) = rotZ * squeeze(coord_Matrix(roiter, 1, :));
    end
    % spokeAnglesArr(spokeiter, 1) = spokeAnglesArr(spokeiter - 1, 1) + delta_phi;
end
disp('finished creating coord_Matrix')

%%  Debug ... plot all coord_Matrix values.

figure,
plot(coord_Matrix(:, 1 , 1), coord_Matrix(:, 1 , 2))
hold on
for spokeiter = 2 : Nspokes
    plot(coord_Matrix(:, spokeiter , 1), coord_Matrix(:, spokeiter, 2))
end
hold off
%%  Now for the reconstruction.  My recon function:

H_and_U_andGrid_Struct = []; % A structure containing the 
                             % interpolation matrix. My function creates it
                             % internally if you leave it emppty.
b_choice = 1; % Least-squares gridding.  if 2, it just use the adjoint 
% which is what people often refer to as "gridding".
D_0 = []; % Density compensation term.  Because we are doing NUFFT, it is not necessary.
b_squareInPlane = 1;  % saying the k-space matrix should be square in plane. 
[raw_grid_ls, raw_grid_ls_osf] = ... %"_ls refers to "least squares"
    func_nonCart2Cart_fa(mu, coord_Matrix, H_and_U_andGrid_Struct, b_choice, D_0, osf, b_squareInPlane);
% raw_grid and raw_grid_osf both give you the gridded k-space. 
% The only difference is that raw_grid_osf is the oversampled result.

[raw_grid_ad, raw_grid_ad_osf] = ... %"_ad refers to "adjoint"
    func_nonCart2Cart_fa(mu, coord_Matrix, H_and_U_andGrid_Struct, 2, D_0, osf, b_squareInPlane);

%%
imGrid_ls = ifftnc(raw_grid_ls);
imGrid_lsosf = ifftnc(raw_grid_ls_osf);

imGrid_ad = ifftnc(raw_grid_ad);
imGrid_adosf = ifftnc(raw_grid_ad_osf);

figure
subplot(3, 2, 1)
imshow(abs(P), [])
title('original phantom')
subplot(3, 2, 2)
imshow(abs(invRadon), [])
title('inverse radon')
subplot(3, 2, 3)
imshow(abs(imGrid_ls), [])
title('raw grid by least-squares')
subplot(3, 2, 4)
imshow(abs(imGrid_lsosf), [])
title('raw grid osf by least-squares')
subplot(3, 2, 5)
imshow(abs(imGrid_ad), [])
title('raw grid using the adjoint')
subplot(3, 2, 6)
imshow(abs(imGrid_adosf), [])
title('raw grid osf using the adjoint')

%% Okay now let's go from gridded Cartesian back to non Cartesian. The Adjoint
[mu_adjoint] = ...
    func_Cart2nonCart_fa(raw_grid_osf, coord_Matrix, ...
    H_and_U_andGrid_Struct,b_choice, D_0, osf, b_squareInPlane);

mu_adjoint = reshape(mu_adjoint, [Nro, Nspokes]);
