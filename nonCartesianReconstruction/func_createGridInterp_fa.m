function [H_matrix_sparse, U_matrix, gridSize] = func_createGridInterp_fa(coord_Matrix,interp_kernel, osf, b_squareInPlane, inputGridSize)
% This function creates the H_matrix_sparse needed to interpolate acquired
% non-uniformly sampled data onto a uniform grid:  H_matrix_sparse * x = mu
% where mu is the acquired (non uniform) raw data and x is the "gridded"
% data. This also outputs the deadopizing matrix and the final gridsize
% (Nx, Ny, Nz).  U_matrix is Nx x Ny x Nz
%
%   We got rid of the mu input ... we really did not need it at all.  i was lazy and put it there to use its length.  
%
% The following are inputs:
% 1. coord_Matrix:  of size Nro x Nspokes x 3.  Nro is the number of 
% points for each spoke/readout.  Nspokes is the number of spokes/readouts.
% The last dimension refers to the kx (1), ky (2), and kz(3) coordinates. 
% 2. interp_kernel: The interpolation kernel of size of the kernel length along each k-space dimension, k_wx x k_wy x k_wz.  
% 3. osf:  the oversampling factor. 

if isempty(osf)%if ~exist(osf)
    osf = 2;
end

Nro = size(coord_Matrix, 1);
Nspokes = size(coord_Matrix, 2);





coord_Matrix_osf =  coord_Matrix .* osf; 

% disp(size(coord_Matrix))
% disp(size(coord_Matrix_osf))

%{
figure,
plot(coord_Matrix(:, 1 , 1), coord_Matrix(:, 1 , 2))
hold on
for spokeiter = 2 : Nspokes
    plot(coord_Matrix(:, spokeiter , 1), coord_Matrix(:, spokeiter, 2))
end
hold off
%}
% 
%{
figure,
plot(coord_Matrix_osf(:, 1 , 1), coord_Matrix_osf(:, 1 , 2))
hold on
for spokeiter = 2 : Nspokes
    plot(coord_Matrix_osf(:, spokeiter , 1), coord_Matrix_osf(:, spokeiter, 2))
end
hold off
%}


% coord_Matrix_x = coord_Matrix(:, :, 1);
% coord_Matrix_y = coord_Matrix(:, :, 2);
% coord_Matrix_z = coord_Matrix(:, :, 3);


coord_Matrix_osf_x = coord_Matrix_osf(:, :, 1);
coord_Matrix_osf_y = coord_Matrix_osf(:, :, 2);
coord_Matrix_osf_z = coord_Matrix_osf(:, :, 3);

% Nx = round(2 * max(coord_Matrix_osf_x(:)));
% Ny = round(2 * max(coord_Matrix_osf_y(:)));
% Nz = round(2 * max(coord_Matrix_osf_z(:)));

%-----------------------------WHAT I MOST RECENTLY HAD------------------------------
% Nx = max(round(2 * abs(max(coord_Matrix_osf_x(:))) ), round(2 * abs(min(coord_Matrix_osf_x(:))) ));
% Ny = max(round(2 * abs(max(coord_Matrix_osf_y(:))) ), round(2 * abs(min(coord_Matrix_osf_y(:))) ));
% Nz = max(round(2 * abs(max(coord_Matrix_osf_z(:))) ), round(2 * abs(min(coord_Matrix_osf_z(:))) ));
% 
% if b_squareInPlane  == 1
%     Nlarger = max(Nx, Ny);
%     Nx = Nlarger;
%     Ny = Nlarger;
% end
% 
% if Nz == 0
%     Nz = 1;
% end
%------------------------------------------------------------------------------------
% use line below to replace what you had directly above ... same thing.
% just to be clean.  input coord_Matrix instead of coord_Matrix_osf because
% func_determineGridSize oversamples. 
[gridSize] = func_determineGridSize(coord_Matrix, osf, b_squareInPlane);
Nx = gridSize(1); Ny = gridSize(2); Nz = gridSize(3);

if ~isempty(inputGridSize)%exist(inputGridSize)
    if Nx < inputGridSize(1)
        Nx = inputGridSize(1);
    end
    if Ny < inputGridSize(2)
        Ny = inputGridSize(2);
    end
    if Nz < inputGridSize(3)
        Nz = inputGridSize(3);
    end
    gridSize = [Nx, Ny, Nz];
end

if mod(Nx,2) == 0
    cXval = round(Nx/2) + 1;
    kxLocationValues = -Nx/2 : Nx/2 - 1;
else
    cXval = round(Nx/2);
    kxLocationValues = -(Nx - 1)/2 : (Nx - 1)/2;
end
if mod(Ny,2) == 0
    cYval = round(Ny/2) + 1;
    kyLocationValues = -Ny/2 : Ny/2 - 1;
else
    cYval = round(Ny/2);
    kyLocationValues = -(Ny - 1)/2 : (Ny - 1)/2;
end
if mod(Nz, 2) == 0
    cZval = round(Nz/2) + 1;
    kzLocationValues = -Nz/2 : Nz/2 - 1;
else
    cZval = round(Nz/2);
    kzLocationValues = -(Nz - 1)/2 : (Nz - 1)/2;
end




if ~isempty(interp_kernel) %exist(interp_kernel)
    [kb_wx, kb_wy, kb_wz] = size(interp_kernel);
    FrecipU_center = interp_kernel;
else % if not specified, just use the kaiser-bessel.
    % kb_w = 5;  kb_beta = 8;
    kb_w = 7;  kb_beta = 20;
    kb_wx = kb_w; kb_wy = kb_w; kb_wz = kb_w;
    if Nz < kb_w %if Nz == 1
        kb_wz = Nz;
    end
   
    syms kb_sym(k)
    kb_sym(k) =  besseli(0, kb_beta * sqrt(1 - ((2/kb_w) * k).^2 )) ./ besseli(0, kb_beta); % (1 / kb_w) *  besseli(0, kb_beta * sqrt(1 - ((2/kb_w) * k).^2 )) ./ besseli(0, kb_beta);

    k_array   = linspace(-kb_w/2, kb_w/2, 200); 
%   figure, plot( abs(ifftnc(double(kb_sym(k_array)))), 'linewidth', 5.0)
%   figure, plot(k_array, double(kb_sym(k_array)))
    if mod(kb_w,2) == 1 % odd
        max_width = (kb_w-1)/2;
    else
        max_width = kb_w/2;
    end

    kernel_indices = -max_width : max_width; % the coordinates of the kernel we evaluate
    kb_output = kb_sym(kernel_indices);
    FrecipU_center = ones(kb_wx, kb_wy, kb_wz);


    for ziter = 1 : kb_wz
        kb_wzcoord = ziter - kb_wz/2;
        for yiter = 1 : kb_wy
            FrecipU_row = squeeze(FrecipU_center(:, yiter, ziter));
            FrecipU_center(:, yiter, ziter) = FrecipU_row.' .* kb_output;
            FrecipU_center(:, yiter, ziter) = FrecipU_center(:, yiter, ziter) .* double(kb_sym(kb_wzcoord));
        end
    
        for xiter = 1 : kb_wx
            FrecipU_col = squeeze(FrecipU_center(xiter, :, ziter));
            FrecipU_center(xiter, :, ziter) = FrecipU_col .* kb_output;
            FrecipU_center(xiter, :, ziter) = FrecipU_center(xiter, :, ziter) .* double(kb_sym(kb_wzcoord)); 
        end
    end

end

clear interp_kernel; 
FrecipU_matrix = zeros(Nx, Ny, Nz);
FrecipU_matrix(Nx/2+1 - max_width : Nx/2 + 1 + max_width, Ny/2+1 - max_width : Ny/2 + 1 + max_width, :) = FrecipU_center;

% debug
% figure, imshow(abs(FrecipU_matrix), [])

recipU_matrix = ifftnc(FrecipU_matrix);
% debug
% figure, imshow(abs(recipU_matrix), [])

U_matrix = recipU_matrix.^(-1);
% % clear FrecipU_matrix recipU_matrix
% lengthU = length(U_matrix(:));
% % U_diag_matrix = sparse(diag(U_matrix(:)));
% % % the above line is too memory intense for larger matrices.  ...
% % U_diag_matrix2 = sparse([1:lengthU], [1:lengthU], U_matrix(:), lengthU, lengthU);
% % % okay above works ... we will work with this.
% U_diag_matrix = sparse([1:lengthU], [1:lengthU], U_matrix(:), lengthU, lengthU);


NinterpNeighbors = nnz(FrecipU_matrix);%kb_wx * kb_wy * kb_wz;


% Generate a list of k space neighbors for each non-cartesian location


% coord_Matrix_x = coord_Matrix(:, :, 1);
% coord_Matrix_y = coord_Matrix(:, :, 2);
% coord_Matrix_z = coord_Matrix(:, :, 3);

% Neighbors = sparse(length(mu(:)), NinterpNeighbors);

% get a vector of ro iter indices and spokeiter indices.
% [roiternc_vec, spokeiter_vec] = ind2sub(size(mu), 1 : length(mu(:)));

kxnc_vec = coord_Matrix_osf_x(:); % coord_Matrix_x(1 : length(mu(:)));
kync_vec = coord_Matrix_osf_y(:); % coord_Matrix_y(1 : length(mu(:)));
kznc_vec = coord_Matrix_osf_z(:); % coord_Matrix_z(1 : length(mu(:)));


% kxc_range = kxnc_vec - round(kb_wx/2) : kxnc_vec + round(kb_wx/2);
% kyc_range = kync_vec - round(kb_wy/2) : kync_vec + round(kb_wy/2);
% kzc_range = kznc_vec - round(kb_wz/2) : kznc_vec + round(kb_wz/2);

% lower values 
kxc_range_lower =  kxnc_vec - floor(kb_wx/2); kxc_range_lower = round(kxc_range_lower);
kyc_range_lower =  kync_vec - floor(kb_wy/2); kyc_range_lower = round(kyc_range_lower); 
kzc_range_lower =  kznc_vec - floor(kb_wz/2); kzc_range_lower = round(kzc_range_lower); 

% upper values
kxc_range_upper =  kxnc_vec + floor(kb_wx/2); kxc_range_upper = round(kxc_range_upper); 
kyc_range_upper =  kync_vec + floor(kb_wy/2); kyc_range_upper = round(kyc_range_upper);
kzc_range_upper =  kznc_vec + floor(kb_wz/2); kzc_range_upper = round(kzc_range_upper);


% now we must evaluate the limits that is out of range. 
% this is what we did above: 
        % % x_oor = find(kxc_range <= max(abs(kxLocationValues(:))));
        % % kxc_range(x_oor) = []; 
        % % y_oor = find(kyc_range <= max(abs(kyLocationValues(:))));
        % % kyc_range(y_oor) = [];
        % % z_oor = find(kzc_range <= max(abs(kzLocationValues(:))));


% Find indices that are out of range. Then make them the minimum or maximum possible
% value on the grid. 
% -----------------------What we had before 01/23/2024----------------------
% oorX = kxc_range_lower <= min(coord_Matrix_osf_x(:));%min(kxLocationValues(:));
% kxc_range_lower(oorX) = ceil(min(coord_Matrix_osf_x(:)))+1;%round(min(kxLocationValues(:)))+1;
% oorX = find(kxc_range_upper > max(coord_Matrix_osf_x(:)));%max( (kxLocationValues(:))));
% kxc_range_upper(oorX) = round(max( (coord_Matrix_osf_x(:))))-1;%round(max( (kxLocationValues(:))))-1;
% 
% oorY = kyc_range_lower <= min(coord_Matrix_osf_y(:));% min(kyLocationValues(:));
% kyc_range_lower(oorY) = ceil(min(coord_Matrix_osf_y(:)))+1;%round(min(kyLocationValues(:)))+1;
% oorY = find(kyc_range_upper > max( (coord_Matrix_osf_y(:))));%max( (kyLocationValues(:))));
% kyc_range_upper(oorY) = round(max( (coord_Matrix_osf_y(:))))-1;% round(max( (kyLocationValues(:))))-1;
% 
% oorZ = kzc_range_lower <= min(coord_Matrix_osf_z(:));%min(kzLocationValues(:));
% kzc_range_lower(oorZ) = ceil(min(coord_Matrix_osf_z(:)))+1;% round(min(kzLocationValues(:)))+1;
% oorZ = find(kzc_range_upper > max( (coord_Matrix_osf_z(:))));%max( (kzLocationValues(:))));
% kzc_range_upper(oorZ) = round(max( (coord_Matrix_osf_z(:))))-1;%round(max( (kzLocationValues(:))))-1;
% -------------------------------------------------------------------------

% -------------------------On and After 01/23/2024----------------------------
oorX = kxc_range_lower < min(kxLocationValues(:)); % <= min(coord_Matrix_osf_x(:));%min(kxLocationValues(:));
kxc_range_lower(oorX) = min(kxLocationValues(:));%round(min(kxLocationValues(:)))+1;
oorX = find(kxc_range_upper > max(kxLocationValues(:)));%max(coord_Matrix_osf_x(:)));%max( (kxLocationValues(:))));
kxc_range_upper(oorX) = max(kxLocationValues(:));%round(max( (coord_Matrix_osf_x(:))))-1;%round(max( (kxLocationValues(:))))-1;

oorY = kyc_range_lower < min(kyLocationValues(:));% <= min(coord_Matrix_osf_y(:));% min(kyLocationValues(:));
kyc_range_lower(oorY) = min(kyLocationValues(:));%= ceil(min(coord_Matrix_osf_y(:)))+1;%round(min(kyLocationValues(:)))+1;
oorY = find(kyc_range_upper > max(kyLocationValues(:)));%> max( (coord_Matrix_osf_y(:))));%max( (kyLocationValues(:))));
kyc_range_upper(oorY) = max(kyLocationValues(:));%= round(max( (coord_Matrix_osf_y(:))))-1;% round(max( (kyLocationValues(:))))-1;

oorZ = kzc_range_lower < min(kzLocationValues(:));% <= min(coord_Matrix_osf_z(:));%min(kzLocationValues(:));
kzc_range_lower(oorZ) = min(kzLocationValues(:));%= ceil(min(coord_Matrix_osf_z(:)))+1;% round(min(kzLocationValues(:)))+1;
oorZ = find(kzc_range_upper > max(kzLocationValues(:)));%> max( (coord_Matrix_osf_z(:))));%max( (kzLocationValues(:))));
kzc_range_upper(oorZ) = max(kzLocationValues(:));%= round(max( (coord_Matrix_osf_z(:))))-1;%round(max( (kzLocationValues(:))))-1;


% ----------------------------------------------------------------------------
% cXval = round(Nx/2);
% cYval = round(Ny/2);
% cZval = round(Nz/2);

% % test code with ngrid
% testx = [1, 2, 3, 4]; testy = [5, 6]; testz = [-1, 0, 1];
% [testxx, testyy, testzz] = ndgrid(testx, testy, testz);
% testcoord = [testxx(:), testyy(:), testzz(:)];
% % now use meshgrid
% [meshxx, meshyy, meshzz] = meshgrid(testx, testy, testz);
% meshcoord = [meshxx(:), meshyy(:), meshzz(:)];
Neighbors_list = zeros(Nro * Nspokes, kb_w * kb_w);%zeros(length(mu(:)), kb_w * kb_w);
for muiter = 1 : Nro * Nspokes % length(mu(:))
    lowervalx = kxc_range_lower(muiter); uppervalx = kxc_range_upper(muiter); xNeighbIndexArr = lowervalx + cXval : uppervalx + cXval;
    lowervaly = kyc_range_lower(muiter); uppervaly = kyc_range_upper(muiter); yNeighbIndexArr = lowervaly + cYval : uppervaly + cYval;
    lowervalz = kzc_range_lower(muiter); uppervalz = kzc_range_upper(muiter); zNeighbIndexArr = lowervalz + cZval : uppervalz + cZval;
    
    % if length(zNeighbIndexArr) == 1
    %     [index_grid_x, index_grid_y]               = ngrid(xNeighbIndexArr, yNeighbIndexArr);
    % else
    %     [index_grid_x, index_grid_y, index_grid_z] = ngrid(xNeighbIndexArr, yNeighbIndexArr, zNeighbIndexArr);
    % end

    % meshgrid is better
    % [index_grid_x, index_grid_y, index_grid_z] = meshgrid(xNeighbIndexArr, yNeighbIndexArr, zNeighbIndexArr);
    if length(zNeighbIndexArr) <= 1
        [index_grid_x, index_grid_y]               = meshgrid(xNeighbIndexArr, yNeighbIndexArr);
        % disp(muiter)
        tempList = sub2ind([Nx, Ny], index_grid_x(:), index_grid_y(:));
    else
        [index_grid_x, index_grid_y, index_grid_z] = meshgrid(xNeighbIndexArr, yNeighbIndexArr, zNeighbIndexArr);
        tempList = sub2ind([Nx, Ny, Nz], index_grid_x(:), index_grid_y(:), index_grid_z(:));
    end
    % coord_neighb = [index_grid_x(:), index_grid_y(:), index_grid_z(:)];

 
    % tempList = sub2ind([Nx, Ny, Nz], index_grid_x(:), index_grid_y(:), index_grid_z(:));
    Neighbors_list(muiter, 1:length(tempList)) = tempList;
    % clear tempList ;
    
end

disp('finished creating Neighbors_list')



% Now for creating H_matrix_sparse

neighborIndex_vec = Neighbors_list(find(Neighbors_list(:))); % All indices corresponding to Cartesian grid points in Neighbors_list.
[mu_iter_vec, neighbiter_vec] = ind2sub(size(Neighbors_list), find(Neighbors_list(:))); % the row (muiter) and col (neighboriter) of nonzero Neighbors_list entries. 
[kxiter_vec, kyiter_vec, kziter_vec] = ind2sub([Nx, Ny, Nz], neighborIndex_vec); % get the Cartesian indices.
kxc_vec = kxiter_vec - cXval; kyc_vec = kyiter_vec - cYval; kzc_vec = kziter_vec - cZval;
[roiternc_vec, spokeiter_vec] = ind2sub([Nro, Nspokes], mu_iter_vec); % get the ro index and spoke 
% [roiternc_vec, spokeiter_vec] = ind2sub(size(mu), mu_iter_vec); % get the ro index and spoke 

kxnc_vec = coord_Matrix_osf_x(mu_iter_vec);
kync_vec = coord_Matrix_osf_y(mu_iter_vec);
kznc_vec = coord_Matrix_osf_z(mu_iter_vec);

dist_val_vec = sqrt((kxc_vec - kxnc_vec).^2 + (kyc_vec - kync_vec).^2 + (kzc_vec - kznc_vec).^2);
% kb(k) =  besseli(0, kb_beta * sqrt(1 - ((2/kb_w) * k).^2 )) ./ besseli(0, kb_beta); % (1 / kb_w) *  besseli(0, kb_beta * sqrt(1 - ((2/kb_w) * k).^2 )) ./ besseli(0, kb_beta);
kb_values_vec = double(besseli(0, kb_beta * sqrt(1 - ((2/kb_w) * dist_val_vec).^2 )) ./ besseli(0, kb_beta)); 
H_matrix_sparse = sparse(mu_iter_vec, neighborIndex_vec, kb_values_vec,Nro * Nspokes, Nx*Ny*Nz ) ;



disp('finished creating H_matrix sparse')


disp('end')