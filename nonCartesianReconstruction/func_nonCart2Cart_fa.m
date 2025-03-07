function [raw_grid, raw_grid_osf] = func_nonCart2Cart_fa(mu, coord_Matrix, H_and_U_andGrid_Struct, b_choice, D_0, osf, b_squareInPlane)
% Input:  
%         1. mu           - non Cartesian (acquired) data.  size Nro (# ro points) x Nspokes (# spokes, aka the number of total readouts).
%         2. coord_Matrix - Tells you the coordinates (kx,ky,kz) of each acquired
%         readoutpoint.  size Nro x Nspokes x 3 (1 - kx, 2 - ky, 3 - kz).
%         3. H_and_U_andGrid_Struct containst the H_matrix and U_matrix.  H_matrix - (OPTIONAL, sparse) the interpolation matrix.  If left
%         blank ([]), this functin will construct it. 
%         4. choice - option for 1 (least squares) or 2 (gridding).  Least
%         squares will solve H_matrix_sparse * U_matrix_recip *IF[raw_grid(:)] = mu(:). 
%         IF[] is the inverse Fourier transform operator, and
%         U_matrix_recip is a diagonal reciprocal of the deapodizing
%         matrix,  U_matrix, that is calculated in this function. 
%         Direct gridding will do the following: raw_grid(:) =
%         H_matrix_sparse' * D_0 * mu_combined, followed by 
%         raw_grid(:) = F[IF[raw_grid(:)] * U_matrix].  This takes us to
%         our last input:
%         5. D_0:  (OPTIONAL) The diagonal  density compensation term.  if not
%         specified, one will be calculated as diag(d0) = (H_matrix_sparse * H_matrix_sparse' * ones(length(mu_combined),1)).^(-1);
%         6. osf:  oversampling factor.  if left blank, osf = 2 by default.
%         7. b_squareInPlane:  just a boolean to determine if the matrix
%         should be squared or not. 
    if isempty(osf)%if ~exist(osf)
        osf = 2;
    end

    if isempty(H_and_U_andGrid_Struct)
        % H_matrix = [];
        % U_matrix = [];
        % [inputGridSize] = func_determineGridSize(coord_Matrix, osf, b_squareInPlane);
        % [H_matrix, U_matrix, inputGridSize] = func_createGridInterp_old_fa(mu, coord_Matrix, [], osf, b_squareInPlane, []);
        [H_matrix, U_matrix, inputGridSize] = func_createGridInterp_fa(coord_Matrix, [], osf, b_squareInPlane, []);

    else 
        H_matrix = H_and_U_andGrid_Struct.H_matrix;
        U_matrix = H_and_U_andGrid_Struct.U_matrix; 
        inputGridSize = H_and_U_andGrid_Struct.gridSize;
    end

    % [inputGridSize] = func_determineGridSize(coord_Matrix, osf, b_squareInPlane);
    
    % gridSize = inputGridSize; %in case the grid size changes in the function call in the if below.
    Nx = inputGridSize(1); Ny = inputGridSize(2); Nz = inputGridSize(3);

    % H_matrix = H_and_UStruct.H_matrix;
    % U_matrix = H_and_UStruct.U_matrix;
    % if isempty(H_matrix) || size(H_matrix,2) ~= Nx * Ny * Nz
    %     % [H_matrix, U_matrix, ~] = func_createGridInterp_fa(mu, coord_Matrix, [], osf, b_squareInPlane, gridSize);
    %     [H_matrix, U_matrix, gridSize] = func_createGridInterp_fa(mu, coord_Matrix, [], osf, b_squareInPlane, inputGridSize);
    % end



    
    % if isempty(H_matrix) || isempty(U_matrix) || isempty(inputGridSize) %|| size(H_matrix,2) ~= Nx * Ny * Nz
    %     % [H_matrix, U_matrix, ~] = func_createGridInterp_fa(mu, coord_Matrix, [], osf, b_squareInPlane, gridSize);
    %     [H_matrix, U_matrix, ~] = func_createGridInterp_fa(mu, coord_Matrix, [], osf, b_squareInPlane, inputGridSize);
    % end   
   % Nx = gridSize(1); Ny = gridSize(2); Nz = gridSize(3);

    if isempty(D_0)
        d_0 = (H_matrix * H_matrix' * ones(length(mu(:)),1)).^(-1);
        D_0 = sparse([1:length(d_0)], [1:length(d_0)], d_0, length(d_0), length(d_0));
    end
    

    % finalIm = zeros(round(Nx / osf), round(Ny / osf), round(Nz / osf));
    if b_choice == 1 % NUFFT
        nufft_result = lsqr(H_matrix, mu(:), 1e-6, 20);%pcg(T_combinedMatrix, mu_combined, 1e-6, 100);
        nufft_result = reshape(nufft_result, [Nx, Ny, Nz]);
        IF_nufft_result = ifftnc(nufft_result);
        UIF_nufft_result =  IF_nufft_result  ./ U_matrix; %ones(size(test_U));% test_U;

        %-------------- crop ----------------------------------------------
        startIndex_x = round((Nx - round(Nx/osf))/2);
        cropX = startIndex_x + 1 : startIndex_x + round(Nx/osf);
        startIndex_y = round((Ny - round(Ny/osf))/2);
        cropY = startIndex_y + 1 : startIndex_y + round(Ny/osf);
        startIndex_z = round((Nz - round(Nz/osf))/2);
        cropZ = startIndex_z + 1 : startIndex_z + round(Nz/osf);
        finalIm = UIF_nufft_result(cropX, cropY, cropZ);
        %------------------------------------------------------------------


        raw_grid = fftnc(finalIm);
        raw_grid_osf = fftnc(UIF_nufft_result);
    else %gridding
        grid_result = H_matrix' * D_0 * mu(:);
        grid_result = reshape(grid_result, [Nx, Ny, Nz]);
        IF_grid_result = ifftnc(grid_result);
        UIF_grid_result =  IF_grid_result .* U_matrix;

        %-------------- crop ----------------------------------------------
        startIndex_x = round((Nx - round(Nx/osf))/2);
        cropX = startIndex_x + 1 : startIndex_x + round(Nx/osf);
        startIndex_y = round((Ny - round(Ny/osf))/2);
        cropY = startIndex_y + 1 : startIndex_y + round(Ny/osf);
        startIndex_z = round((Nz - round(Nz/osf))/2);
        cropZ = startIndex_z + 1 : startIndex_z + round(Nz/osf);
        finalIm = UIF_grid_result(cropX, cropY, cropZ);
        %------------------------------------------------------------------

        raw_grid = fftnc(finalIm);
        raw_grid_osf = fftnc(UIF_grid_result);
    end
end