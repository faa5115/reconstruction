function [mu] = func_Cart2nonCart_fa(raw_grid_osf, coord_Matrix, H_and_U_andGrid_Struct,b_choice, D_0, osf, b_squareInPlane)
   % Input:  
%         1. raw_grid_osf           -  Oversampled Cartesian data.  size Nx x Ny x Nz.
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
    Nx = inputGridSize(1); Ny = inputGridSize(2); Nz = inputGridSize(3);
    % The H matrix that was used to go from non Cart to cart is oversampled
    % by factor osf. 

    if isempty(D_0)
        % d_0 = (H_matrix * H_matrix' * ones(length(mu(:)),1)).^(-1);
         d_0 = (H_matrix * H_matrix' * ones(size(H_matrix,1),1)).^(-1);
        D_0 = sparse([1:length(d_0)], [1:length(d_0)], d_0, length(d_0), length(d_0));
    end

    if b_choice == 1 % NUFFT
        UIF_nufft_result = ifftnc(raw_grid_osf);
        IF_nufft_result = UIF_nufft_result .* U_matrix;
        nufft_result = fftnc(IF_nufft_result);
        mu = H_matrix * nufft_result(:);
    else % gridding
        UIF_grid_result = ifftnc(raw_grid_osf);
        IF_grid_result = UIF_grid_result ./ U_matrix;
        grid_result = fftnc(IF_grid_result);
        mu = D_0 * H_matrix * grid_result(:);
    end
end