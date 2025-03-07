function [gridSize] = func_determineGridSize(coord_Matrix, osf, b_squareInPlane)
    
    coord_Matrix_osf = coord_Matrix .* osf;

    coord_Matrix_osf_x = coord_Matrix_osf(:, :, 1);
    coord_Matrix_osf_y = coord_Matrix_osf(:, :, 2);
    coord_Matrix_osf_z = coord_Matrix_osf(:, :, 3);

    Nx = max(round(2 * abs(max(coord_Matrix_osf_x(:))) ), round(2 * abs(min(coord_Matrix_osf_x(:))) ));
    Ny = max(round(2 * abs(max(coord_Matrix_osf_y(:))) ), round(2 * abs(min(coord_Matrix_osf_y(:))) ));
    Nz = max(round(2 * abs(max(coord_Matrix_osf_z(:))) ), round(2 * abs(min(coord_Matrix_osf_z(:))) ));
    
    if b_squareInPlane  == 1
        Nlarger = max(Nx, Ny);
        Nx = Nlarger;
        Ny = Nlarger;
    end
    
    if Nz == 0
        Nz = 1;
    end

    gridSize = [Nx, Ny, Nz];
end