function final_im = SENSE_fa1D(input_im, sensmap, R)
    
    [Nx,Ny,Nz,Nc] = size(input_im);
    final_im = zeros(Nx,Ny,Nz);
    
    %----------------------------------------------------------------------------------
%     % determine the channel with the max intensity.  this will be the channel's
%     % relative phase that we will use. 
%     sumEachChannel = zeros(1, Nc);
%     for channelIter = 1 : Nc
%         imCh = input_im(:, :, :, channelIter);
%         sumEachChannel(1, channelIter) = sum(imCh(:));
%     end
%     [~,maxIntensityChannel] = max(sumEachChannel(:));
%     locationIndices  = 1 : Nx * Ny * Nz;
%     % locationIndicesM = reshape(locationIndices, Nxc, Nyc, Nzc);
% 
%     for locIter = 1 : Nx * Ny * Nz
%         [row, col, s] = ind2sub([Nx, Ny, Nz], locIter);
%         vect = squeeze(input_im(row, col, s, :));
%         input_im(row, col, s, :) = vect .* exp(-1i*angle(vect(maxIntensityChannel)));
%     end
    %----------------------------------------------------------------------------------

    sensChannels = size(sensmap, 4);

    % loop over the top-half of the image
    for y = 1:Ny/R
        % loop over the entire left-right extent
        for z = 1:Nz
            for x = 1:Nx
                % pick out the sub-problem sensitivities
                S_R2 = transpose(reshape(sensmap(x,[y y+Ny/R],z,:),R,[])); 
                % solve the sub-problem in the least-squares sense
                imVect = repmat([reshape(input_im(x,y,z,:),[],1)], sensChannels/Nc, 1);
                final_im(x, [y y+Ny/R], z) = pinv(S_R2)*imVect;%reshape(input_im(x,y,z,:),[],1);
            end
        end
    end
end