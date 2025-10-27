function [img_orig_pil, img_orig_np,img_LR_pil,img_LR_np,img_HR_pil,img_HR_np]  = load_LR_HR_imgs_sr(fname, imsize, factor, enforse_div32)
% Loads an image, resizes it, center crops and downscales.
%
% Args:
%     fname: path to the image (string)
%     imsize: new size for the image, -1 for no resizing (scalar or [height, width])
%     factor: downscaling factor (integer)
%     enforse_div32: if 'CROP', center crops an image to be divisible by 32 (string)

    % Load image
    img_orig = im2double(imread(fname));
    
    % Resize if needed
    if imsize ~= -1
        if isscalar(imsize)
            img_orig = imresize(img_orig, [imsize, imsize]);
        else
            img_orig = imresize(img_orig, imsize);
        end
    end
    
    img_orig_np = img_orig; % Store original numpy-like version
    img_orig_pil = img_orig; % MATLAB uses matrices, no need to separate formats
    
    % Enforce divisible by 32 crop if specified
    if exist('enforse_div32', 'var') && strcmp(enforse_div32, 'CROP')
        [h, w, ~] = size(img_orig_pil);
        new_h = h - mod(h, 32);
        new_w = w - mod(w, 32);
        
        top = floor((h - new_h)/2) + 1;
        left = floor((w - new_w)/2) + 1;
        
        img_HR_pil = imcrop(img_orig_pil, [left, top, new_w-1, new_h-1]);
    else
        img_HR_pil = img_orig_pil;
    end
    
    img_HR_np = img_HR_pil;
    
    % Downscale to get LR image
    [h_hr, w_hr, ~] = size(img_HR_pil);
    LR_size = [floor(h_hr / factor), floor(w_hr / factor)];
    
    img_LR_pil = imresize(img_HR_pil, LR_size, 'lanczos3'); % MATLAB’s LANCZOS3 = PIL LANCZOS
    img_LR_np = img_LR_pil;
    
    fprintf('HR and LR resolutions: [%d, %d], [%d, %d]\n', ...
            size(img_HR_pil,2), size(img_HR_pil,1), ...
            size(img_LR_pil,2), size(img_LR_pil,1));

end
