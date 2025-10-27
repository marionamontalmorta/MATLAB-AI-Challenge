function [img_bicubic_np, img_bic_sharp_np, img_nearest_np] = get_baselines(img_LR_pil, img_HR_pil)
% Gets 'bicubic', sharpened bicubic and 'nearest' baselines.
%
% Inputs:
%   img_LR_pil: Low-resolution image (matrix)
%   img_HR_pil: High-resolution image (matrix) – only used for size reference
%
% Outputs:
%   img_bicubic_np: Bicubic upsampled image
%   img_bic_sharp_np: Sharpened bicubic upsampled image
%   img_nearest_np: Nearest-neighbor upsampled image

    target_size = [size(img_HR_pil, 1), size(img_HR_pil, 2)];

    % Bicubic interpolation
    img_bicubic_np = imresize(img_LR_pil, target_size, 'bicubic');

    % Nearest-neighbor interpolation
    img_nearest_np = imresize(img_LR_pil, target_size, 'nearest');

    % Apply unsharp mask to bicubic result
    h = fspecial('unsharp');
    img_bic_sharp_np = imfilter(img_bicubic_np, h, 'replicate');

end
