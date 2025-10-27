function [outn, outp] = pil_to_np(img)
    img=im2single(img);
     % If grayscale (2D), add a channel dimension
    if ndims(img) == 2
        img = reshape(img, size(img,1), size(img,2), 1);
    end
    % Permute dimensions from H x W x C to C x W x H
    outn=img;
    outp = permute(img, [3, 2, 1]);
end