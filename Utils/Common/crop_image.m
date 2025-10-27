function img_cropped = crop_image(img,d)
    if nargin < 2
        d = 32;
    end
    [height, width, ~] = size(img);
    new_height = height - mod(height,d); %mod function returns the remainder after division of heigth/d
    new_width = width - mod(width,d);
    %Calculate crop box (centered)
    y1 = floor((height - new_height) / 2) + 1; %floor rounds each element of what is inside to the closest integer (smaller or equal to the element)
    y2 = y1 + new_height - 1;
    x1 = floor((width - new_width) / 2) + 1;
    x2 = x1 + new_width - 1;
    img_cropped=img(y1:y2,x1:x2,:);
end