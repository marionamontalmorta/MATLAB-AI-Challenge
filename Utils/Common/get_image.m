function [img,img_np] = get_image(fname, imsize)
    if nargin < 2 %nargin=nº of input arguments
        imsize=-1;
    end
    fname=string(fname);
    
    % Check if file exists in current directory first
    if exist(fname, 'file')
        path = fname;
    else
        % Try to find the file in common image directories
        possible_paths = {
            fullfile(pwd, fname),
            fullfile(pwd, 'Images', fname),
            fullfile(pwd, 'Data', fname),
            fullfile(pwd, '..', 'Images', fname),
            fullfile(pwd, '..', 'Data', fname)
        };
        
        path_found = false;
        for i = 1:length(possible_paths)
            if exist(possible_paths{i}, 'file')
                path = possible_paths{i};
                path_found = true;
                break;
            end
        end
        
        if ~path_found
            error('Image file not found: %s. Please ensure the image is in the current directory or a subdirectory.', fname);
        end
    end
    
    img=imread(path); %the path is the path to the image file
    % Convert grayscale to RGB (if needed)
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]); %repmat specifies the repetition scheme with row vector, repeats the image matrix 3 times in the depth direction. Converts it to RGB
    end
    % Convert scalar imsize to [height, width]
    if isscalar(imsize)
        imsize = [imsize, imsize];
    end
    % Resize if necessary
    if imsize(1) ~= -1 && any(size(img, 1:2) ~= imsize)
        if imsize(1) > size(img, 1)
            img = imresize(img, imsize, 'bicubic'); %returns img with the size stated, interpolates the values of the pixels with bicubic method
        else
            img = imresize(img, imsize, 'lanczos3'); % 'lanczos3' is a good antialiasing choice
        end
    end
    % Convert to double in [0, 1], like pil_to_np
    img_np = im2double(img); %converteix l'imatge de unit8 a tipus double
end
