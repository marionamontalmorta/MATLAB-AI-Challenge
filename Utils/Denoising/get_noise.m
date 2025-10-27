function net_input = get_noise(input_depth, mode, img_size, var)
if numel(img_size)~=2
    error('img_size must be a 2-element vector for [height,widht].');
end
sz = [img_size, input_depth, 1];

switch lower(mode)
    case 'uniform'
        net_input = rand(sz, 'single');
    case 'normal'
        net_input = randn(sz, 'single');
    otherwise
        error('Modo de ruido no soportado.');
end

net_input = net_input * var;
net_input = dlarray(net_input, 'SSCB');

end
