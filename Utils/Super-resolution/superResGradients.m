function [loss, gradients] = superResGradients(net, noisy, clean)
    predictions = forward(net, noisy);   % Usar 'forward', no 'predict'
    factor = 4;sigma = 25;n_planes = 3; kernel_type = 'lanczos2'; phase = 0; kernel_width = 4 * factor + 1; support = 2; preserve_size = false;
    down = Downsampler(n_planes, factor, kernel_type, phase, kernel_width, support, sigma, preserve_size);
    predictions_LR = apply_downsampler(down, predictions);
    loss = mse(predictions_LR, clean);      % CORREGIT: usar predictions_LR en lloc de predictions
    gradients = dlgradient(loss, net.Learnables);  % Derivar w.r.t. pesos
end