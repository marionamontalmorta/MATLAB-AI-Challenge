function [loss, gradients] = denoisingGradients(net, noisy, clean)
    predictions = forward(net, noisy);   % Usar 'forward', no 'predict'
    loss = mse(predictions, clean);      % Loss escalar
    gradients = dlgradient(loss, net.Learnables);  % Derivar w.r.t. pesos
end