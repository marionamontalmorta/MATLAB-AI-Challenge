function gradients = clipGradients(gradients, threshold)
    % Clips the gradients if their global L2 norm exceeds the 'threshold'.

    % 1. Calculate the global L2 norm of all gradients
    globalNorm = 0;
    params = fieldnames(gradients);
    for i = 1:numel(params)
        % Ensure we only process numeric gradients
        if isnumeric(gradients.(params{i}))
            globalNorm = globalNorm + sum(gradients.(params{i})(:).^2);
        end
    end
    globalNorm = sqrt(globalNorm);
    
    % 2. Scale the gradients if the norm exceeds the threshold
    if globalNorm > threshold
        % Calculate the scaling factor
        scaleFactor = threshold / (globalNorm + 1e-6); % Add epsilon for numerical stability
        
        % Apply the scaling to all gradients
        for i = 1:numel(params)
            if isnumeric(gradients.(params{i}))
                gradients.(params{i}) = gradients.(params{i}) * scaleFactor;
            end
        end
    end
end