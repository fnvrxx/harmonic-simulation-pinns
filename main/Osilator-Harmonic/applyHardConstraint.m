function x_hat = applyHardConstraint(model, t, x0, v0)
    % Hard constraint: x_hat(0) = x0, dx_hat/dt(0) = v0
    % Transformasi: x_hat(t) = x0 + v0*t + t^2 * NN(t)
    % Pada t=0: x_hat = x0 (terpenuhi secara eksak, tidak perlu dilatih)
    nn_out = forward(model, t);
    x_hat  = x0 + v0 .* t + t.^2 .* nn_out;
end
