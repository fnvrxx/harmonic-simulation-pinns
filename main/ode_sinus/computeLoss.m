function [loss, grads, loss_phys, loss_jac] = computeLoss(net, x_physics, alpha)
% Hard constraint ansatz: U(x) = x·(π-x)·f(x;θ)
% Menjamin U(0)=0 dan U(π)=0 secara otomatis (Dirichlet BC).
%
% ODE: U'' + U^3 = -sin(x) + sin^3(x)
% x_physics: dlarray format 'CB' (1 × N)
%
% Loss gabungan:
%   loss = mean(residual²) + alpha · ||J_full||_F²
%   J_full = ∂residual/∂w  (N_colloc × N_params), dihitung via AD loop

    % Ansatz hard constraint
    p    = x_physics .* (pi - x_physics);         % p(x)  = x(π-x)
    p_x  = pi - 2*x_physics;                      % p'(x) = π-2x
    p_xx = dlarray(-2 * ones(size(x_physics)));   % p''(x) = -2

    % Forward pass: f = network output (tanpa BC)
    % x_physics format 'CB' sudah sesuai featureInputLayer
    f = forward(net, x_physics);

    % Turunan f via AD
    f_x  = dlgradient(sum(f),   x_physics, 'EnableHigherDerivatives', true);
    f_xx = dlgradient(sum(f_x), x_physics, 'EnableHigherDerivatives', true);

    % U = p·f,  U' = p'f + pf',  U'' = p''f + 2p'f' + pf''
    U    = p    .* f;
    U_xx = p_xx .* f + 2*p_x .* f_x + p .* f_xx;

    % Residual ODE: U'' + U^3 - (-sin(x) + sin^3(x)) = 0
    f_force  = -sin(x_physics) + sin(x_physics).^3;
    residual = U_xx + U.^3 - f_force;
    loss_phys = mean(residual.^2);

    % --- Full Jacobian ||∂residual/∂w||_F² via AD loop ---
    N      = numel(residual);
    params = net.Learnables.Value;

    jac_frob_sq = dlarray(0);
    for i = 1:N
        g_cell = dlgradient(residual(i), params, 'EnableHigherDerivatives', true);
        for c = 1:numel(g_cell)
            gv = g_cell{c};
            jac_frob_sq = jac_frob_sq + sum(gv(:).^2);
        end
    end
    loss_jac = jac_frob_sq / N;

    % Loss gabungan
    loss  = loss_phys + alpha * loss_jac;

    grads = dlgradient(loss, net.Learnables);
end
