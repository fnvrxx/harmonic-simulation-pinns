%% =========================================================
%  FUNGSI LOSS (dipanggil oleh dlfeval untuk automatic diff)
%% =========================================================
function [loss, grads] = computeLoss(net, t_boundary, t_physics, mu, k, lambda_bc)

    % ---- BOUNDARY LOSS ----
    % Kondisi awal: u(0) = 1
    u0 = forward(net, t_boundary);                     % prediksi u(0)
    loss_u0 = (u0 - 1).^2;                            % MSE: u(0) harus = 1

    % Kondisi awal: u'(0) = 0  --> turunan pertama di t=0
    du0 = dlgradient(sum(u0), t_boundary);             % du/dt di t=0
    loss_du0 = (du0 - 0).^2;                           % u'(0) harus = 0

    loss_boundary = loss_u0 + loss_du0;

    % ---- PHYSICS LOSS ----
    % Forward pass di titik-titik physics
    u = forward(net, t_physics);                       % u(t)

    % Turunan pertama: du/dt
    du_dt = dlgradient(sum(u), t_physics, 'EnableHigherDerivatives', true);

    % Turunan kedua: d²u/dt²
    d2u_dt2 = dlgradient(sum(du_dt), t_physics);

    % Residual persamaan: d²u/dt² + mu*du/dt + k*u = 0
    residual = d2u_dt2 + mu .* du_dt + k .* u;
    loss_physics = mean(residual.^2);

    % ---- JOINT LOSS ----
    % Backpropagate joint loss, take optimiser step
    loss = lambda_bc * loss_boundary + loss_physics;

    % Hitung gradien terhadap parameter network
    grads = dlgradient(loss, net.Learnables);
end
