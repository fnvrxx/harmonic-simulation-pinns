function [F, J] = residual_and_jac(w, net_template, x_physics)
% Fungsi residual + Jacobian untuk fsolve (Levenberg-Marquardt).
%
% Input:
%   w            : flat parameter vector (1 × N_params) dari fsolve
%   net_template : dlnetwork kosong sebagai template struktur
%   x_physics    : titik kolokasi dlarray (1 × N, format 'CB')
%
% Output:
%   F : vektor residual (N × 1)
%   J : Jacobian (N × N_params),  ∂F_i/∂w_j  via AD

    % Unpack w ke dalam network
    net = unpack_params(net_template, w);

    % Hard constraint ansatz: U = p·f
    % x_physics format 'CB': (1 × N) — transpose untuk operasi elemen
    p    = x_physics .* (pi - x_physics);
    p_x  = pi - 2*x_physics;
    p_xx = dlarray(-2 * ones(size(x_physics)));

    x_dl = dlarray(extractdata(x_physics), 'CB');

    % Forward pass — perlu EnableHigherDerivatives karena akan diturunkan lagi
    f    = forward(net, x_dl);
    f_x  = dlgradient(sum(f),   x_dl, 'EnableHigherDerivatives', true);
    f_xx = dlgradient(sum(f_x), x_dl, 'EnableHigherDerivatives', true);

    U    = p    .* f;
    U_xx = p_xx .* f + 2*p_x .* f_x + p .* f_xx;

    f_force  = -sin(x_dl) + sin(x_dl).^3;
    residual = U_xx + U.^3 - f_force;   % (N × 1) dlarray

    F = double(extractdata(residual))';  % (N×1) kolom untuk fsolve

    if nargout > 1
        % Full Jacobian: ∂residual_i/∂w  via AD loop per baris
        N      = numel(F);
        P      = numel(w);
        J      = zeros(N, P);

        params = net.Learnables.Value;

        for i = 1:N
            g_cell = dlgradient(residual(i), params, 'EnableHigherDerivatives', false);
            col = 1;
            for c = 1:numel(g_cell)
                gv  = double(extractdata(g_cell{c}));
                n   = numel(gv);
                J(i, col:col+n-1) = gv(:)';
                col = col + n;
            end
        end
    end
end
