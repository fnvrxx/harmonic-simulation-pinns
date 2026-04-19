function [loss, gradients] = modelGradients(model, t_data, u_data, t_pinn, m, mu, k)
    % IC: u(0)=1, du/dt(0)=0
    u0 = 1;
    v0 = 0;

    % Loss data: prediksi dengan hard constraint
    u_pred = applyHardConstraint(model, t_data, u0, v0);
    loss_data = mean((u_pred - u_data).^2, 'all');

    % Loss fisika: residual ODE => m*u'' + mu*u' + k*u = 0 (LaTeX Eq. 1)
    lambda2 = 1e-4;
    u_pinn  = applyHardConstraint(model, t_pinn, u0, v0);

    du_dt  = dlgradient(sum(u_pinn, 'all'), t_pinn, 'EnableHigherDerivatives', true);
    du_dt2 = dlgradient(sum(du_dt, 'all'), t_pinn);

    residual     = m * du_dt2 + mu * du_dt + k * u_pinn;
    loss_physics = lambda2 * mean(residual.^2, 'all');

    loss = loss_data + loss_physics;

    gradients = dlgradient(loss, model.Learnables);
end
