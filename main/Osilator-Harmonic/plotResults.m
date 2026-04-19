function plotResults(model, t_true, u_exact, t_obs, u_obs)
    u0 = 1;
    v0 = 0;

    t_true_dl = dlarray(t_true(:)', 'CB');
    u_pred    = extractdata(applyHardConstraint(model, t_true_dl, u0, v0));

    figure;
    plot(t_true, u_exact, 'k-', 'LineWidth', 2); hold on;
    plot(t_true, u_pred,  'b--', 'LineWidth', 2);
    scatter(t_obs, u_obs, 80, 'r', 'filled');

    legend('Solusi Analitik (LaTeX Eq.4)', 'Prediksi PINN (Hard Constraint)', 'Observasi', 'Location', 'northeast');
    xlabel('Waktu t'); ylabel('u(t)');
    title('PINNs - Osilator Harmonik Teredam (\delta=2, \omega_0=20)');
    grid on;
end
