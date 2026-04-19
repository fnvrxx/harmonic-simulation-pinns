%% PINN - ODE Sinus (Nonlinear) — Optimizer: Levenberg-Marquardt
% Persamaan: u''(x) + u^3(x) = -sin(x) + sin^3(x),  x ∈ [0, π]
% Kondisi batas: u(0) = 0, u(π) = 0  (Dirichlet)
% Solusi eksak: u(x) = sin(x)
%
% Hard constraint ansatz: U(x) = x·(π-x)·f(x;θ)
% Optimizer: fsolve dengan algoritma Levenberg-Marquardt
% Jacobian: ∂residual/∂w dihitung via AD loop (full, N × P)

clc;
clear;
close all;

rng(42);

%% 1. Solusi Analitik
N_test = 500;
[u_exact, x_test] = exact_sol(N_test);

%% 2. Titik Kolokasi Training
N_phys     = 30;
x_phys_vec = linspace(0, pi, N_phys);   % (1 × N) row vector
x_physics  = dlarray(x_phys_vec, 'CB'); % format 'CB': channel × batch

%% 3. Inisialisasi Network
net = nn(15);

%% 4. Setup fsolve — Levenberg-Marquardt
options = optimoptions('fsolve', ...
    'Algorithm',               'levenberg-marquardt', ...
    'SpecifyObjectiveGradient', true, ...   % aktifkan penyediaan Jacobian
    'Display',                 'iter', ...
    'FunctionTolerance',        1e-12, ...
    'StepTolerance',            1e-12, ...
    'MaxIterations',            1000, ...
    'MaxFunctionEvaluations',   1e6);

% Ekstrak parameter awal sebagai flat vector
w0 = pack_params(net);

% Fungsi residual+Jacobian dengan net sebagai template
objFun = @(w) residual_and_jac(w, net, x_physics);

%% 5. Training — LM via fsolve
fprintf('Mulai training Levenberg-Marquardt...\n');
fprintf('Jumlah parameter: %d,  titik kolokasi: %d\n\n', numel(w0), N_phys);

tic;
[w_opt, ~, exitflag, output] = fsolve(objFun, w0, options);
elapsed = toc;

fprintf('\nWaktu training : %.2f detik\n', elapsed);
fprintf('Exit flag      : %d  (%s)\n', exitflag, output.message);
fprintf('Iterasi        : %d\n', output.iterations);

%% 6. Prediksi
net_opt = unpack_params(net, w_opt);
x_dl    = dlarray(x_test', 'CB');        % (1 × N_test)
p_test  = x_test .* (pi - x_test);
f_pred  = extractdata(forward(net_opt, x_dl));
u_pred  = p_test .* f_pred;

%% 7. Evaluasi Error
rel_L2 = sqrt(sum((u_pred - u_exact).^2)) / sqrt(sum(u_exact.^2));
mae    = mean(abs(u_pred - u_exact));
res_loss = mean(residual_and_jac(w_opt, net, x_physics).^2);

fprintf('\n--------------------------------------------------\n');
fprintf('  HASIL:\n');
fprintf('    Relative L2 Error  : %.6e\n', rel_L2);
fprintf('    Mean Absolute Error: %.6e\n', mae);
fprintf('    Residual Loss      : %.6e\n', res_loss);
fprintf('--------------------------------------------------\n');

%% 8. Visualisasi
figure;

subplot(2,1,1);
plot(x_test, u_exact, 'b-',  'LineWidth', 2, 'DisplayName', 'Solusi Eksak');
hold on;
plot(x_test, u_pred,  'r--', 'LineWidth', 2, 'DisplayName', 'Prediksi PINN');
xlabel('x');
ylabel('u(x)');
title('PINN (LM) vs Solusi Analitik - ODE Sinus');
legend;
grid on;

subplot(2,1,2);
plot(x_test, abs(u_pred - u_exact), 'r-', 'LineWidth', 1.5);
xlabel('x');
ylabel('|u_{pred} - u_{exact}|');
title('Absolute Error');
grid on;
