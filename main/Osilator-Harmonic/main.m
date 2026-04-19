% Required: Deep Learning Toolbox
% Osilator harmonik teredam - sesuai LaTeX problem statement
% Persamaan: m*u'' + mu*u' + k*u = 0
% Parameter: delta=2, omega0=20, m=1 => mu=4, k=400
% IC: u(0)=1, du/dt(0)=0, domain t in [0,1]

clear; clc; close all;

% --- Parameter (notasi sesuai LaTeX) ---
m      = 1;
delta  = 2;
omega0 = 10; % untuk omega = 5, sudah konvergen
mu     = 2 * m * delta;       % mu = 4
k      = omega0^2 * m;        % k  = 400
T_END  = 1;

% --- Collocation & data points ---
t_data = linspace(0, T_END, 50)';
t_pinn = linspace(0, T_END, 10000)';

% Solusi exact sebagai data referensi (LaTeX Eq. 4)
omega = sqrt(omega0^2 - delta^2);
phi   = atan(-delta / omega);
A     = 1 / (2 * cos(phi));
u_data = exp(-delta*t_data) .* 2*A .* cos(phi + omega*t_data);

t_data = dlarray(t_data', 'CB');
u_data = dlarray(u_data', 'CB');
t_pinn = dlarray(t_pinn', 'CB');

% --- Arsitektur Neural Network ---
layers = [
    featureInputLayer(1, "Normalization", "none", "Name", "input")
    fullyConnectedLayer(32, "Name", "fc1")
    tanhLayer("Name", "tanh1")
    fullyConnectedLayer(32, "Name", "fc2")
    tanhLayer("Name", "tanh2")
    fullyConnectedLayer(1, "Name", "output")
];

trainer = pinn_train(layers, 1e-3);

% --- Training loop ---
numEpochs = 15000;
for i = 1:numEpochs
    trainer.trainStep(t_data, u_data, t_pinn, m, mu, k);

    if mod(i, 100) == 0
        disp("Epoch " + i + ": Loss = " + trainer.lossValues(end));
    end
    if abs(trainer.lossValues(end)) < 1e-4
        disp("Konvergen pada epoch " + i);
        break;
    end
end

% --- Plot hasil ---
t_true  = linspace(0, T_END, 1000)';
u_exact = exp(-delta*t_true) .* 2*A .* cos(phi + omega*t_true);

idx_obs = round(linspace(1, length(t_true), 10));
t_obs   = t_true(idx_obs);
u_obs   = u_exact(idx_obs);

plotResults(trainer.model, t_true, u_exact, t_obs, u_obs);
