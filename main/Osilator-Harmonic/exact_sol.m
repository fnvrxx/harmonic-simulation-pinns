% Solusi analitik osilator harmonik teredam (sesuai LaTeX Eq. 4)
% u(t) = e^(-delta*t) * 2A * cos(phi + omega*t)
% Parameter: delta=2, omega0=20, domain t in [0,1]
% IC: u(0)=1, du/dt(0)=0

clear; clc; close all;

d  = 2;
w0 = 20;
t  = 1;
time = 0:0.01:t;

w   = sqrt(w0^2 - d^2);
phi = atan(-d/w);
A   = 1 / (2*cos(phi));

u = exp(-d*time) .* 2*A .* cos(phi + w*time);

plot(time, u, 'b-', 'LineWidth', 2);
xlabel('t (s)');
ylabel('u(t)');
title('Osilator Harmonik Teredam - Solusi Exact');
grid on;
