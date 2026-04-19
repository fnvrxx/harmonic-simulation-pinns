function x = generateAnalyticalSolution(c, k, t, x0, v0)
    if nargin < 4, x0 = 1; end
    if nargin < 5, v0 = 0; end

    omega_n = sqrt(k);
    zeta    = c / (2 * omega_n);

    if zeta < 1
        omega_d = omega_n * sqrt(1 - zeta^2);
        A = x0;
        B = (v0 + zeta * omega_n * x0) / omega_d;
        x = exp(-zeta * omega_n * t) .* (A * cos(omega_d * t) + B * sin(omega_d * t));
    elseif zeta == 1
        % Critically damped
        x = exp(-omega_n * t) .* ((x0) + (v0 + omega_n * x0) * t);
    else
        % Overdamped
        r1 = omega_n * (-zeta + sqrt(zeta^2 - 1));
        r2 = omega_n * (-zeta - sqrt(zeta^2 - 1));
        C2 = (v0 - r1*x0) / (r2 - r1);
        C1 = x0 - C2;
        x  = C1 * exp(r1*t) + C2 * exp(r2*t);
    end
end
