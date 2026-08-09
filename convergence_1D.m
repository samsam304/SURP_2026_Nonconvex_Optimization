clc; clear; close all;

%% Global
S = (-2:0.01:2)';
n = numel(S);

% Uniform initial distribution
x0 = (1/n) .* ones(n,1); 

% % Custom initial distribution
% a = 1;
% m0 = a * ones(n,1);
% 
% sigma0 = 0.3;
% 
% dist2 = (S - m0).^2;
% 
% x0 = exp(-dist2/(2*sigma0^2));
% x0 = x0/sum(x0);

% Time
tspan = [0,10];

%% Objective minimization and derivatives
g = @(s) (3/2) * s.^4 - (1/4) * s.^3 - 3 * s.^2 + (3/4) * s + 1;
dg = @(s) 6 * s.^3 - (3/4) * s.^2 - 6 * s + 3/4;
ddg = @(s) 18 * s.^2 - (3/2) * s - 6;

F = @(x) x' * g(S) - g(S);              % Vector payoff

f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics

[tg,xg] = ode45(f, tspan, x0);          % Simulate dynamics
xg = xg';

%% Approximation minimization centered at global minima
qg = @(s) (27/4) * (s + 1).^2 - 1;

Fqg = @(x) x' * qg(S) - qg(S);             % Vector payoff

fqg = @(t,x) x .* (Fqg(x) - x' * Fqg(x));  % Replicator dynamics

[tqg,xqg] = ode45(fqg, tspan, x0);         % Simulate dynamics
xqg = xqg';

%% Approximation minimization with time varying center
m = @(x) x' * S;

q = @(m,s) g(m) + ...
    dg(m) * (s - m) + ...
    0.5 * ddg(m) * (s - m).^2;

Fq = @(x) x' * q(m(x),S) - q(m(x),S);   % Vector payoff

fq = @(t,x) x .* (Fq(x) - x' * Fq(x));  % Replicator dynamics

[tq,xq] = ode45(fq, tspan, x0);         % Simulate dynamics
xq = xq';

%% Plots
% Distribution Evolution
dS = S(2)-S(1);
xgDensity = xg / dS;
xqgDensity = xqg / dS;
xqDensity = xq / dS;

figure;

subplot(2,2,1)
surf(tg,S,xgDensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Objective');
grid on;

subplot(2,2,2)
surf(tqg,S,xqgDensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Approximation at Optimizer');
grid on;

subplot(2,2,[3 4])
surf(tq,S,xqDensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Time Varying Approximation');
grid on;

sgtitle('Strategy Distribution Updates via Replicator Dynamics')

% Mean convergence rates
mg = S' * xg;
mqg = S' * xqg;
mq = S' * xq;

sStar = -1;
epsilon = 2e-2;

err_g  = abs(mg(:)  - sStar);
err_qg = abs(mqg(:) - sStar);
err_q  = abs(mq(:)  - sStar);

idx_g  = settlingIndex(err_g,  epsilon);
idx_qg = settlingIndex(err_qg, epsilon);
idx_q  = settlingIndex(err_q,  epsilon);

figure;

plot(tg,mg,'LineWidth',2,'Color','#1D2DCF');
hold on;
plot(tqg,mqg,'LineWidth',2,'Color','#CF1DB1');
plot(tq,mq,'LineWidth',2,'Color','#AE1DCF');

yline(sStar,'k--','LineWidth',1.5);
yline(sStar + epsilon,'k:','LineWidth',1.5);
yline(sStar - epsilon,'k:','LineWidth',1.5);

if ~isnan(idx_g)
    plot(tg(idx_g),mg(idx_g),'o', ...
        'MarkerSize',8,'LineWidth',1.5,'Color','#1DCF2C');
end

if ~isnan(idx_qg)
    plot(tqg(idx_qg),mqg(idx_qg),'s', ...
        'MarkerSize',8,'LineWidth',1.5,'Color','#1D85CF');
end

if ~isnan(idx_q)
    plot(tq(idx_q),mq(idx_q),'^', ...
        'MarkerSize',8,'LineWidth',1.5,'Color','#1DCCCF');
end

xlabel('Time t');
ylabel('Mean strategy m(t)');
title('Mean trajectories and convergence points');
grid on;
hold off;

legend('m(t) for g', ...
    'm_{qg}(t) centered at optimizer', ...
    'm_q(t) for time-varying q', ...
    'Global minimizer s = -1', ...
    '\epsilon-neighborhood', ...
    '', ...
    'Convergence time for m(t)', ...
    'Convergence time for m_{qg}(t) centered at optimizer', ...
    'Convergence time for m_q(t) for time-varying');

%% Functions
% 1
function idx = settlingIndex(err, epsilon)
err = err(:);

% Last recorded point outside the epsilon-neighborhood
lastOutside = find(err > epsilon, 1, 'last');

if isempty(lastOutside)
    % Already inside the neighborhood at the initial time
    idx = 1;
elseif lastOutside == numel(err)
    % Did not settle during the simulation
    idx = NaN;
else
    % First point after the final tolerance violation
    idx = lastOutside + 1;
end
end