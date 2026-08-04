clc; clear; close all;

%% Global
S = (-2:0.01:2)';
n = numel(S);

% % Uniform initial distribution
% x0 = (1/n) .* ones(n,1); 

% Custom initial distribution
a = 1;
m0 = a * ones(n,1);

sigma0 = 2;

dist2 = (S - m0).^2;

x0 = exp(-dist2/(2*sigma0^2));
x0 = x0/sum(x0);

% Time
tspan = [0,60];

%% Objective minimization and derivatives
g = @(s) (3/2) * s.^4 - (1/4) * s.^3 - 3 * s.^2 + (3/4) * s + 1;
dg = @(s) 6 * s.^3 - (3/4) * s.^2 - 6 * s + 3/4;
ddg = @(s) 18 * s.^2 - (3/2) * s - 6;

F = @(x) x' * g(S) - g(S);              % Vector payoff

f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics

[tf,xf] = ode45(f, tspan, x0);          % Simulate dynamics
xf = xf';

%% Approximation minimization
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
xfplot = xf ./ sum(xf,1);
xqplot = xq ./ sum(xq,1);

dS = S(2)-S(1);
xfdensity = xfplot / dS;
xqdensity = xqplot / dS;

figure;

subplot(1,2,1)
surf(tf,S,xfdensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Replicator dynamics for objective');
grid on;

subplot(1,2,2)
surf(tq,S,xqdensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Replicator dynamics for approximation');
grid on;

% Mean Trajectories
mf = S' * xfplot;           % mean trajectory for f
mq = S' * xqplot;           % mean trajectory for q

figure;

plot(tf,mf,'LineWidth',1.5);
hold on;
plot(tq,mq,'LineWidth',1.5);
yline(-1,'w--','LineWidth',1.2);
hold off;

xlabel('Time t');
ylabel('Mean strategy m(t)');
ylim([-2,2])
title('Mean trajectories for f and q');
legend('m(t) for g','m_q(t) for q','Global minimizer s = -1');
grid on;