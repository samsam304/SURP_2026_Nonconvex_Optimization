clc; clear; close all;

% Global
S = (-2:0.01:2)';
n = numel(S);
x0 = rand(n,1); x0 = x0/sum(x0);    % Initial state (probability distribution)
tspan = [0,10];

% Objective minimization
obj = @(x) (3/2)*x.^4 - (1/4)*x.^3 - 3*x.^2 + (3/4)*x + 1;

F = @(x) x'*obj(S)-obj(S);          % Vector payoff

f = @(t,x) x.*(F(x) - x'*F(x));     % Replicator dynamics

[t1,x1] = ode45(f, tspan, x0);      % Simulate dynamics
x1 = x1';

% Approximation minimization
q = @(x) (27/4)*(x+1).^2 - 1;

Fq = @(x) x'*q(S)-q(S);             % Vector payoff

fq = @(t,x) x.*(Fq(x) - x'*Fq(x));  % Replicator dynamics

[t2,x2] = ode45(fq, tspan, x0);     % Simulate dynamics
x2 = x2';

% Plot: Distribution Evolution
x1plot = x1 ./ sum(x1,1);
x2plot = x2 ./ sum(x2,1);

dS = S(2)-S(1);
x1density = x1plot / dS;
x2density = x2plot / dS;

figure;

subplot(1,2,1)
surf(t1,S,x1density,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Replicator dynamics for objective');
grid on;

subplot(1,2,2)
surf(t2,S,x2density,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Replicator dynamics for approximation');
grid on;

% Plot: Mean Trajectories
m1 = S' * x1plot;           % mean trajectory for f
m2 = S' * x2plot;           % mean trajectory for q

figure;

plot(t1,m1,'LineWidth',1.5);
hold on;
plot(t2,m2,'LineWidth',1.5);
yline(-1,'w--','LineWidth',1.2);
hold off;

xlabel('Time t');
ylabel('Mean strategy m(t)');
title('Mean trajectories for f and q');
legend('m(t) for f','m_q(t) for q','Global minimizer s = -1');
grid on;