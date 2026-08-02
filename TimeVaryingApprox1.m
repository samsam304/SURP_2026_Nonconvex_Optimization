clc; clear; close all;

% Global
S = (-2:0.02:2)';
n = numel(S);

x0 = rand(n,1);
x0 = x0/sum(x0);

tspan = [0,20];

% Objective and derivatives
obj = @(x) (3/2)*x.^4 - (1/4)*x.^3 - 3*x.^2 + (3/4)*x + 1;

dobj = @(x) 6*x.^3 - (3/4)*x.^2 - 6*x + 3/4;

ddobj = @(x) 18*x.^2 - (3/2)*x - 6;

%% Original objective dynamics

objectiveValues = obj(S);

F = @(x) x' * objectiveValues - objectiveValues;
f = @(t,x) x .* (F(x) - x' * F(x));

[tf,xf] = ode45(f,tspan,x0);
xf = xf';

% Normalize each population state
xfprob = xf ./ sum(xf,1);

% Mean at each output time
mf = S' * xfprob;

%% Time-varying quadratic dynamics

% Continuous approximation of the original mean trajectory
mfAtTime = @(t) interp1(tf,mf(:),t,'pchip');

fq = @(t,x) movingQuadraticDynamics( ...
    t,x,S,obj,dobj,ddobj,mfAtTime);

[tq,xq] = ode45(fq,tspan,x0);
xq = xq';

function dx = movingQuadraticDynamics( ...
    t,x,S,obj,dobj,ddobj,mfAtTime)

% Normalize the current approximation population
p = x/sum(x);

% Original AGRF mean at the current ODE time
mfCurrent = mfAtTime(t);

% Evaluate the current Taylor approximation at every strategy
delta = S - mfCurrent;

qS = obj(mfCurrent) ...
    + dobj(mfCurrent) .* delta ...
    + 0.5 .* ddobj(mfCurrent) .* delta.^2;

% qS is n-by-1
averageQ = p' * qS;
Fq = averageQ - qS;

% Replicator dynamics
averagePayoff = p' * Fq;
dx = x .* (Fq - averagePayoff);
end

%% Plot: Distribution Evolution
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
title('Replicator dynamics on objective');
grid on;

subplot(1,2,2)
surf(tq,S,xqdensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Replicator dynamics on approximation about objective mean');
grid on;

%% Plot: Mean Trajectories
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
title('Mean trajectories for f and q');
legend('m(t) for f','m_q(t) for q','Global minimizer s = -1');
grid on;