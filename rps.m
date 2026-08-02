clc; clear; close all;

% game
n = 3;  % number of strategies
w = 3;  % win reward
l = 1;  % loss cost
F = @(x) [0, -l, w;
    w, 0, -l;
    -l, w, 0]*x;    % payoff function

% replicator dynamics
f = @(t,x) x.*(F(x) - x'*F(x));   % xdot = f(x)

% simulate dynamics
tspan = [0,50]; % time span
x0 = rand(n,1); x0 = x0/sum(x0);    % initial state
[t,x] = ode45(f, tspan, x0);
x = x'; % make x have nice size

% plot in 3D simplex
figure(); box on; hold on;
fill3([1,0,0],[0,1,0],[0,0,1],'k','FaceAlpha',0.5); % draw simplex
plot3(x(1,:),x(2,:),x(3,:),'w','LineWidth',2);
view([1,1,1]);