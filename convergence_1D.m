clc; clear; close all;

%% Global
S = (-2:0.01:2)';
n = numel(S);

% Plotting options
axis_label_font_size = 12;
title_font_size = 20;

% Uniform initial distribution
mu0 = (1/n) .* ones(n,1); 

% % Custom initial distribution
% a = 1;
% m0 = a * ones(n,1);
% 
% sigma0 = 0.3;
% 
% dist2 = (S - m0).^2;
% 
% mu0 = exp(-dist2/(2*sigma0^2));
% mu0 = mu0/sum(mu0);

% Time
tspan = [0,60];

%% Objective minimization and derivatives
g = @(s) (3/2) * s.^4 - (1/4) * s.^3 - 3 * s.^2 + (3/4) * s + 1;
dg = @(s) 6 * s.^3 - (3/4) * s.^2 - 6 * s + 3/4;
ddg = @(s) 18 * s.^2 - (3/2) * s - 6;

F = @(x) x' * g(S) - g(S);              % Vector payoff

f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics

[tg,mu_g] = ode45(f, tspan, mu0);          % Simulate dynamics
mu_g = mu_g';

%% Global quadratic approximation 
sStar = -1;         % Global minimizer

qg = @(s) (27/4) * (s + 1).^2 - 1;

Fqg = @(x) x' * qg(S) - qg(S);

fqg = @(t,x) x .* (Fqg(x) - x' * Fqg(x));

[tqg,mu_qg] = ode45(fqg, tspan, mu0);
mu_qg = mu_qg';

%% Time-varying quadratic approximation
m = @(x) x' * S;

q = @(m,s) g(m) + ...
    dg(m) * (s - m) + ...
    0.5 * ddg(m) * (s - m).^2;

Fq = @(x) x' * q(m(x),S) - q(m(x),S);

fq = @(t,x) x .* (Fq(x) - x' * Fq(x));

[tq,mu_q] = ode45(fq, tspan, mu0);
mu_q = mu_q';

%% Plots
% Set default interpreter as LaTeX
set(0,'defaultTextInterpreter','latex');

%% Plot 1
% Distribution Evolution
dS = S(2)-S(1);
mu_gDensity = mu_g / dS;
mu_qgDensity = mu_qg / dS;
mu_qDensity = mu_q / dS;

figure;
box on;

tl = tiledlayout(2,3, ...
    'TileSpacing','compact', ...
    'Padding','compact');

title(tl, '\textbf{Distribution Flows for Quartic Objective}', ...
    'FontSize', title_font_size, ...
    'Interpreter','latex');

skip = 1;   % Speeds up plot vectorization

% Nonconvex objective
nexttile;

surf(tg(1:skip:end), ...
    S(1:skip:end), ...
    mu_gDensity(1:skip:end,1:skip:end), ...
    'EdgeColor','none');

title('Nonconvex', 'FontSize', 12);
set_latex_axes(axis_label_font_size);

% Adaptive quadratic approximation
% Span both columns of the second row
nexttile([2,2]);

surf(tq(1:skip:end), ...
    S(1:skip:end), ...
    mu_qDensity(1:skip:end,1:skip:end), ...
    'EdgeColor','none');

title('Adaptive Approximation', 'FontSize', 18);
set_latex_axes(axis_label_font_size);
zlim([0,12]);

% Optimal quadratic approximation
nexttile;

surf(tqg(1:skip:end), ...
    S(1:skip:end), ...
    mu_qgDensity(1:skip:end,1:skip:end), ...
    'EdgeColor','none');

title('Optimal Approximation', 'FontSize', 12);
set_latex_axes(axis_label_font_size);

% % Saving graphics
% % For report 
% exportgraphics(gcf, ...
%     'convergence_1D_distributions.pdf', ...
%     'BackgroundColor','none', ...
%     'ContentType','auto');

% % For poster --- change skip back to 2... or not (~12 min comp time)
% exportgraphics(gcf, ...
%     'convergence_1D_distributions.pdf', ...
%     'BackgroundColor','none', ...
%     'ContentType','vector');

%% Plot 2
% Mean convergence rates
mg = S' * mu_g;
mqg = S' * mu_qg;
mq = S' * mu_q;

err_g  = abs(mg(:)  - sStar);
err_qg = abs(mqg(:) - sStar);
err_q  = abs(mq(:)  - sStar);

epsilon = max(err_g) * 2e-2;    % 2% settling time

idx_g  = settlingIndex(err_g,  epsilon);
idx_qg = settlingIndex(err_qg, epsilon);
idx_q  = settlingIndex(err_q,  epsilon);

% End plot after final convergence
if isnan(idx_q)
    T = ceil(max([tg(idx_g),tqg(idx_qg)]));
else 
    T = ceil(max([tg(idx_g),tqg(idx_qg),tq(idx_q)]));
end

figure; box on; hold on; grid on;
title('$\textbf{Mean Trajectory for Quartic Objective}$','FontSize', title_font_size);

% Convergence region
cRegion = fill([0,0,tspan(2),tspan(2)],[sStar-epsilon,sStar+epsilon,...
    sStar+epsilon,sStar-epsilon],'g','linestyle','none');

% Mean trajectories
mgTraj = plot(tg,mg,'k','LineWidth',3);
mqgTraj = plot(tqg,mqg,'--k','LineWidth',3);
mqTraj = plot(tq,mq,':k','LineWidth',3);

% Region entry points
if ~isnan(idx_g)
    mgPoint = plot(tg(idx_g),mg(idx_g),'o', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mgPoint = plot(NaN,NaN,'o', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

if ~isnan(idx_qg)
    mgqPoint = plot(tqg(idx_qg),mqg(idx_qg),'s', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mgqPoint = plot(NaN,NaN,'s', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

if ~isnan(idx_q)
    mqPoint = plot(tq(idx_q),mq(idx_q),'^', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mqPoint = plot(NaN,NaN,'^', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

xlabel('Time', 'fontsize', axis_label_font_size);
xlim([0,T]);
ylabel('Mean Strategy', 'fontsize', axis_label_font_size);

xtickformat('$%g$');
ytickformat('$%g$');
ztickformat('$%g$');
set(gca,'ticklabelinterpreter','latex','fontsize',axis_label_font_size);

legend([mgTraj,mqgTraj,mqTraj,mgPoint,mgqPoint,mqPoint,cRegion], ...
    'Nonconvex', ...
    'Optimal Approximation', ...
    'Adaptive Approximation', ...
    'Settling Time, Nonconvex', ...
    'Settling Time, Optimal Approximation', ...
    'Settling Time, Adaptive Approximation', ...
    '$2\%$ Settling Time Tube', ...
    'interpreter','latex','fontsize',axis_label_font_size,'backgroundalpha',0.9);

% % Saving graphics
% tightfig();
% exportgraphics(gcf,...
%     'convergence_1D_meanTraj.pdf',...
%     'BackgroundColor', 'none',...
%     'ContentType', 'vector');

%% Functions
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

function set_latex_axes(axis_label_font_size)
    % call this function to set a current plot's axes to use latex tick
    % labels and correct axis labels
    grid on;
    
    xlabel('Time', 'fontsize', axis_label_font_size);
    ylabel('Strategy', 'fontsize', axis_label_font_size);
    zlabel('Density', 'fontsize', axis_label_font_size);
    
    xtickformat('$%g$');
    ytickformat('$%g$');
    ztickformat('$%g$');
    set(gca,'ticklabelinterpreter','latex','fontsize',axis_label_font_size);
end