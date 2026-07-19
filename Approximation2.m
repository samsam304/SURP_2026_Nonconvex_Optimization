clc; clear; close all;

% Global
S = (-4:0.2:4)';
[X,Y] = meshgrid(S);
strategies = [X(:),Y(:)];           % Each row is a strategy (s1,s2)

% Inital distributions 
startingPoints = {[-2; 2],[2; 2.5]};
p = numel(startingPoints);
distributions = {};

for k = 1:p
    m0 = startingPoints{k};

    sigma0 = 1;
    
    dist2 = (strategies(:,1) - m0(1)).^2 ...
          + (strategies(:,2) - m0(2)).^2;
    
    x0 = exp(-dist2/(2*sigma0^2));
    x0 = x0/sum(x0);

    distributions{end+1} = {x0,sigma0};
end

% Objective minimization (2D Styblinski-Tang) 
ST = @(x,y) (1/2)*(x.^4 - 16*x.^2 + 5*x + y.^4 - 16*y.^2 + 5*y);

% Approximation minimization
sStar = [-2.903534;-2.903534];      % 2D global minimizer

q = @(x,y) -78.3323 + 9.60412*10^(-7)*(2.90353 + x) + 17.2915*(2.90353 + x).^2 + ...
    9.60412*10^(-7)*(2.90353 + y) + 17.2915*(2.90353 + y).^2;

% Values on strategy grid for dynamics
objValues = ST(strategies(:,1),strategies(:,2));
qValues = q(strategies(:,1),strategies(:,2));

tspan = [0,10];
solutions = {};

for k = 1:p
    x0 = distributions{k}{1};
    
    F = @(x) x'*objValues - objValues;  % Vector payoff
    f = @(t,x) x.*(F(x) - x'*F(x));     % Replicator dynamics
    
    [tf,xf] = ode45(f, tspan, x0);      % Simulate dynamics
    xf = xf';
    
    Fq = @(x) x'*qValues - qValues;     % Vector payoff
    fq = @(t,x) x.*(Fq(x) - x'*Fq(x));  % Replicator dynamics
    
    [tq,xq] = ode45(fq, tspan, x0);     % Simulate dynamics
    xq = xq';

    solutions{end+1} = {xf,xq,tf,tq};
end 

% Plotting 
dS = S(2)-S(1);                         % Grid spacing
dA = dS^2;                              % Area element

objGrid = reshape(objValues, size(X));  % Turn function into square matrix
qGrid = reshape(qValues, size(X));

% % Objective and Approximation
% figure;
% 
% subplot(1,2,1)
% surf(X,Y,objGrid,'EdgeColor','none');
% xlabel('s_1');
% ylabel('s_2');
% zlabel('f(s_1,s_2)');
% title('Styblinski-Tang objective');
% grid on;
% view(45,30);
% 
% subplot(1,2,2)
% surf(X,Y,qGrid,'EdgeColor','none');
% xlabel('s_1');
% ylabel('s_2');
% zlabel('q(s_1,s_2)');
% title('Quadratic approximation');
% grid on;
% view(45,30);

% Mean Trajectories on Contour Plots
figure;

for k = 1:p
    xf = solutions{k}{1};
    xq = solutions{k}{2};

    xfplot = xf ./ sum(xf,1);
    xqplot = xq ./ sum(xq,1);

    xfdensity = xfplot / dA;
    xqdensity = xqplot / dA;

    M1 = strategies' * xfplot;
    M2 = strategies' * xqplot;

    subplot(1,2,k);

    [~,h1] = contour(X,Y,objGrid,30);
    hold on;

    h2 = plot(M1(1,:),M1(2,:),'LineWidth',1.5);
    h3 = plot(M2(1,:),M2(2,:),'LineWidth',1.5);
    h4 = plot(sStar(1),sStar(2),'g.','MarkerSize',25);

    % Initial distribution plot
    r = 3*distributions{k}{2};
    cx = startingPoints{k}(1);
    cy = startingPoints{k}(2);

    theta = linspace(0,2*pi,100);
    x = r*cos(theta) + cx;
    y = r*sin(theta) + cy;

    h5 = plot(x,y,'LineWidth',2);
    hold off;

    xlabel('m_1(t)');
    ylabel('m_2(t)');
    grid on;
    axis equal;
    xlim([-4, 4]);
    ylim([-4, 4]);
    title(['Inital mean: ', mat2str(startingPoints{k}')]);
end

led1 = 'Inital distribution w/ sigma_0=';
led2 = num2str(sigma0);

sgtitle('Mean trajectories on objective contours');
legend([h1, h2, h3, h4, h5], ...
    {'Objective contours', 'm(t)', 'm_q(t)', 'Global minimizer', ...
    strcat(led1,led2)}, ...
    'Position', [0.35 0.01 0.3 0.05], ...
    'Orientation', 'horizontal');

% Mean Convergence Rate Comparison
figure;

for k = 1:p
    xf = solutions{k}{1};
    xq = solutions{k}{2};

    tf = solutions{k}{3};
    tq = solutions{k}{4};

    xfplot = xf ./ sum(xf,1);
    xqplot = xq ./ sum(xq,1);

    mf = strategies' * xfplot;
    mq = strategies' * xqplot;

    nf = vecnorm(mf - sStar,2,1);
    nq = vecnorm(mq - sStar,2,1);

    subplot(1,2,k);
    plot(tf,nf,'LineWidth',1.5);
    hold on;
    plot(tq,nq,'LineWidth',1.5);
    hold off;
    
    xlabel('Time t');
    ylabel('||m(t) - m^*||_2');
    ylim([0,8]);
    title(['Starting point ', mat2str(startingPoints{k}')]);
    legend('m(t)','m_q(t)');
    grid on;
end

sgtitle('Mean convergence to global minimizer');

% Evolution of Max EVal from Covariance Matrix
figure;

for k = 1:p
    xf = solutions{k}{1};
    xq = solutions{k}{2};

    tf = solutions{k}{3};
    tq = solutions{k}{4};

    xfplot = xf ./ sum(xf,1);
    xqplot = xq ./ sum(xq,1);

    mf = strategies' * xfplot;
    mq = strategies' * xqplot;

    Cf = zeros(2,2,length(tf));
    Cq = zeros(2,2,length(tq));

    for j = 1:length(tf)
        xj = xfplot(:,j);
        mj = mf(:,j);

        Cf(:,:,j) = strategies' * (strategies .* xj) - mj*mj';
    end

    for j = 1:length(tq)
        xj = xqplot(:,j);
        mj = mq(:,j);

        Cq(:,:,j) = strategies' * (strategies .* xj) - mj*mj';
    end

    maxEvaluesF = zeros(1, length(tf));
    maxEvaluesQ = zeros(1, length(tq));

    for j = 1:length(tf)
        maxEvaluesF(j) = max(eig(Cf(:,:,j)));
    end

    for j = 1:length(tq)
        maxEvaluesQ(j) = max(eig(Cq(:,:,j)));
    end

    subplot(1,2,k);
    plot(tf,maxEvaluesF,'LineWidth',1.5);
    hold on;
    plot(tq,maxEvaluesQ,'LineWidth',1.5);
    hold off;

    xlabel('Time t');
    ylabel('Max eigenvalue');
    ylim([0,8]);
    title(['Starting point ', mat2str(startingPoints{k}')]);
    legend('lambda_{max}(C(t))','lambda_{max}(C_q(t))');
    grid on;
end

sgtitle('Covariance max eigenvalue evolution');