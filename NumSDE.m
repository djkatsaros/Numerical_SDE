% Descretized Brownian Motion

%% Single Motion
close all; clc;

randn('state',100)
T = 1; N = 500; dt = T/N; % \delta t is the incr. s < t in  defn

dW = sqrt(dt)*randn(1,N); % first approx outside loop
W = cumsum(dW);         % cumulative sum

plot([0:dt:T],[0,W],'r-') % plot W v.s. t using incr dt
xlabel('t', 'FontSize', 16)
ylabel('W(t)','FontSize', 16, 'Rotation',0)

%% averaged over 1000 disc'd Brownian Motions
close all; clc;

randn('state',100)
T = 1; N = 500; dt = T/N; t = [dt:dt:1]; % \delta t is the incr. s < t in  defn

M = 1000;                       % M paths simultly
dW = sqrt(dt) * randn(M,N);     %increments
W = cumsum(dW,2);

U = exp(repmat(t,[M 1]) + 0.5*W);
Umean = mean(U);
 
plot([0,t],[1,Umean],'b-'), hold on                 % plot mean over M paths
plot([0,t],[ones(5,1),U(1:5,:)], 'r--'), hold off  % plot 5 individual paths
xlabel('t', 'FontSize', 16) 
ylabel('U(t)','FontSize', 16, 'Rotation',0, 'HorizontalAlignment','right')
legend('mean of 1000 paths', '5 individual paths',2)

averr = norm((Umean - exp(9*t/8)),'inf') % error 


%%
% Approximate stochastic integrals
%
% Ito and Stratonovich integrals of WdW

randn('state', 100)                     % set state of the randn
T =1; N = 500; dt = T/N;                % discretization 

dW = sqrt(dt) * randn(1,N);
W = cumsum(dW);                         

ito = sum([0,W(1:end-1)].*dW)
strat = sum((0.5*([0,W(1:end-1)] + W) + 0.5*sqrt(dt)*randn(1,N)).*dW)

itoerr = abs(ito - 0.5*(W(end)^2 - T))
straterr = abs(strat - 0.5*W(end)^2)


%% Euler-Maru yama method on linear SDE
%
% SDE is dX = lambda*X dt + mu * X dW X(0) = Xzero,
%   where lambda = 2, mu = 1 and Xzero = 1    
%
% Discretized Brownian path over [0,1] has dt = 2^(- 8)
% Euler-Maruyama uses timestep R*dt
 
randn('state',100) 
lambda = 2; mu = 1; Xzero = 1;
T = 1; N = 2^8; dt = 1/N;
dW = sqrt(dt)*randn(1,N);
W = cumsum(dW);

Xtrue = Xzero*exp((lambda - 0.5*mu^2)*([dt:dt:T])+mu*W);
plot([0:dt:T],[Xzero,Xtrue],'m-'), hold on

R =4; Dt = R*dt; L = N/R;      % L EM steps of sie Dt = R*dt
Xem = zeros(1,L);
Xtemp = Xzero;
for j = 1:L
    Winc = sum(dW(R*(j-1) + 1:R*j));
    Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp*Winc;
    Xem(j) = Xtemp;
end

plot([0:Dt:T],[Xzero,Xem],'r--*'), hold off
xlabel('t','FontSize',12)
ylabel('X','FontSize',16,'Rotation', 0, 'HorizontalAlignment','right')

EMerr = abs(Xem(end)-Xtrue(end))

%% Euler-Maru yama method on linear SDE
%
% SDE is dX = lambda*X dt + mu * X dW X(0) = Xzero,
%   where lambda = 2, mu = 1 and Xzero = 1    
%
% Discretized Brownian path over [0,1] has dt = 2^(-9)
% Euler-Maruyama uses 5 different timesteps: 16dt, 8dt, 4dt, 2dt, dt.
% Examine strong convergence at T=1: E|X_L - X(T)| . 
 
randn('state',100) 
lambda = 2; mu = 1; Xzero = 1;
T = 1; N = 2^9; dt = T/N;
M = 1000;                               % number of paths sampled

Xerr = zeros(M,5);                      % preallocate array
for s = 1:M,                            % sample over discrete Brownian paths
    dW = sqrt(dt)*randn(1,N);           % Brownian increments
    W = cumsum(dW);
    Xtrue = Xzero*exp((lambda - 0.5*mu^2)+mu*W(end));
    for p = 1:5
        R = 2^(p-1); Dt = R*dt; L = N/R; % L Euler steps of size Dt = R*dt
        Xtemp = Xzero;
        for j = 1:L
            Winc = sum(dW(R*(j-1) + 1:R*j));
            Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp*Winc;
        end
        Xerr(s,p) = abs(Xtemp - Xtrue);
    end
end
        
Dtvals = dt*(2.^([0:4]));
subplot(221)                            % Top LH picture
loglog(Dtvals,mean(Xerr),'b*-'), hold on
loglog(Dtvals,(Dtvals.^(.5)),'r--'), hold off % reference slope of 1/2
axis([1e-3 1e-1 1e-4 1])
xlabel('\Delta t'), ylabel('Sample average of | X(T) - X_L |')
title('emstrong.m','FontSize',10)

%%%% Least squares fit of error = C * Dt^q %%%%
A = [ones(5,1), log(Dtvals)']; rhs = log(mean(Xerr)');
sol = A\rhs; q = sol(2)
resid = norm(A*sol - rhs)

%EMWEAK Test weak convergence of Euler-Maruyama
%
% Solves dX = lambda*X dt + mu*X dW, X(0) = Xzero,
% where lambda = 2, mu = 1 and Xzer0 = 1.
%
%E-Muses5differenttimesteps:2^(p-10),p=1,2,3,4,5.
% Examine weak convergence at T=1: | E (X_L) - E (X(T)) |.
%
% Different paths are used for each E-M timestep.
% Code is vectorized over paths.
%
% Uncommenting the line indicated below gives the weak E-M method.
randn('state',100);
lambda = 2; mu = 0.1; Xzero = 1; T = 1;  % problem parameters
M = 50000;                               % number of paths sampled
Xem = zeros(5,1);                        % preallocate arrays 
for p=1:5                                 %takevariousEulertimesteps
       Dt = 2^(p-10); L = T/Dt;          % L Euler steps of size Dt
       Xtemp = Xzero*ones(M,1);
       for j = 1:L
           Winc = sqrt(Dt)*randn(M,1);
           % Winc = sqrt(Dt)*sign(randn(M,1));     %% use for weak E-M %%
           Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp.*Winc;
       end
       Xem(p) = mean(Xtemp);
end
Xerr = abs(Xem - exp(lambda));
Dtvals = 2.^([1:5]-10);
subplot(222)
loglog(Dtvals,Xerr,'b*-'), hold on
loglog(Dtvals,Dtvals,'r--'), hold off
axis([1e-3 1e-1 1e-4 1])
xlabel('\Delta t'), ylabel('| E(X(T)) - Sample average of X_L |')
title('emweak.m','FontSize',10)
%%%% Least squares fit of error = C * dt^q %%%%
A = [ones(p,1), log(Dtvals)']; rhs = log(Xerr);
sol = A\rhs; q = sol(2)
resid = norm(A*sol - rhs)


randn('state',100);
lambda = 2; mu = 0.1; Xzero = 1; T = 1;  % problem parameters
M = 50000;                               % number of paths sampled
Xem = zeros(5,1);                        % preallocate arrays 
for p=1:5                                 %takevariousEulertimesteps
       Dt = 2^(p-10); L = T/Dt;          % L Euler steps of size Dt
       Xtemp = Xzero*ones(M,1);
       for j = 1:L
           Winc = sqrt(Dt)*randn(M,1);
           Winc = sqrt(Dt)*sign(randn(M,1));     %% use for weak E-M %%
           Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp.*Winc;
       end
       Xem(p) = mean(Xtemp);
end
Xerr = abs(Xem - exp(lambda));
Dtvals = 2.^([1:5]-10);
subplot(223)
loglog(Dtvals,Xerr,'b*-'), hold on
loglog(Dtvals,Dtvals,'r--'), hold off
axis([1e-3 1e-1 1e-4 1])
xlabel('\Delta t'), ylabel('| E(X(T)) - Sample average of X_L |')
title('emweak.m','FontSize',10)
%%%% Least squares fit of error = C * dt^q %%%%
A = [ones(p,1), log(Dtvals)']; rhs = log(Xerr);
sol = A\rhs; q = sol(2)
resid = norm(A*sol - rhs)

%% MILSTRONG  Test strong convergence of Milstein: vectorized
%
% Solves   dX = r*X*(K-X) dt + beta*X dW,  X(0) = Xzero,
%        where r = 2, K= 1, beta = 1 and Xzero = 0.5.
%
% Discretized Brownian path over [0,1] has dt = 2^(-11).
% Milstein uses timesteps 128*dt, 64*dt, 32*dt, 16*dt (also dt for reference).
%
% Examines strong convergence at T=1:  E | X_L - X(T) |.
% Code is vectorized: all paths computed simultaneously.

randn('state',100)
r=2;K = 1; beta = 0.25; Xzero = 0.5;    % problem parameters
T=1; N = 2^(11); dt = T/N;              %
M=500;                                  % number of paths sampled
R = [1; 16; 32; 64; 128];               % Milstein stepsizes are R*dt
dW = sqrt(dt)*randn(M,N);               % Brownian increments
Xmil = zeros(M,5);                      % preallocate array
for p = 1:5
    Dt = R(p)*dt; L = N/R(p);          % L timesteps of size Dt =Rdt
    Xtemp = Xzero*ones(M,1);
    for j = 1:L
        Winc = sum(dW(:,R(p)*(j-1)+1:R(p)*j),2);
        Xtemp = Xtemp + Dt*r*Xtemp.*(K-Xtemp) + beta*Xtemp.*Winc ...
              + 0.5*beta^2*Xtemp.*(Winc.^2 - Dt);
    end
    Xmil(:,p) = Xtemp;  % store Milstein solution at t =1
end
Xref = Xmil(:,1);                                % Reference solution
Xerr = abs(Xmil(:,2:5) - repmat(Xref,1,4));      % Error in each path
mean(Xerr);                                      % Mean pathwise erorrs
Dtvals = dt*R(2:5);                              % Milstein timesteps used
subplot(224)
loglog(Dtvals,mean(Xerr),'b*-'), hold on
loglog(Dtvals,Dtvals,'r--'), hold off            % reference slope of 1
axis([1e-3 1e-1 1e-4 1])
xlabel('\Delta t')
ylabel('Sample average of | X(T) - X_L |')
title('milstrong.m','FontSize',10)

%%%% Least squares fit of error = C * Dt^q %%%%
A = [ones(4,1), log(Dtvals)]; rhs = log(mean(Xerr)');
sol = A\rhs; q = sol(2)
resid = norm(A*sol - rhs)

%%

%STAB  Mean-square and asymptotic stability test for E-M
%
% SDE is  dX = lambda*X dt + mu*X dW,   X(0) = Xzero,
%      where lambda and mu are constants and Xzero = 1.
%
% Modified
% two equations are 
% dX = (1-y^2)X dt
% dY = - (alpha/epsilon)ydy + \sqrt{(2\lambda)/epsilon) dW
%

randn('state',100)
T = 20; M = 1; Xzero = 0.1;
ltype = {'b-','r--','m-.'};    

lambda = 1.1; mu = sqrt(3); alpha = 1;
%for eps = [1,1e-1,1e-3,1e-4]
    eps  = 0.1;
for k = 1:3
    Dt = 2^(-5-k);
    N = T/Dt;
    Xms = zeros(1,N); Yms = zeros(1,N); Xtemp = Xzero*ones(M,1); Ytemp = Xzero*ones(M,1);
    subplot(3,1,k)
    for j = 1:N
           Winc = sqrt(Dt)*randn(M,1);
           Ytemp = Ytemp + (1-Xtemp.^2).*Ytemp.*Dt;
           Xtemp = Xtemp +  (-alpha/eps)*(Xtemp).*Dt + sqrt(2*lambda/eps).*Winc;
           Xms(j,:) = Xtemp;     
           Yms(j,:) = Ytemp;
    end
    plot([Dt:Dt:T],Xms,ltype{k},'Linewidth',2), hold on
    plot([Dt:Dt:T],Yms,ltype{k},'Linewidth',2), hold on
end
%end
legend('\Delta t = 1','\Delta t = 1/2','\Delta t = 1/4')
%title('Mean-Square: \lambda = -3, \mu = \surd 3','FontSize',16)
ylabel('E[X^2]','FontSize',12), axis([0,T,1e-20,1e+20]), hold off












