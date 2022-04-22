%testing_function
% clc
% clear all
%%%%%%%%%%%%%%%%%%%%%%%%
global data_log10_Ntotal
global No
global Kg
global Kd
global  Nmax
global  jmax
% global rmin_upper
%%%%%%%%%%%%%%%%%%%%%%%%

Nmax = 10^8;
dummy=linspace(1,240,240);
dt=5./60;
tspan = [0, dummy*dt]';
jmax = 1000;


% FOR OLD DATA
% Kg = 2.1342;
% Kd = 0.36178;

%% Set C= 16
% No = 10^6.261588663;

% data_log10_Nlive = 5.289573864;
% 
% data_log10_Ntotal = [6.261588663
% 6.305045931
% 6.421717075
% 6.523192531
% 6.533983601
% 6.617052583
% 6.600676305
% 6.675912163
% 6.722765342
% 6.714793965
% 6.731647781
% 6.787220231
% 6.777620467
% 6.74202225
% 6.706864172
% 6.771637062];
% 
% weight=[ones(length(data_log10_Ntotal),1); (10^-10)];

%% Set C= 64

% No = 10^6.250824497;
% 
% data_log10_Nlive =5.298992442 ;
% 
% data_log10_Ntotal = [6.250824497
% 6.320576573
% 6.34597143
% 6.400721885
% 6.504925528
% 6.519872394
% 6.607578468
% 6.604616477
% 6.655043133
% 6.616101497
% 6.653812831
% 6.683548624
% 6.71194043
% 6.698562406
% 6.68193731
% 6.67635848
% 6.682326411];
% 
% weight=[ones(length(data_log10_Ntotal),1); (1.01285407*10^-10)];

%% Set C= 256

% data_log10_Nlive = 2.8613 ;
% 
% No = 10^6.345794841;
% data_log10_Ntotal = [6.345794841
% 6.391027889
% 6.401518563
% 6.417468383
% 6.566075768
% 6.561976676
% 6.607321587
% 6.616114324
% 6.616631544
% 6.617148764
% 6.617665984
% 6.618183204
% 6.618700424
% 6.619217644
% 6.619734863];
% 
% weight=[ones(length(data_log10_Ntotal),1); (10^-10 )];

%% FOR NEW DATA

Kg = 2.3115;
Kd = 0.013935861;
% 
 %% Set = 4/2
No =  1.1824e+06;

data_log10_Ntotal =[6.072776203
6.144607622
6.207366165
6.260929191
6.29557831
6.317540517
6.328540909
6.333548813
];

data_log10_Nlive= 1.6;

weight=[ones(length(data_log10_Ntotal),1); 10^-10];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% C = 16 ploting values
% A = linspace(0,1, 40);
% rmin = linspace(0,3.4, 50);
% lambda =3.64170000000000;

% A =[0.100000000000000]; %0.1
% lambda = [3.64170000000000]; %3
% rmin = [2.06567535910927]; %2
%% C = 64 ploting values
% A = linspace(0,3, 40);
% rmin = linspace(0,3, 50);
% lambda = 2.4883;

% A = [0.370961956661256];
% rmin = [2.11583859099827];
% lambda = [2.48833926710723];
%% C = 256 ploting values
% A = linspace(0,3, 40);
% rmin = linspace(0,6, 50);
% lambda = [3.02294945388794];

% A = [0.913634646599723];
% rmin = [2.36377847712380];
%  lambda = [3.02294945388794];
%% C = 4/2 ploting values 
% A = linspace(0,0.42, 40);
% lambda = [10.5638662532146];
% rmin = linspace(0, 5, 50);


A = [0.140851550042861];
lambda = [10.5638662532146];
rmin = [2.30447071318006];

% resutls=Objective_function_min(Kg,Kd,A)
%% WE NEED THIS. IT IS CODED OUT JUST FOR NOW 
for j = 1:length(lambda)
    for i = 1:length(A) 
        for w = 1:length(rmin)
            [obj_fun_value, N_live, N_total] = Objective_function_min_2(rmin(w), lambda(j), A(i),...
                data_log10_Ntotal, data_log10_Nlive, tspan, weight, Nmax, No, Kg, Kd, jmax);
            Objective_function(w, i) = ((obj_fun_value)); %(log10(obj_fun_value))
%          figure( 1)
%         hold on
%         plot(rmin(w),log10(obj_fun_value), 'ob')
%         hold on

% xlabe( 'rmin')
% ylabel( 'Objective function')

        end
    end
    grid on
    hold off
    figure( j+1)
%     surf(A, rmin, Objective_function)
%     mesh(A, rmin, Objective_function)
%     contour3(A, rmin, Objective_function,350) %weight has been incorporated

    zlabel( 'Objective function')
    xlabel( 'A')
    ylabel( 'rmin')
    title([ 'lambda = ' num2str( lambda(j))])

end
%%  Testing
%             Objective_function_log_scale = (log10(obj_fun_value))
            Objective_function = ((obj_fun_value))
             

%% 
clc 
clearvars -except data_log10_Ntotal data_log10_Nlive tspan No jmax Kg Kd Nmax Objective_function 
% clf reset

raw_data = [ data_log10_Ntotal; data_log10_Nlive];
fitting_time = [ tspan( 1: length( data_log10_Ntotal)); tspan( length( tspan))];


%% for C = 16 initial guesses

% A = 0.1; %0.1
% lambda = 3.6417; %3
% rmin = 2.06567535910927; %2

%% for C = 64 initial guesses
% A = 1; %0.38
% lambda = 3.6417;
% rmin = 2.06567535910927; %2.11

%% for C = 256 initial guesses
% A = 0.9;
% lambda =  3;
% rmin =2.34;


%% for C= 4/2 inital guesses
A = 0.1; %0.1
lambda = 7.5; %7.5
rmin = 2; %1.5

B0 = [ rmin A lambda]; %this is initial guess for parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C = 16 WEIGHTS
% weight=[ones(length(data_log10_Ntotal),1);10^-10];

% C = 64 WEIGHTS
% weight=[ones(length(data_log10_Ntotal),1); 1.01285407*10^-10];

% C = 256 WEIGHTS
% weight=[ones(length(data_log10_Ntotal),1);10^-10];% 3.9999999999*1000*10^-10

% C = 4/2 WEIGHTS 
weight=[ones(length(data_log10_Ntotal),1); 10^-10];

%% 0.0000000004, 10^-14
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts=statset('Display','iter','TolX',10^-300,'TolFun',10^-40000);
% 
% opts.MaxIter = 20000;

% opts = statset('nlinfit');
% opts.RobustWgtFun = 'bisquare'

% opts=statset('glmfit');
opts.MaxIter = 20000;

%% C = 16
% opts.DerivStep = 10^-14.07; %10^-14.07;
%% C = 64
% opts.DerivStep = 10^-14.07;
%% C = 256
% opts.DerivStep = 10^-13.89;

%% C = 4/2
opts.DerivStep = 10^-13.89;


%%%%%%%%%%%%%%%%%%%
% rmin_upper=rmin_256 %% to be used in the CAZ case no in the CAS+CEF CASE
%%%%%%%%%%%%%%%%%%%

md1 = fitnlm( fitting_time, raw_data, @model_output, B0,'Options',opts,'Weight',weight)
for i=1:length(B0)
b(i)= md1.Coefficients.Estimate(i,1);
end
b = b';
rmin = b(1);
A = b(2);
lambda = b(3);

CM = md1.CoefficientCovariance
Nikolaou = inv(diag(sqrt(diag(CM))))
Condition_Number = cond(CM)
CM2 = Nikolaou*CM*Nikolaou %correlation matrxi 

N_live(1,1) = No;
N_total(1,1) = No; 
    for k = 1: length(tspan)-1  %reaching 20 hours 
    N_live_temp(1,1) = N_live(k,1);
    N_total_temp(1,1) = N_total(k,1);
    dt_internal = (tspan(k+1,1)-tspan(k,1))/jmax;  
    for j = 1 : jmax-1 %intermediate number of steps
        N_live_temp(j+1,1) = N_live_temp(j,1) + dt_internal*...
            (-Kg*N_live_temp(j,1)/Nmax+Kg-rmin-lambda*A*exp(-A*(tspan(k,1)...  
            +j*dt_internal)))*N_live_temp(j,1);
        N_total_temp(j+1,1) = N_total_temp(j,1) ...
            + dt_internal*(Kg*(1- N_live_temp(j,1)/Nmax)+ Kd) * N_live_temp(j,1) ;
    end
    N_live(k+1,1) = N_live_temp(jmax,1);
    N_total(k+1,1) = N_total_temp(jmax,1);
    end


 %% showing point in previous figure   
 hold on 
 plot3(A, rmin,Objective_function, 'or') 
 
 %%
 
figure(1000)
plot(fitting_time,raw_data,'ob')
hold on 
plot(tspan, log10(N_live),'--r')
hold on 
plot(tspan, log10(N_total),'--b')

grid on 
legend







%%
% function model_data = model_output( x, tspan)
% % rmin = real, scalar
% % lambda = real, scalar
% % A = real, scalar
% % tspan = column vector,  size(tspan,1) x 1
% % Nmax = real, scalar
% % No = real, scalar
% % Kg = real, scalar
% % Kd = real, scalar
% % N_live = column vector 
% % N_total = column vector
% 
% 
% %%%%%%%%%%%%%%%%%%
% global data_log10_Ntotal
% %%%%%%%%%%%%%%%%%%
% 
% 
% rmin = x(1);
% A = x(2);
% lambda = x(3);
% % A = 0.184026346943656;
% % lambda = 3.72453496058727;
% % rmin = 2.06567535910927;
% Kg = 2.1342;
% Kd = 0.36178;
% No = 1970353.76557831;
% Nmax = 10^8;
% 
% 
% 
% n_time_points = size(tspan,1);
% N_live(1,1) = No;
% N_total(1,1) = No;
% 
% for k = 1: length(tspan)-1  %reaching 20 hours 
%     N_live(k+1,1) = N_live(k,1) +...
%         (tspan(k+1,1)-tspan(k,1))*...
%         (-Kg*N_live(k,1)/Nmax+Kg-rmin-lambda*A*exp(-A*tspan(k+1,1)))*N_live(k,1);
%     N_total(k+1,1) = N_total (k,1) +...
%         (tspan(k+1,1)-tspan(k,1))*(Kg*(1- N_live(k,1)/Nmax)+ Kd) * N_live(k,1); 
% end
%     
% 
% model_data = [log10( N_total( 1: length( data_log10_Ntotal)));...
%             log10( N_live( length( tspan)-1))];
%         return
%         
% end
