%testing_function
% clc
clear all
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
%% FOR OLD DATA 
% Kg = 2.13424372088011; 
% Kd = 0.361780189128987;
%% FOR OLD DATA 
% C = 1;
% No = 10^(6.38161147-1);
% data_log10_Ntotal =      [ 6.38161147
% 6.457527011
% 6.49609176
% 6.537010194
% 6.603399988
% 6.699612468
% 6.807576605
% 6.850636247
% 6.941466213
% 7.013970252
% 7.129537871
% 7.201470354
% 7.284694378
% 7.401441904
% 7.47540533
% 7.524993841
% 7.57967474
% 7.677654691
% 7.765540015
% 7.810567213
% 7.907370913
% 7.920022667
% 7.920167615
% 7.995542998]-1;


% C = 4;
% No = 10^(6.358531398-1);
% data_log10_Ntotal =  [  6.358531398
% 6.396170797
% 6.478615374
% 6.508810099
% 6.541390788
% 6.582453078
% 6.736684299
% 6.756548868
% 6.861477956
% 6.888909968
% 6.987556473
% 6.978694774
% 6.974131294
% 6.98246977
% 7.075762471
% 7.103222313
% 7.152825966
% 7.135221765
% 7.17412795
% 7.175263153
% 7.202566965
% 7.281445679
% 7.246088341
% 7.309334256
% 7.280812272
% 7.367296806
% 7.36082304
% 7.391350617
% 7.395564838
% 7.389954057
% 7.424262265
% 7.471257892
% 7.472587411
% 7.464434256
% 7.52570673
% 7.573922462
% 7.483730387
% 7.588908452
% 7.614208644
% 7.647236827
% 7.642358685
% 7.623354088
% 7.668959788
% 7.679861813
% 7.643924133
% 7.653760984
% 7.729323923
% 7.699081962
% 7.699477153] - 1;
%%
% C = 16
% No = 10^(6.261588663-1);
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
% 6.771637062]-1;

%% 
% C = 64;
% No = 10^(6.250824497-1);
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
% 6.682326411]-1;
%%
% C = 256;
% No = 10^(6.345794841 - 1);
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
% 6.619734863] - 1;


%% FOR NEW DATA 
Kg = 2.3115;
Kd = 0.013935861;

%%
% C = 1/ 0.5;
% No =  10^(6.121403527-1);
% data_log10_Ntotal = [6.121403527
% 6.205954068
% 6.29478689
% 6.362522457
% 6.434103599
% 6.504552733
% 6.566887782
% 6.638821254
% 6.712294461
% 6.771204187
% 6.816539566
% 6.853203171]-1;

%% 
% C = 4/2;
% No = 10^(log10( 1.1824e+06)-1);
% 
% data_log10_Ntotal =[6.072776203
% 6.144607622
% 6.207366165
% 6.260929191
% 6.29557831
% 6.317540517
% 6.328540909
% 6.333548813
% ]-1;

%% 
% C = 16/8;
% No = 10^(6.154033559-1);
% data_log10_Ntotal = [6.154033559
% 6.220649855
% 6.238016339
% 6.252328127
% 6.254797608
% 6.259520825
% 6.262782278]-1;

%%
% C = 64/32;
% No = 10^(6.02196923-1);
% data_log10_Ntotal = [6.02196923
% 6.053587127
% 6.061157144
% 6.066183166
% 6.065290588
% 6.059365437
% ]-1;

%% 
C = 256/128;
No = 10^(6.162064308-1);
data_log10_Ntotal = [6.162064308
6.151509816
6.156734334
6.156070426
6.162674072
6.165416419
6.167679811
6.169927882
]-1;


weight=ones(length(data_log10_Ntotal),1); 

dummy=linspace(1,240,240);
dt=5./60;
tspan = [0, dummy*dt]';
jmax = 1000;



%% FOR OLD DATA


%  C = 1; % ploting values
% A = linspace(0,1, 40);
% rmin = linspace(-1,1, 50);
%  lambda =  0 ;

% A = [0.829155131096260];
% rmin = [0.0394067784761810];
% lambda =  [-0.310484218094826];

% A = 0.6; 
% rmin = 0.238846347709794;
% lambda =  0 
% 
% A = 0; %previous point of optimum
% rmin = 0;
% lambda =  0; 


%% C = 4 ploting values
% A = linspace(0,0.35, 40);
% rmin = linspace(0,2, 50);
% lambda = 7.9379;

% A = [0.0682343032533271];
% rmin = [1.11772505509486];
% lambda = [7.93790212067068];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A = linspace(0,0.35, 40);
% rmin = linspace(0,2, 50);
% lambda = 3.16159852726082;
% 
% A = 0.111620669007826;
% rmin = 1.41297194830770;
% lambda = 3.16159852726082;
%% C = 16 ploting values
% A = linspace(0,0.6, 40);
% rmin = linspace(0,3, 50);
% lambda = [5.21747654635654];


% A = [0.125020563952843];
% rmin = [1.78601928594799];
% lambda = [5.21747654635654];%3.9875;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% rmin = 1.98831497801474;
% A = 0.169219161431292;
% lambda = 3.20619301693884;
%% C = 64 ploting values
% A = linspace(0,0.35, 40);
% rmin = linspace(0,3.5, 50);
% lambda = 10.4207204765757;

% A = [0.0967171370947542];
% rmin = [1.88273881810925];
% lambda = [11.4503170248694];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A = 0.104391664571670;
% rmin = 1.94671407188453;
% lambda = 10.4207204765757;

%% C = 256 ploting values
% A = linspace(0,1, 40);
% rmin = linspace(0,6, 50);
% lambda =[6.82963074721327];

% A = [0.426265798975286];
% rmin = [1.96188199221106];
% lambda =[6.82963074721327];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A = linspace(0,1, 40);
% rmin = linspace(0,6, 50);
% lambda =5.54084471815461;

% A = 0.441340241716014;
% rmin = 2.46518408010348;
% lambda =5.54084471815461;
%% FOR NEW DATA 
%% C = 1/0.5 ploting values 
% A = linspace(0,0.4, 40);
% rmin = linspace(0,0.7, 50);
% lambda = [1.86827676426305];
% 
% A = [0.0916663695530375];
% rmin = [0.447535814068696];
% lambda = [1.86827676426305];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A = linspace(0,0.4, 40);
% rmin = linspace(0,0.7, 50);
% lambda = 6.55414785476369;

% A = 0.0983122178214555;
% rmin = 0.0966820169011865;
% lambda = 6.55414785476369;

%% C = 4/2 ploting values 
% A = linspace(0,3.5, 40);
% rmin = linspace(0,4.5, 50);
% lambda = [1.91376242899743];

% A = [0.350406238036801];
% rmin = [3.11771794932434];
% lambda = [1.91376242899743];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A = linspace(0,3.5, 40);
% rmin = linspace(0,4.5, 50);
% lambda = 1.85185879025294;

% A =0.116016967457860;
% rmin = 3.61274034035565;
% lambda = 1.85185879025294;

%% C = 16/8 ploting values 
% A = linspace(0,1.5, 40);
% rmin = linspace(0,14, 50);
% lambda = [10.5105347156774];

% A = [0.157658020735161];
% rmin = [8.65735748273705];
% lambda = [10.5105347156774];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A = linspace(0,1.5, 40);
% rmin = linspace(0,14, 50);
% lambda =9.42356666831896;

% A = 0.141353500024784;
% rmin =9.06094713889759;
% lambda =9.42356666831896;
%% C = 64/32 ploting values 
% A = linspace(0,2, 40);
% rmin = linspace(20,26, 50);
% lambda = [3.15485015770326] ;

% A = [0.866312479804285];
% rmin = [23.0424801872451];
% lambda =  [3.15485015770326];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A = linspace(0,2, 40);
% rmin = linspace(20,26, 50);
% lambda = 2.75567693898233 ;

% A = 0.756700699942974;
% rmin = 23.8949300513201;
% lambda = 2.75567693898233 ;

%% C = 256/128 ploting values 
% A = linspace(110,130, 40);
% rmin = linspace(90,110, 10);
% lambda =  3.64169999999999 ;

% A = 120;
% rmin = 100;
% lambda =  3.64169999999999 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A = linspace(390,400, 40);
% rmin = linspace(80,150, 10);
% lambda = 11.93383668;

% A = 3228.10355459487;
% rmin = 3260.70515119458;
% lambda =  24.3079764827963 ;
% 
rmin = 100;
A = 393.2395315;
lambda = 11.93383668;

% resutls=Objective_function_min(Kg,Kd,A)
%% WE NEED THIS. IT IS CODED OUT JUST FOR NOW 
for j = 1:length(lambda)
    for i = 1:length(A) 
        for w = 1:length(rmin)
            [obj_fun_value, N_live, N_total] = Objective_function_min_2_NTOTAL(rmin(w), lambda(j), A(i),...
                data_log10_Ntotal, tspan, weight, Nmax, No, Kg, Kd, jmax);
            Objective_function(w, i) = ((obj_fun_value)); %log10(obj_fun_value)
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
%      contour3(A, rmin, Objective_function,350) %weight has been incorporated

    zlabel( 'Objective function')
    xlabel( 'A')
    ylabel( 'rmin')
    title([ 'lambda = ' num2str( lambda(j))])

end
%% showing point in previous figure   
 
hold on 
  plot3(A, rmin,Objective_function, 'or') 
%%  Testing
%             Objective_function_log_scale = (log10(obj_fun_value))
%               Objective_function = ((obj_fun_value))
            

% clc 
clearvars -except data_log10_Ntotal  tspan No jmax Kg Kd Nmax Objective_function TK_1_MODEL time_TK_1 C
% clf reset
raw_data = data_log10_Ntotal;
fitting_time =  tspan( 1: length( data_log10_Ntotal));



%% For OLD DATA

%% C = 1 initial guesses
% rmin =0.3; %0.3
% A = 0.6; %0.6
% lambda =3.5; %3.5

% A = 0; %previous point of optimum
% rmin = 1;
% lambda =  0; 

% rmin_upper = 0;
% A = 0.0001; %previous point of optimum
% rmin = 1;
% lambda =  3.21E-01;
%% C = 4 initial guesses
% A = 1;
% lambda = 3;
% rmin = 0.1;

%% C = 16 initial guesses
% A = 0.2;
% lambda = 3.6417;
% rmin =1;
%% C = 64 initial guesses
% A = 0.1; %0.0967;
% lambda = 10; %11.45;
% rmin = 1; %1.8827; 

%% C = 256 initial guesses
% A = 0.4;
% lambda = 4;
% rmin = 1;

%% For NEW DATA
%% for C= 1/0.5 inital guesses
% rmin = 1;
% A = 0.15;
% lambda = 10;

%% for C= 4/2 inital guesses
% A = 1;
% lambda = 1;
% rmin = 6;

%% for C= 16/8 inital guesses
% rmin = 10;
% A = 0.15;
% lambda = 10;

%% for C= 64/32 inital guesses
% A = 1;
% lambda = 3.6417;
% rmin =17;% 20;

%% for C= 256/128 inital guesses
A = 120;
lambda = 3.6417;
rmin = 100;


B0 = [ rmin A lambda]; %   this is initial guess for parameters1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

weight=ones(length(data_log10_Ntotal),1); 
%% 0.0000000004, 10^-14
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts=statset('Display','final','TolX',10^-300,'TolFun',10^-40000);

% opts=statset('glmfit');
% opts.MaxIter = 20000;


%% For OLD DATA

%% C = 1
% opts.DerivStep = 10^-14;%13.54; %10^-12;

%% C = 4
% opts.DerivStep = 10^-14;

%% C = 16
% opts.DerivStep = 10^-13.89;

%% C = 64
% opts.DerivStep = 10^-14;

%% C = 256
% opts.DerivStep = 10^-14;

%% For NEW DATA
%% C = 1/0.5
% opts.DerivStep = 10^-13.89;

%% C = 4/2
% opts.DerivStep = 10^-13.89;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% opts.DerivStep = 10^-13;


%% C = 16/8
% opts.DerivStep = 10^-13.89;
%%%%%%%%%%%
% opts.DerivStep = 10^-13;
%% C = 64/32
% opts.DerivStep = 10^-13;

%% C = 256/128
opts.DerivStep = 10^-13;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%
%% rmin_upper=rmin_256 %% to be used in the CAZ case no in the CAS+CEF CASE
%%%%%%%%%%%%%%%%%%%

md1 = fitnlm( fitting_time, raw_data, @model_output_rmin_upper_NTOTAL, B0,'Options',opts,'Weight',weight)% model_output_rmin_upper_NTOTAL
for i=1:length(B0)
    b(i)= md1.Coefficients.Estimate(i,1);
end
b = b';
rmin = b(1);
A = b(2);
lambda = b(3);

CM = md1.CoefficientCovariance;
Nikolaou = inv(diag(sqrt(diag(CM))));
Condition_Number = cond(CM);
CM2 = Nikolaou*CM*Nikolaou; %correlation matrxi 

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save results in a file
temp = fileparts(matlab.desktop.editor.getActiveFilename);
filename = strcat(temp,'\results_C_', num2str(C),'.txt');
save(filename, 'C', 'rmin', 'lambda', 'A', '-ascii','-tabs')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 
%%
figure(1000)
plot(fitting_time,raw_data,'ob')
hold on 
plot(tspan, log10(N_live),'--r')
hold on 
plot(tspan, log10(N_total),'--b')

grid on 
legend





