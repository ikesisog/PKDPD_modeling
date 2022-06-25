%% Ntotal integration and fitnlm from 2 equations 
%1st file afte TG estimation file 
clearvars %-except Kg N0 Kd
clc
dataset_1 = xlsread('Akis_7_5_max_condition_NK_DATA_MODIFICATION_vol_2.xlsx','Sheet1','AG29:AM253'); %updated data set 
X = dataset_1;

dataset_2 = xlsread('Akis_7_5_max_condition_NK_DATA_MODIFICATION_vol_2.xlsx','Sheet1','AG19:AM253'); %updated data set 
X2 = dataset_2;
% time_TK_1 = X(:,1);

dataset(3,:) = {X(1:24,3)};
time_TK_1(3,:) = {X2(1:24,1)};

dataset(4,:) = {X(1:49,4)};
time_TK_1(4,:) = {X2(1:49,1)};
AKIS = 25;
dataset(5,:) = {X(1:16,5)}; %Choise was based until when we had true values nothing else matters 
time_TK_1(5,:) = {X2(1:16,1)};

dataset(6,:) = {X(1:17,6)}; %Choise was based until when we had true values nothing else matters 
time_TK_1(6,:) = {X2(1:17,1)};

dataset(7,:) = {X(1:15,7)}; %Choise was based until when we had true values nothing else matters BUT CAN BE CHANGED DUE TO EXPECTED LINE PLATEAU
time_TK_1(7,:) = {X2(1:15,1)};


%% those three are the changing paramters and No
rmin = 1;
A = 0.1;
lambda = 0.03;

xt0 = zeros(1,5);
%   INITIAL_GUESSES = [0.0100000000000704,1.77088394420109,2.17531122046707,3.00337598705801,4.50303861226613;1.9 ,0.0100000000302403,0.129287599767539,0.0100000000000000,0.545301851905182;2 ,0.0100000091032490,0.137380554036107,0.168859049392403,0.372713491752471];

xt0(1) = rmin; %changing this to try new initial points as well as the other two paramters 
xt0(2) = A;
xt0(3) = lambda;
%% 'levenberg-marquardt' best case but big SD 
  options_1 = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Diagnostics','on','Display','final-detailed',...% 'FiniteDifferenceType','central', central takes more integrations allowing better fitting (in this case from 83 to 133 function evaluations) yet for this case I didnt notice any difference in the final resutls 
          'MaxFunctionEvaluations',5000,'TolX',10e-30,'TolFun',10e-30,'MaxIterations',5000,'FunValCheck','on')%,'Display','iter','PlotFcn','optimplotx'
%     'levenberg-marquardt''trust-region-reflective'

%% 'trust-region-reflective' best case 
  options_2 = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective','Diagnostics','on','Display','final-detailed',...% 'FiniteDifferenceType','central', central takes more integrations allowing better fitting (in this case from 83 to 133 function evaluations) yet for this case I didnt notice any difference in the final resutls 
         'TolX',10e-30,'MaxFunctionEvaluations',5000,'TolFun',10e-30,'FunValCheck','on','OptimalityTolerance',10e-30)%,'Display','iter','PlotFcn','optimplotx'
%     'levenberg-marquardt''trust-region-reflective'
%      'MaxIterations',5000,

 %% 'levenberg-marquardt' best case but big SD 
  options_3 = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Diagnostics','on','Display','final-detailed',...% 'FiniteDifferenceType','central', central takes more integrations allowing better fitting (in this case from 83 to 133 function evaluations) yet for this case I didnt notice any difference in the final resutls 
          'MaxFunctionEvaluations',2000,'TolX',10e-30,'TolFun',10e-30,'MaxIterations',2000,'FunValCheck','on')%,'Display','iter','PlotFcn','optimplotx'
%     'levenberg-marquardt''trust-region-reflective'

%% 'levenberg-marquardt' best case but big SD 
  options_4 = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Diagnostics','on','Display','final-detailed')%,...% 'FiniteDifferenceType','central', central takes more integrations allowing better fitting (in this case from 83 to 133 function evaluations) yet for this case I didnt notice any difference in the final resutls 
        %  'MaxFunctionEvaluations',2000,'TolX',10e-30,'TolFun',10e-30,'MaxIterations',2000,'FunValCheck','on')%,'Display','iter','PlotFcn','optimplotx'
%     'levenberg-marquardt''trust-region-reflective'


% options = [options_1 options_1 options_1 options_1 options_1]; %My opinion is that option 4 must be applied to C = 64 and 256 the fitting may not be as accurate but the final D?kg results are 
 options = [options_2 options_2 options_4 options_1 options_1];
    

xq1 = [6.38161147	6.358531398	6.261588663	6.250824497	6.345794841];%[6.063580125, 6.05374236, 6.198805193,6.263234182,6.306644761]; change based on time 0 
  xq1 = 10.^xq1; %change # 1 from 10^ kept in log scale as in line 78
xq2 = xq1;

% figure
% alpha = 0.05; 
Concentration = [1 4 16 64 256];
S = 'Concentration = ';


   
for i = 1:5%one for each concentration
%      xt0(1) = INITIAL_GUESSES(1,i); %changing this to try new initial points as well as the other two paramters
    % xt0(2) = INITIAL_GUESSES(2,i);
    % xt0(3) = INITIAL_GUESSES(3,i);
    disp(S)
    disp(Concentration(i))
    xt0(4:5)  = [xq1(i) xq2(i) ];
    
   
    lb = [0.01 ,0.01 ,0.01 , xq1(i) xq2(i)]; %change #2
    
%                 lb = [0.000000000001 0.000000000001 0.000000000001 xq1(i) xq2(i)]; % This is the correct boundary to use
    
    %    lb = [1 1 1 xq1(i) xq2(i)]; %lower bound us this when running fitnlm after
    
    %       %Testing Lower Bound
    
    ub = [10 10 10 10^7 10^7]; %upper bound of estiamted paramters
    
    % lb = [];
    % ub = [];
    
    [best,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(@paramfun,xt0,time_TK_1{2+i},dataset{2+i},lb,ub,options(i));
    % lsqcurvefit implicitly computes the sum of squares of the components of fun(x,xdata)-ydata. See Examples.
    NORM(i) = norm(residual)';
    
    best = best(1:3)';
    
    Final_rmin(i) = best(1);
    Final_A(i) = best(2);
    Final_lambda(i) = best(3);
    
    %% Error estimation.. SD really big..
    ci = nlparci(best,residual,'jacobian',jacobian(:,1:3),'alpha',0.32)%returns 100(1-alpha)% confidence intervals.
    SD(1:3,i) = (ci(:,2) - best); % Standard deviation
    % xCovariance = (jacobian(:,1:3)'*jacobian(:,1:3))\var(residual)%
    % Covarianve estimates FIX
    
    %% Error estiamation Mathematical Approach
    n = length(time_TK_1{2+i});
    
    DOF = n-1;
    
    K = 5;
    
    SSE = residual'*residual;
    
    s = sqrt(SSE/(n-(K+1)));
    
    jacobian = full(jacobian)
    C = (jacobian(:,1:5)'*jacobian(:,1:5))\eye(K); %%% same if I use inv
    
    COVARIANCE = (jacobian(:,1:5)'*jacobian(:,1:5))\eye(K )*s^2
    
    CORRELATION_MATRIX = corrcov(COVARIANCE) %corrcov VERIFIED
    
    CONDITION_NUMBER = cond(COVARIANCE); %cONDITION NUMBER OF INFORMATION MATRIX
    
    rmin_error(i) = tinv(0.68+(1-0.68)/2,DOF)*s*sqrt(C(1,1));
    
    A_error(i) =  tinv(0.68+(1-0.68)/2,DOF)*s *sqrt(C(2,2));
    
    lamda_error(i) = tinv(0.68+(1-0.68)/2,DOF)*s *sqrt(C(3,3));
    
    
    %%
    RESIDUAL(:,i) = {residual(:,1)};
    y_modeled_values(:,i) = {dataset{2+i}+residual(:,1)};
    
    % hold on
    % figure(15987)
    % WQ = rand(1,3);
    % plot(time_TK_1{2+i},dataset{2+i},'o','color',WQ)
    % hold on
    % plot(time_TK_1{2+i},y_modeled_values{i},'-','color',WQ) % Value of objective function at solution, returned as an array. In general, residual = fun(x,xdata)-ydata.
    
    fprintf('New parameters:rmin =  %f,A =  %f,Lambda =  %f',best(1:3))
    fprintf('The''trust-region-reflective'' algorithm took %d function evaluations.',...
        output.funcCount)
    %
    % f = @(t,a)[(Kg*(1-a(2)/Nmax)+Kd)*a(2);...
    %     a(2)*(-Kg*a(2)/Nmax+Kg-rmin-lambda*A*exp(-A*t)) ];
    
    %% SVD ANALYSIS
    info_matrix_1 = (jacobian(:,1:5)'*jacobian(:,1:5)) %5
     info_matrix_1 = full(info_matrix_1);
    % info_matrix_1(info_matrix_1<0.01) = 1;
    EIG = eig(info_matrix_1);
    [Uinfo_1,Sinfo_1,Vinfo_1] = svd(info_matrix_1)
    
%     n_pc = 2; %%% Change depending on graph
    gmax = 5;
    for g = 1:gmax %npc value
%       for g = 2
        nps_value = g
        format shortG
        USV_temp = Vinfo_1(:,1:g)*inv(Sinfo_1(1:g,1:g))*Uinfo_1(:,1:g)'
        
        SVD_DATA_COVARIANCE = USV_temp*s^2
        TESTING_Right = info_matrix_1*USV_temp
        TESTING_Left = USV_temp*info_matrix_1
        Right_Difference = eye(5)-TESTING_Right
        Right_Error = norm(Right_Difference,'fro')
        
        Left_Difference = eye(5)-TESTING_Right
        Left_Error = norm(Left_Difference,'fro')

        
%          CORRELATION_MATRIX = corrcov(SVD_DATA_COVARIANCE) %corrcov VERIFIED
        
     end
    
    
    figure(2789+i)
    
    sing_1 = diag(Sinfo_1);
    totalVar_1 = norm(sing_1)^2;
    clear varCaptured
    for p = 1:size(sing_1,1)
        varCaptured_1(p) = norm(sing_1(1:p))^2/totalVar_1;
    end
    plot(varCaptured_1,'o-')
    title('Singular values')
    ylabel('(\sigma_1^2 + ... + \sigma_p^2)/(\sigma_1^2 + ... + \sigma_n^2)')
    grid on
    hold off

end

% legend('BS Data','Ntotal')
% grid on 
% hold off





%% EXTRAS 


SUM_NORM = sum(NORM)

Error_all = [rmin_error;A_error;lamda_error];
Error_all = full(Error_all)
Final_values = [Final_rmin;Final_A;Final_lambda]

Kd = 0.36178 ;
Kg = 2.1342;
Nmax = 10^8;
%% Functions 
function pos = paramfun(x,tspan)
% Kg = 1.94774041;
% Nmax = 8.88E+13;
% Kd = 0.037;
% Kg = 2.21513886109267;

Kd = 0.36178 ;
Kg = 2.1342;
Nmax = 10^8;

rmin = x(1);
A = x(2);
lambda = x(3);
xt0 = x(4:5);

f = @(t,a)[(Kg*(1-a(2)/Nmax)+Kd)*a(2);...
    a(2)*(-Kg*a(2)/Nmax+Kg-rmin-lambda*A*exp(-A*t)) ];

%    options2 = odeset('RelTol',1e-10,'AbsTol',1e-12,'NormControl','on');%'Maxstep',1e-5,,'Stats','on'1 ,'NormControl','on'

[~,pos] = ode45(f,tspan,xt0); %change # 3
pos = log10(pos(:,1));
end