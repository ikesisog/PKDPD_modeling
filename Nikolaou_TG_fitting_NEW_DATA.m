%Fitting Tg model Nikolaou 
clc
clear all


%% Read data 
% Nmax = floor(1.9167/5*60);

data_1=xlsread('BacterioScan CAZ+AMK AB747 (5 log) raw data 0222_AKIS.xlsx','Data','AQ3:AR23'); %updated data set 
data=data_1(1:21,2);
time=data_1(1:21,1);

Nmax=10^8;
%% Plting data 


% No=10^6.212990849;
Kd= 0.013935861 ; 

%%Defining non-linear function
modelfun=@(b,x) log10( b(2)*(exp(b(1).*x)+Kd/b(1)*exp(b(1).*x)-Kd/b(1)));

% modelfun=@(b,x) log10( b(1).*(1./(b(1)./Nmax+exp(-b(2).*x).*(1-b(1)./Nmax)))+Nmax*b(3)/b(2)*log((exp(b(2).*x)-1)*b(1)/Nmax+1));

%% Use Fitnlm
opts=statset('glmfit');
opts.MaxIter = 2000;
beta0=[2.1342 10^6.212990849];

mdl = fitnlm(time,data,modelfun,beta0,'Options',opts)

% properties(mdl)
% 
% predict(mdl,dataset_1)


for i=1:length(beta0)
b(i)=mdl.Coefficients.Estimate(i,1);
end
b=b';
 No=	b(2);
Kg=	b(1);
%  Kd=b(2);

y1=log10( No*(exp(Kg.*time)+Kd/Kg*exp(Kg.*time)-Kd/Kg));

% anova(mdl,'summary')

plot(time,data,'ob')
hold on 
plot(time,y1,'-r')


grid on 
legend 
axis auto 

% CM = mdl.CoefficientCovariance
% 
% % Compute the coefficient standard deviation.
% 
% SE = diag(sqrt(CM))

% SE is your error of each newly estimated parameters, so plot it


