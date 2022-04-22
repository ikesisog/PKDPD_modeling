%Fitting Tg model Nikolaou 
clc
clear all


%% Read data 
Nmax = floor(1.9167/5*60);

dataset_1=xlsread('Akis_7_5_max_condition_NK_DATA_MODIFICATION_vol_2.xlsx','Sheet1','AH29:AH253'); %updated data set 
dataset_1=dataset_1(1:Nmax,1);


dataset_2=xlsread('Akis_7_5_max_condition_NK_DATA_MODIFICATION_vol_2.xlsx','Sheet1','AG19:AG243'); %updated data set 
dataset_2=dataset_2(1:Nmax,1);

%% Plting data 
plot( dataset_2,dataset_1)


%%Defining non-linear function
modelfun=@(b,x) log10( b(1)*(exp(b(2).*x)+b(3)/b(2)*exp(b(2).*x)-b(3)/b(2)));



%% Use Fitnlm
opts=statset('glmfit');
opts.MaxIter = 2000;
beta0=[10^6 1.9 0.0037];

mdl = fitnlm(dataset_2,dataset_1,modelfun,beta0,'Options',opts)

% properties(mdl)
% 
% predict(mdl,dataset_1)

for i=1:length(beta0)
b(i)=mdl.Coefficients.Estimate(i,1);
end
b=b';
N0=	b(1);
Kg=	b(2);
Kd=b(3);

y=log10( N0*(exp(Kg.*dataset_2)+Kd/Kg*exp(b(2).*dataset_2)-Kd/Kg));


plot(dataset_2,dataset_1,'ob')
hold on 
plot(dataset_2,y,'-r')
 
grid on 
legend 
axis auto 






