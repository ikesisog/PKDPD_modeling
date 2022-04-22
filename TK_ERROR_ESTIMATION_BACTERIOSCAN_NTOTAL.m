%%3rd file after New_bacterioscan_Data_TG-TK_with_assumption_NK_MOD_updated

clearvars -except Kb Hb C50b Kg Final_rmin SD SD_Conc Final_values Error_all
clc
% clearvars -except Kg Kb C_50b Hb  Kk C_50 H A No XO Nmax XMAX
%% 
y=Final_rmin(1:5);
C_original=[1 4 16 64 256]';
n=length(C_original);

%% 
Rmin=Kb.*C_original.^Hb./(C_original.^Hb+C50b^Hb);

for i =1:length(C_original)  
Par_Kb(i)=C_original(i)^Hb/(C_original(i)^Hb + C50b^Hb);
Par_Hb(i)=(C_original(i)^Hb*C50b^Hb*Kb*(log(C_original(i)) - log(C50b)))/(C_original(i)^Hb + C50b^Hb)^2;
Par_C50b(i)=-(C_original(i)^Hb*C50b^(Hb - 1)*Hb*Kb)/(C_original(i)^Hb + C50b^Hb)^2  ;
end


% X256=[dH256',dC_50256',d_Kk256',d_kg256',d_kb256',dC_50b256',dHb256', A256'];
X=[Par_Hb', Par_Kb', Par_C50b'];


% e256=(y256-Time_Kill_Model_exp256);
e=y-Rmin';

 e=e';

SSE=e'*e;
K=3;

% DOF=n-(K+1); %CHANGED
DOF=2;
ss=sqrt(SSE/(n-(K+1)));
%C=ss^2.*inv(X'*X); % it was wrong (inverse not .^(-1), s^2 here as it is in linear form
%issue with singularity cause the S values are of great difference in Order
%of Magnitude. For that reason we do the PCA! See later C_inv



info_matrix=(X'*X);

eig(info_matrix);
[Uinfo,Sinfo,Vinfo]=svd(info_matrix);

n_pc=3; %%% Change depending on graph 


C_inv=Vinfo(:,1:n_pc)*inv(Sinfo(1:n_pc,1:n_pc))*Uinfo(:,1:n_pc)'*ss^2; % covariance matrix CHANGED BASED ON dR. NIKOLAOU WORD DOC " CONFIDENCE_INTERVALS_MN




figure(200) 
 
sing = diag(Sinfo);
totalVar = norm(sing)^2; 
clear varCaptured 
for i = 1:size(sing,1)
    varCaptured(i) = norm(sing(1:i))^2/totalVar;
end
plot(varCaptured,'o-')
title('Singular values')
ylabel('(\sigma_1^2 + ... + \sigma_p^2)/(\sigma_1^2 + ... + \sigma_n^2)')
grid on
hold off
%% CONFIDENCE INTERVALS

C_conf=linspace(0.001,256,1000);

Rmin_conf=Kb.*C_conf.^Hb./(C_conf.^Hb+C50b^Hb);
for i =1:length(C_conf)  
Par_Kb_conf(i)=C_conf(i)^Hb/(C_conf(i)^Hb + C50b^Hb);

Par_Hb_conf(i)=(C_conf(i)^Hb*C50b^Hb*Kb*(log(C_conf(i)) - log(C50b)))/(C_conf(i)^Hb + C50b^Hb)^2;

Par_C50b_conf(i)=-(C_conf(i)^Hb*C50b^(Hb - 1)*Hb*Kb)/(C_conf(i)^Hb + C50b^Hb)^2;      
end
X67=[Par_Hb_conf', Par_Kb_conf', Par_C50b_conf'];
info_matrix_conf=(X67'*X67);
%%
conf=[0.997, 0.95, 0.68];
npoints12=length(C_conf);
for akis=1:length(conf)
    for l=1:npoints12    
     
     Error671112(l)=tinv(conf(akis)+(1-conf(akis))/2,DOF)*ss*sqrt(1+X67(l,1:3)*info_matrix_conf^(-1)*X67(l,1:3)'); %used eqn 116 top page 105 from chee 6397 Data Analytics 

    end 
Error67111(akis,1:npoints12)=Error671112;
end
Error67111=Error67111';
figure (3000)
  %1SD%

 q1=plot(C_original,Final_rmin,'or',...
    'MarkerFaceColor',[1 0 0]); % data points 
% hold on 
%  e=errorbar(C_original,Final_rmin,SD(1,1:5),'.');
%  e.Color='r';
%  e.CapSize = 15;
  
hold on 
q2= plot(C_conf,Rmin_conf,'-','color',[0.0745    0.6235    1.0000],...
    'MarkerFaceColor',[0.0745    0.6235    1.0000]); % Modeled data 

hold on 

%% Adding SD error estimations 
alpha_shading=0.1;
q4=plot(C_conf,Rmin_conf'+Error67111(:,3),'-m');%%%%%% STANDARD ERROR IMPACT 3SD USED HERE
hold on 
q3=plot(C_conf,Kg*ones(length(C_conf)),'-k','LineWidth',3);
hold on 

plot(C_conf,Rmin_conf'-Error67111(:,3),'-m')%%%%%% STANDARD ERROR IMPACT 3SD USED HERE
hold on 
x3=C_conf;
x2 = [x3, fliplr(x3)];

inBetween = [Rmin_conf+Error67111(:,3)', fliplr(Rmin_conf)];
fill(x2, inBetween, 'm');
hold on
inBetween = [Rmin_conf-Error67111(:,3)', fliplr(Rmin_conf)];
fill(x2, inBetween, 'm');
alpha(alpha_shading+0.2)
hold on

e=plot(C_conf,Rmin_conf'+Error67111(:,3),'-w','LineWidth',0.05);
 hold on
f= plot(C_conf,Rmin_conf'-Error67111(:,3),'-w','LineWidth',0.05);
hold on 
 plot(C_conf,Rmin_conf','-w','LineWidth',0.05);
 hold on 
q2= plot(C_conf,Rmin_conf,'-','color',[0.0745    0.6235    1.0000],...
    'MarkerFaceColor',[0.0745    0.6235    1.0000]); % Modeled data 

 hold off

%%

% plot(C_conf,Rmin_conf,'-b')
 legend ({ 'Data ','Model','1s Model','Kg' },'FontSize',20)%,'1s Data'
 
 xratio = 2;
yratio = 1;
widthWindow = 3.25;
heightWindow = 2.5;
xlabel('C (mg/L)','FontSize',20)
ylabel('$r_{\mbox{min}} = K_b \frac {C^{H_b}}{{C^{H_b}+C_{50b}^{H_b}}}$','Interpreter','latex','FontSize',17 )
set(gca,'FontSize',20)
grid on
xlim([0 256])
% ylim([0 3])

