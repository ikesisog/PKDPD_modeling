%4th file after TK_ERROR_ESTIAMTION_BACTERIOSCAN_NTOTAL file

%Adding them together

clearvars -except Kg Kb C50b Hb  C_inv Final_values SD Error_all ss
Line_width=5;
akis=0.25;
%%
C_50b=C50b;
% dataset=xlsread('Updated_summed_plating_data_11_15_2020_TESTING.xlsx','Sheet1','A1:AV98');

%SAME FOR WITH AND WITHOUT DR TAMS ASSUMPTION 

X_50HB=Kb/Kg-1;

Period=8; %h
Half_life=2.5; %h
x=Period/Half_life*log(2);

Cmax=linspace(1,450,500);%%%%%%%%%%%%%%%%%%%%
% Cmax=254;
C_avg=0.4017820839*Cmax;
% Y=C_avg/C_50;

YB=C_avg/C50b;

% D_over_KG=zeros(1,500);
D_over_KG_B=zeros(1,length(Cmax));

D_over_KG_B_2=zeros(1,length(Cmax));

% df_dkg_1=zeros(1,length(Cmax));
df_dHb_1=zeros(1,length(Cmax));
df_dKb_1=zeros(1,length(Cmax));
df_dC50b_1=zeros(1,length(Cmax));

%%

for i =1:length(C_avg)
   % i hat 

D_over_KG_B(i)=(1+X_50HB)/(Hb*x)*log((1+(exp(x)*x*YB(i)/(exp(x)-1))^Hb)/(1+(x*YB(i)/(exp(x)-1))^Hb));

D_over_KG_B_2(i)=1/(Period*Kg)*(-(20000*Half_life*Kb*log((Cmax(i)*exp(-(13863*Period)/(20000*Half_life)))^Hb + C50b^Hb))/(13863*Hb)-(-(20000*Half_life*Kb*log((Cmax(i)*exp(-(13863*0)/(20000*Half_life)))^Hb + C50b^Hb))/(13863*Hb)));



df_dHb_1(i)=(Kb*((log((2*Cmax(i)*x*exp(x))/(5*C_50b*(exp(x) - 1)))*((2*Cmax(i)*x*exp(x))/(5*C_50b*(exp(x) - 1)))^Hb)/(((2*Cmax(i)*x)/(5*C_50b*(exp(x) - 1)))^Hb + 1) - (log((2*Cmax(i)*x)/(5*C_50b*(exp(x) - 1)))*(((2*Cmax(i)*x*exp(x))/(5*C_50b*(exp(x) - 1)))^Hb + 1)*((2*Cmax(i)*x)/(5*C_50b*(exp(x) - 1)))^Hb)/(((2*Cmax(i)*x)/(5*C_50b*(exp(x) - 1)))^Hb + 1)^2)*(((2*Cmax(i)*x)/(5*C_50b*(exp(x) - 1)))^Hb + 1))/(Hb*Kg*x*(((2*Cmax(i)*x*exp(x))/(5*C_50b*(exp(x) - 1)))^Hb + 1)) - (Kb*log((((2*Cmax(i)*x*exp(x))/(5*C_50b*(exp(x) - 1)))^Hb + 1)/(((2*Cmax(i)*x)/(5*C_50b*(exp(x) - 1)))^Hb + 1)))/(Hb^2*Kg*x);


df_dKb_1(i)=log((((2*Cmax(i)*x*exp(x))/(5*C_50b*(exp(x) - 1)))^Hb + 1)/(((2*Cmax(i)*x)/(5*C_50b*(exp(x) - 1)))^Hb + 1))/(Hb*Kg*x);


df_dC50b_1(i)=(2^Hb*5^Hb*Kb*(((Cmax(i)*x)/(C_50b*(exp(x) - 1)))^Hb - ((Cmax(i)*x*exp(x))/(C_50b*(exp(x) - 1)))^Hb))/(C_50b*Kg*x*(2^Hb*((Cmax(i)*x)/(C_50b*(exp(x) - 1)))^Hb + 5^Hb)*(5^Hb + 2^Hb*((Cmax(i)*x*exp(x))/(C_50b*(exp(x) - 1)))^Hb));


end

 figure(101)
 d1=plot(Cmax,D_over_KG_B_2,'-','color',[0.0745    0.6235    1.0000],...
    'MarkerFaceColor',[0.0745    0.6235    1.0000]);
grid on 

 figure(100)
 d1=plot(Cmax,D_over_KG_B,'-','color',[0.0745    0.6235    1.0000],...
    'MarkerFaceColor',[0.0745    0.6235    1.0000]);
grid on 

 COV=C_inv;

alpha111=[df_dHb_1' ,df_dKb_1',df_dC50b_1'];

% Var_D_Kg= mean(mean(alpha*COV*(alpha'))'+(D_over_KG').^2)
for W=1:length(Cmax)
Var_D_Kg(W)= alpha111(W,:)*COV*alpha111(W,:)'+ss^2; %% CORRECTION HAS BEEN INCORPORATED σ_"noise" ^2 x^⊤ (X^T X)^(-1) x+σ_"noise" ^2
end
Var_D_Kg=Var_D_Kg';
s=(Var_D_Kg.^2).^0.25;
s=real(s);
Cmax=Cmax';
D_over_KG_B=D_over_KG_B';



%%
hold on

g=fill([Cmax;flipud(Cmax)],[D_over_KG_B+s;flipud(D_over_KG_B-s)],'k');
alpha(akis)
hold on
e=plot(Cmax,D_over_KG_B+s,'-w','LineWidth',0.05);
 hold on
f= plot(Cmax,D_over_KG_B-s,'-w','LineWidth',0.05);
hold on
one=ones(length(Cmax),1);
plot(Cmax,one,'-r')
xlabel('Cmax (\mug/ml)','FontSize',17)
ylabel('D/Kg','FontSize',17)
set(gca,'FontSize',17)
grid on

legend ( [d1,g],{ 'D/Kg','1s'});
% test=Var_D_Kg;
% UpperLimit=D_over_KG_B+test;
% bottom=D_over_KG_B-test;

syms x 

m=D_over_KG_B;

   
for i=1:length(Cmax)

   Area222(i)=-1/2*erf((D_over_KG_B(i)-inf)/(2^0.5*s(i)))+1/2*erf((D_over_KG_B(i)-1)/(2^0.5*s(i)));
%  % wolframalpha


 Area(i)=-(1125899906842624*2^(1/2)*pi^(1/2)*erf((2^(1/2)*(D_over_KG_B(i) - 10^99)*(1/s(i)^2)^(1/2))/2))/(5644425081792261*s(i)*(1/s(i)^2)^(1/2))+(1125899906842624*2^(1/2)*pi^(1/2)*erf((2^(1/2)*(D_over_KG_B(i) - 1)*(1/s(i)^2)^(1/2))/2))/(5644425081792261*s(i)*(1/s(i)^2)^(1/2));
% matlab both correct


end

figure(3)

% plot(Cmax,Area222,'--')
% figure(4)
 plot(Cmax,Area,'--r')
xlabel('Cmax (\mug/ml)','FontSize',17)
ylabel('P[D/Kg] >1 ','FontSize',17)

set(gca,'FontSize',17)
legend ( { 'P[D/Kg] >1'});
grid on 




