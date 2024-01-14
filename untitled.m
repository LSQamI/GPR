close all;

today=143;
train_day=142;
test_day=50;

x=1:train_day;
xtest=train_day+1:train_day+test_day;
%ytest=temperature(today:1:today+test_day-1,2);
ytest=zeros(50,1);
y=temperature(today-train_day:1:today-1,2)';
m=mean(y);
y=y-m;

xplot = 1:0.2:(test_day+train_day);
[K,sigma_f,rho,sigma_n,i] = Optm(x,y);
sigma_f
rho
i

K1=SqrExp1(x,xplot,sigma_f,rho);
K2=SqrExp2(xplot,xplot,sigma_f,rho,sigma_n);
K_inv=inv(K);
yforecast=K1'*K_inv*y'+m;
var_y2=K2-K1'*K_inv*K1;

xconf = [xplot+1879+today-train_day,xplot(end:-1:1)+1879+today-train_day] ;
yconf = [yforecast'+1.96*sqrt(diag(var_y2)'), yforecast(end:-1:1)'-1.96*sqrt(diag(var_y2)')];
figure;
p=fill(xconf,yconf,'red','FaceColor',[1 0.8 0.8],'EdgeColor','none');
hold on;
plot(xplot+1879+today-train_day,yforecast,'-');
plot(x+1879+today-train_day,y+m,'*');
%plot(xtest+1879+today-train_day,ytest,'o');
ylim([min([ytest',y+m])-0.5,max([ytest',y+m])+1.5]);
xlim([1879+today-train_day,1879+today+test_day]);
set(gca,'Fontsize',16);
xlabel('year');
ylabel('Temperature/\circC');
title('Global Surface Temperature Anomalies');
hold off;

% error_compute=1:1:test_day;
% test_error=zeros(1,length(error_compute));
% for i=1:length(error_compute)
%     [~,pos]=ismember(xtest(1:error_compute(i)),xplot);
%     y_error_compute=yforecast(pos);
%     %RMSE
%     %test_error(i)=norm(y_error_compute-ytest(1:error_compute(i)))/sqrt(error_compute(i));
%     %test_error(i)=mean(abs(y_error_compute-ytest(1:error_compute(i)))./ytest(1:error_compute(i)));
%     %MAE
%     test_error(i)=mean(abs(y_error_compute-ytest(1:error_compute(i))));
% end
% %test_error
% figure;
% plot(error_compute,test_error,'-');
% hold on;
% set(gca,'Fontsize',16);
% xlabel('Number of predicting years');
% ylabel('Mean absolute error/\circC');
% title('The variation trend of Mean absolute error');
% hold off;

function K=SqrExp1(x1,x2,sigma_f,rho)
    temp=x1'*ones(1,length(x2))-ones(length(x1),1)*x2;
    K=(sigma_f^2)*exp((-temp.^2/(2*rho^2)));
end

function K=SqrExp2(x1,x2,sigma_f,rho,sigma_n)
    temp=x1'*ones(1,length(x2))-ones(length(x1),1)*x2;
    K=(sigma_f^2)*exp((-temp.^2/(2*rho^2)))+(sigma_n^2)*eye(length(x1));
end

function [K,sigma_f,rho,sigma_n,i] = Optm(x,y)
%calculate square exponential covariance and optimize its parameters
    
    sigma_f=10;
    sigma_n=0.1;
    rho=80;
    temp=x'*ones(1,length(x))-ones(length(x),1)*x;

    flag=1;
    i=0;
    endval=0.17;
    learningrate=0.01;
    
    while(flag)
        K=(sigma_f^2)*exp((-temp.^2/(2*rho^2)))+(sigma_n^2)*eye(length(x));
        dK_dsigma_f=(K-(sigma_n^2)*eye(length(x)))*2/sigma_f;
        dK_drho=(K-(sigma_n^2)*eye(length(x))).*(temp.^2)/(rho^3);
        K_inv=inv(K);
        alpha=K_inv*y';
        dlikelihood_dsigma_f=0.5*trace((alpha*alpha'-K_inv)*dK_dsigma_f);
        dlikelihood_drho=0.5*trace((alpha*alpha'-K_inv)*dK_drho);
        dlikelihood=[dlikelihood_dsigma_f,dlikelihood_drho];
        likelihood=-0.5*y*K_inv*y'-0.5*log(det(K))-0.5*length(x)*log(2*pi);
%         dlikelihood
        norm(dlikelihood)
%        likelihood
%         rho
%         sigma_f
        if(norm(dlikelihood)>endval&&i<5000)
            sigma_f=sigma_f+learningrate*dlikelihood_dsigma_f;
            rho=rho+learningrate*dlikelihood_drho;
            i=i+1;
        else
            flag=0;
        end
    end
    likelihood

end