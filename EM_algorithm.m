clear
clc


N = 1e4;
rng('default');

% Class 0
mu_0 = [0,0];
cov_0 = [1, 0.64; 0.64, 1];
% Class 1
mu_1 = [0.5,0.5];
cov_1 = [1, -0.64; -0.64, 1];

% simulating data
S_0 = mvnrnd(mu_0, cov_0, N/2);
S_1 = mvnrnd(mu_1, cov_1, N/2);


% gather whole data with labels
labels = [zeros(N/2,1); ones(N/2,1)];
data = [S_0; S_1];


% scatter plot
scatter(S_0(:,1),S_0(:,2),'b')
hold on
scatter(S_1(:,1),S_1(:,2),'r')


% multivariate function
normal_dist = @(x, mu, cov) 1/2*(1/(2*pi*sqrt(det(cov))) * exp(-1/2 * ((x-mu) * inv(cov) * (x-mu).')));

% randomly assign parameters (our starting point)
mu_0_hat = [-1,-1];
mu_1_hat = [1,1];
cov_0_hat = [1,0;0,1];
cov_1_hat = [1,0;0,1];



repeat = 50;

a = zeros(N, 1);
b = zeros(N, 1);

for r=1:repeat
    % prior
    prior_0 = find_prior(data, normal_dist, mu_0_hat, mu_1_hat, cov_0_hat, cov_1_hat);

    % expectation step
    for i=1:N
        x = data(i,:);
        % bayes rule
        % p(0|x) = (p(x|0)*p(0)) / (p(x|0)*p(0) + p(x|1)*p(1))
        p_0_x= normal_dist(x, mu_0_hat, cov_0_hat);
        p_1_x= normal_dist(x, mu_1_hat, cov_1_hat);
        a(i) =  p_0_x * prior_0 / (p_0_x * prior_0 + p_1_x * (1-prior_0));
        b(i) = 1 - a(i);   
    end
    
    % maximization step
    cov_0_hat = zeros(2);
    cov_1_hat = zeros(2);
    sum_a = sum(a);
    sum_b = sum(b);
    for i=1:N
        x = data(i,:);
        cov_0_hat = cov_0_hat + (a(i) * (x-mu_0_hat).' * (x-mu_0_hat));
        cov_1_hat = cov_1_hat + (b(i) * (x-mu_1_hat).' * (x-mu_1_hat));
    end
    cov_0_hat = cov_0_hat / sum_a;
    cov_1_hat = cov_1_hat / sum_b;
    
    mu_0_hat = sum(a.*data) / sum(a);
    mu_1_hat = sum(b.*data) / sum(b);
end


x1range = min(data(:,1)):.03:max(data(:,1));
x2range = min(data(:,2)):.03:max(data(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
xy = [xx1(:) xx2(:)];

pos_data = [];
neg_data = [];
pos_data_oracle = [];
neg_data_oracle = [];

predict = @(x,mu_0, mu_1, cov_0, cov_1) int8(normal_dist(x,mu_0,cov_0) < normal_dist(x,mu_1,cov_1));


for i=1:length(xy)
   x = xy(i,:);
   if predict(x, mu_0_hat, mu_1_hat, cov_0_hat, cov_1_hat) == 1
      pos_data = [pos_data; x];
   else
      neg_data = [neg_data; x];
   end
   if predict(x, mu_0, mu_1, cov_0, cov_1) == 1
      pos_data_oracle = [pos_data_oracle; x]; 
   else
      neg_data_oracle = [neg_data_oracle; x]; 
   end
end

figure;
scatter(pos_data(:,1),pos_data(:,2),'r')
hold on
scatter(neg_data(:,1),neg_data(:,2),'b')
title('Decision boundaries of the EM Algorithm');


figure;
scatter(pos_data_oracle(:,1),pos_data_oracle(:,2),'r')
hold on
scatter(neg_data_oracle(:,1),neg_data_oracle(:,2),'b')
title('Decision boundaries of Oracle');

false_alarms = 0;
detections = 0;

false_alarms_oracle = 0;
detections_oracle = 0;

os = 0;
os_oracle = 0;

for i=1:N
   x = data(i,:);
   if predict(x, mu_0_hat, mu_1_hat, cov_0_hat, cov_1_hat) == 1
       os = os + 1;
       if labels(i) == 0
           false_alarms = false_alarms + 1;
       else
           detections = detections + 1;
       end
   end
   if predict(x, mu_0, mu_1, cov_0, cov_1) == 1
       os_oracle = os_oracle + 1;
       if labels(i) == 0
           false_alarms_oracle = false_alarms_oracle + 1;
       else
           detections_oracle = detections_oracle + 1;
       end
   end
end

fprintf('EM algorithm false alarm rate: %f\n',false_alarms*2/N);
fprintf('Oracle false alarm rate: %f\n',false_alarms_oracle*2/N);
fprintf('EM algorithm detection rate: %f\n',detections*2/N);
fprintf('Oracle detection rate: %f\n',detections_oracle*2/N)
fprintf('EM algorithm Pi_1: %f\n',os/N);
fprintf('Oracle Pi_1: %f\n',os_oracle/N);

fprintf('mu_0 vs mu_0_hat: [%f,%f] - [%f,%f]\n', mu_0(1),mu_0(2), mu_0_hat(1),mu_0_hat(2));
fprintf('mu_1 vs mu_1_hat: [%f,%f] - [%f,%f]\n', mu_1(1),mu_1(2), mu_1_hat(1),mu_1_hat(2));

fprintf('cov_0: [%f,%f;%f,%f]\n', cov_0(1,1),cov_0(1,2),cov_0(2,1),cov_0(2,2));
fprintf('cov_0_hat: [%f,%f;%f,%f]\n', cov_0_hat(1,1),cov_0_hat(1,2),cov_0_hat(2,1),cov_0_hat(2,2));

fprintf('cov_1: [%f,%f;%f,%f]\n', cov_1(1,1),cov_1(1,2),cov_1(2,1),cov_1(2,2));
fprintf('cov_1_hat: [%f,%f;%f,%f]\n', cov_1_hat(1,1),cov_1_hat(1,2),cov_1_hat(2,1),cov_1_hat(2,2));



