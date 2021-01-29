function y=find_prior(data, func, mu_0, mu_1, cov_0, cov_1)
N = length(data);
labels = zeros(N,1);

for i = 1:N
   x = data(i,:);
   labels(i) = func(x, mu_0, cov_0) < func(x, mu_1, cov_1);
end
y = sum(labels(:)==0)/N;

end




