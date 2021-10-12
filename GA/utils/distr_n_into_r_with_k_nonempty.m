function num = distr_n_into_r_with_k_nonempty(n, r, k)
%% Inputs
% n - number of distinct objects
% r - number of distinct groups to divide into
% k - specific groups that need to be non-empty

assert(r<=n, 'Can make at most n distinct groups');
assert(k<=r, 'Number of non-empty groups must be less than the number of groups');

num = 0;
for i=0:1:(n-k)
    num = num + nchoosek(n, i) * distr_n_into_r(n-i, k) * (r-k)^i;
end
    
end