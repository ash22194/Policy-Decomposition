function num = distr_n_into_r(n, r)
%% Inputs
% n - number of distinct objects
% r - number of distinct non-empty groups to divide into

assert(r<=(n), 'Can make at most n distinct groups');

num = 0;
for i=0:1:r
    num = num + nchoosek(r, i) * (r-i)^n * (-1)^i;
end
    
end