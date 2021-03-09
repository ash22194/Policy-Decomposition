function num = trees_with_n_nodes_and_k_nonzero_leaves(n, k)
%% Inputs
% n - number of distinct nodes in the tree
% k - number of nodes that are leaves (belong to a single edge) except the 0 node

assert(k <= n-1, 'Number of leaves cannot be more than n-1');
assert(k >= 1, 'Tree must have at least 1 non-zero leaf');

if (k==1)
    num = nchoosek(n-1, k) * distr_n_into_r(n-2, n-(k+1)); 
elseif (k==(n-1))
    num = nchoosek(n-1, k) * distr_n_into_r(n-2, n-k); 
else
    num = nchoosek(n-1, k) * distr_n_into_r(n-2, n-(k+1)) ...  % when 0 is a leaf node
            + nchoosek(n-1, k) * distr_n_into_r(n-2, n-k); % when 0 is not a leaf node
end
end