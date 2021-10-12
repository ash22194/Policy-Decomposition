function num = calcAllDecompositions(n, m)
%% Inputs
% n - number of states
% m - number of inputs

num = 0;
for r=2:1:m
    % Loop over possible number of pseudo-inputs 
    
    % For policy calculation
    % r = 1 represents all inputs coupled (which we exclude from the count)
    % r = m represents treating all inputs independently
    G = distr_n_into_r(m, r) / factorial(r); % Number of pseudo-input sets
    num_r_decompositions = 0;
    for k=1:1:r
        % Loop over possible number of leaves in action tree
        % k = 1 represents a purely cascaded policy decomposition
        % k = r represents a purely decoupled policy decomposition
        num_r_decompositions = num_r_decompositions ...
                               + trees_with_n_nodes_and_k_nonzero_leaves(r+1, k) ...
                                 * distr_n_into_r_with_k_nonempty(n, r, k);
        
    end
    num = num + num_r_decompositions * G;
end
end