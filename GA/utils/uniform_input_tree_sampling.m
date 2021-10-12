function population = uniform_input_tree_sampling(sys, n)

    X_DIMS_MASKED = sys.X_DIMS_MASKED;
    U_DIMS = sys.U_DIMS;
    X_DIMS = sys.X_DIMS;
    
    numTreesOfSizes = zeros(U_DIMS-1,1);
    inputSubsetsOfSizes = cell(U_DIMS-1, 1);
    for uu=2:1:U_DIMS
        numTreesOfSizes(uu-1,1) = calcAllDecompositionsSizeR(size(X_DIMS_MASKED, 1), U_DIMS, uu);
        inputSubsetsOfSizes{uu-1, 1} = partitions(U_DIMS, uu);
    end
    rTreeLikelihood = numTreesOfSizes ./ sum(numTreesOfSizes);
    
    p = zeros(U_DIMS, 2, n);
    s = zeros(U_DIMS, X_DIMS, n);
    for pp=1:1:n
        r = find(mnrnd(1, rTreeLikelihood)) + 1;
        inputs = randi([1, size(inputSubsetsOfSizes{r-1}, 1)], 1);
        inputs = inputSubsetsOfSizes{r-1}{inputs};
        inputs = cat(2, inputs, [0]);
        % Trees with r + 1 nodes and k non-zero leafs (k = 1, ..., r)
        treesWithKLeafs = zeros(r, 1);
        for k=1:1:r
            treesWithKLeafs(k) = trees_with_n_nodes_and_k_nonzero_leaves(r+1, k) ...
                                    * distr_n_into_r_with_k_nonempty(size(X_DIMS_MASKED, 1), r, k);
        end
        kLeafTreeLikelihood = treesWithKLeafs ./ sum(treesWithKLeafs);
        
        k = find(mnrnd(1, kLeafTreeLikelihood));
        % Sample a prufer code of length (r-1) such that it has only (r-k)
        % distinct entries other than the zero node
        is_valid = false;
        while(~is_valid)
            prufer_code = randi([1, r+1], 1, r-1);
            non_leafs = unique(prufer_code);
            non_leafs(non_leafs==(r+1)) = [];
            is_valid = length(non_leafs) == (r-k);
        end
        leafs_non_zero = linspace(1, r, r)';
        leafs_non_zero(non_leafs) = [];
        assert(length(leafs_non_zero) <= size(X_DIMS_MASKED, 1));
        
        node_list = linspace(1, r + 1, r + 1)';
        child_count = zeros(r + 1, 1);
        while(length(node_list) > 2)
            child = min(setdiff(node_list, prufer_code));
            parent = prufer_code(1);
            child_count(parent) = child_count(parent) + 1;
            
            node_list(node_list == child) = [];
            prufer_code = prufer_code(2:end);
            
            p(inputs{child}, 1, pp) = inputs{parent}(1);
            p(inputs{child}, 2, pp) = child_count(parent);
        end
        child = node_list(1);
        parent = node_list(2);
        child_count(parent) = child_count(parent) + 1;
        p(inputs{child}, 1, pp) = inputs{parent}(1);
        p(inputs{child}, 2, pp) = child_count(parent);
        
        % State assignment
        is_valid_state_assignment = false;
        while(~is_valid_state_assignment)
            state_assign = randi([1,r], 1, size(X_DIMS_MASKED, 1));
            is_valid_state_assignment = all(any(leafs_non_zero == state_assign, 2));
        end
        
        for ss=1:1:size(X_DIMS_MASKED, 1)
            s(inputs{state_assign(ss)}, :, pp) = s(inputs{state_assign(ss)}, :, pp) ...
                                                 + repmat(X_DIMS_MASKED(ss, :), ...
                                                          [length(inputs{state_assign(ss)}), 1]);
        end
        
        [c, ~] = constraints(p(:,:,pp), s(:,:,pp));
        assert((all(c<=0)), 'Invalid Decomposition');
    end
    
    population = [reshape(p, 2*U_DIMS, n)', reshape(s, U_DIMS*X_DIMS, n)'];
end