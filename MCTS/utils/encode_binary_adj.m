function encoding = encode_binary_adj(action_tree)

    X_DIMS = sum(cellfun(@(x) sum(x), action_tree(:,2)));
    U_DIMS = sum(cellfun(@(x) length(x), action_tree(:,1))) - 1;
    
    state_dependence = zeros(U_DIMS, X_DIMS);
    adjacency_matrix = zeros(U_DIMS, U_DIMS);
    
    for aa =1:1:size(action_tree,1)
        node = action_tree{aa, 1};
        if all(node~=0)
            node_bool = zeros(U_DIMS, 1);
            node_bool(node) = 1;
            state_dependence(logical(node_bool * action_tree{aa, 2})) = 1;
            adjacency_matrix(logical(node_bool * node_bool')) = 1;
            
            children = action_tree{aa, end};
            for cc=1:1:length(children)
                child = children{cc};
                child_bool = zeros(U_DIMS, 1);
                child_bool(child) = 1;
                adjacency_matrix(logical(child_bool * node_bool')) = 1;
            end
        end
    end
    
    adjacency_matrix = adjacency_matrix - diag(diag(adjacency_matrix));
    encoding = [reshape(adjacency_matrix, 1, U_DIMS * U_DIMS), ...
                reshape(state_dependence, 1, U_DIMS * X_DIMS)];
end