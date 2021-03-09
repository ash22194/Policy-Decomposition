function [X, U, c] = ilqg_decomposition_multitraj(sys, Op, p, s, starts)
%%  Inputs
    % sys    - system description
    % Op     - optimization parameters
    % (p,s)  - action tree and state assignment for the decomposition
    % starts - start states for ilqg calculations

%% Logging
    if (~isfield(Op, 'save_dir'))
        Op.save_dir = '';
    end
    save_dir = strcat(Op.save_dir, '/', sys.name, '/decomp', num2str(sys.decomposition_id));
    if (~isdir(save_dir))
        mkdir(save_dir);
    end
    
%%  Build action tree
    p = round(p);
    s = logical(round(s));
    assert(any(p(:,1)==0), 'Root of the tree must be 0');
    assert(all(p(:,1)~=linspace(1,sys.U_DIMS,sys.U_DIMS)'), 'No input can be its own parent');

    p_ = [linspace(1, sys.U_DIMS, sys.U_DIMS)', p];
    action_tree = {};
    queue = {{0; -1}};

    while(~isempty(queue))
        curr_node = queue{1}; % pop the top element
        curr_parent = curr_node{2};
        curr_node = curr_node{1};
        if (curr_node~=0)
            curr_state = s(curr_node(1), :);
        else
            curr_state = zeros(1, sys.X_DIMS);
        end
        
        queue = queue(2:end);
        children = p_(any(p_(:,2)==curr_node', 2), [1, 3]); % Find inputs that are children to curr_parent
        childID = unique(children(:,2));                    % Find unique childIDs, inputs are coupled if child IDs are same
        curr_children = cell(length(childID), 1);
        for jj=1:1:length(childID)
            curr_children{jj} = children(children(:,2)==childID(jj), 1);
            assert(all(s(curr_children{jj}, :) == prod(s(curr_children{jj}, :), 1), 'all'), ...
                   'Coupled inputs must have same state assignment');
               
            queue{end+1} = {curr_children{jj}; curr_node};
        end

        sub_policies_LQR = cell(0, 5);
        sub_policies_DDP = cell(0, 6);
        action_tree(end+1, :) = {curr_node, curr_state, ...
                                 sub_policies_LQR, sub_policies_DDP, ...
                                 curr_parent, curr_children};
    end

%% Compute DDP solutions for subsystems
    
    % Optimization parameters
    Op.gamma_ = sys.gamma_;
    initial_trajectories = cell(0, 5);
    
    while (~isempty(action_tree))
        % Find leaf nodes
        leaf_node_ids = cellfun(@(x) isempty(x), action_tree(:,end));
        leaf_nodes = action_tree(leaf_node_ids, :);
    
        if (size(leaf_nodes, 1)==1 && all(leaf_nodes{1,1}==0))
            load_successful = false;
            if (isfield(Op, 'reuse_policy') && Op.reuse_policy && isfile(strcat(save_dir, '/final.mat')))
                prev_save = load(strcat(save_dir, '/final.mat'), 'p', 's', 'X', 'U', 'c');
                if (all(prev_save.p==p, 'all') && all(prev_save.s==s, 'all'))
                    load_successful = true;
                    X = prev_save.X;
                    U = prev_save.U;
                    c = prev_save.c;
                end
            end
            if (~load_successful)
                % Compute the final trajectories
                sys_ = sys;
                sys_.X_DIMS_FREE = linspace(1, sys_.X_DIMS, sys_.X_DIMS)';
                sys_.X_DIMS_FIXED = [];
                sys_.U_DIMS_FREE = [];
                sys_.U_DIMS_CONTROLLED = linspace(1, sys_.U_DIMS, sys_.U_DIMS)';
                sys_.U_DIMS_FIXED = [];
                sub_policies_DDP = leaf_nodes{1, 4};
                [X, U, c] = ForwardPassGeneral_multitraj(sys_, sub_policies_DDP, starts);
                save(strcat(save_dir, '/final.mat'), 'sys', 'action_tree', 'initial_trajectories', 'p', 's', 'X', 'U', 'c');
            end
            return;
        end
        
        % Compute DDP solutions for all leaf nodes and update the action tree
        for ii=1:1:size(leaf_nodes,1)
            U_DIMS_FREE = leaf_nodes{ii, 1};
            U_DIMS_CONTROLLED = [];
            X_DIMS_FREE = find(leaf_nodes{ii, 2})';
            
            % Compute LQR trajectories to seed DDP computation
            K = zeros(sys.U_DIMS, sys.X_DIMS);
            u0 = zeros(sys.U_DIMS, 1);
            u0(U_DIMS_FREE) = sys.u0(U_DIMS_FREE);
            sub_policies_LQR = leaf_nodes{ii, 3};
            NUM_SUBSYS = size(sub_policies_LQR, 1);
            for jj=1:1:NUM_SUBSYS
                K(sub_policies_LQR{jj, 1}, sub_policies_LQR{jj, 2}) = sub_policies_LQR{jj, 4};
                u0(sub_policies_LQR{jj, 1}) = sub_policies_LQR{jj, 3};
                
                U_DIMS_CONTROLLED = [U_DIMS_CONTROLLED; sub_policies_LQR{jj, 1}];
                X_DIMS_FREE = [X_DIMS_FREE; sub_policies_LQR{jj, 2}];
            end
            X_DIMS_FREE = sort(unique(X_DIMS_FREE));
            
            if (isfield(sys, 'fxfu_func'))
                fxfu = sys.fxfu_func(sys.l_point, u0);
            else
                fxfu = eval(subs(sys.fxfu, [sys.x; sys.u], [sys.l_point; u0]));
            end
            A_ = fxfu(:,1:sys.X_DIMS);
            B_ = fxfu(:,(sys.X_DIMS+1):end);
            Q_ = sys.Q;
            R_ = sys.R;

            A_ = A_ + B_*K;
            A_ = A_(X_DIMS_FREE, X_DIMS_FREE);
            B_ = B_(X_DIMS_FREE, U_DIMS_FREE);
            Q_ = Q_ + K'*R_*K;
            Q_ = Q_(X_DIMS_FREE, X_DIMS_FREE);
            R_ = R_(U_DIMS_FREE, U_DIMS_FREE);
            try
                [K_, ~, ~] = lqr(A_ - eye(size(A_, 1))*sys.lambda_/2, B_, ...
                                 Q_, R_, zeros(size(A_, 1), size(B_, 2)));
            catch ME
                K_ = zeros(size(B_,2), size(A_,1));
            end
            
            % Compute DDP trajectories
            sub_policies_DDP = leaf_nodes{ii, 4};
            sub_policies_DDP_ = sub_policies_DDP;
            sub_policies_LQR_ = sub_policies_LQR;
            for jj=1:1:NUM_SUBSYS
                [~, sub_policies_LQR_{jj, 1}] = find(sub_policies_LQR_{jj, 1} == U_DIMS_CONTROLLED');
                [~, sub_policies_LQR_{jj, 2}] = find(sub_policies_LQR_{jj, 2} == X_DIMS_FREE');
                [~, sub_policies_DDP_{jj, 1}] = find(sub_policies_DDP_{jj, 1} == U_DIMS_CONTROLLED');
                [~, sub_policies_DDP_{jj, 2}] = find(sub_policies_DDP_{jj, 2} == X_DIMS_FREE');
            end
            sub_policies_LQR = cat(1, sub_policies_LQR, ...
                                   {U_DIMS_FREE, X_DIMS_FREE, sys.u0(U_DIMS_FREE), -K_, sys.l_point(X_DIMS_FREE)});
            sys_ = sys;
            sys_.X_DIMS_FREE = X_DIMS_FREE;
            sys_.X_DIMS_FIXED = linspace(1, sys_.X_DIMS, sys_.X_DIMS)';
            sys_.X_DIMS_FIXED(X_DIMS_FREE) = [];
            sys_.U_DIMS_CONTROLLED = U_DIMS_CONTROLLED;
            sys_.U_DIMS_FREE = U_DIMS_FREE;
            sys_.U_DIMS_FIXED = linspace(1, sys_.U_DIMS, sys_.U_DIMS)';
            sys_.U_DIMS_FIXED([U_DIMS_CONTROLLED; U_DIMS_FREE]) = [];
            Op.lims = sys_.lims(sys_.U_DIMS_FREE, :);
            starts_unique = unique(starts(X_DIMS_FREE, :)', 'stable', 'rows')';
            
            subsystem_id = strcat('U', sprintf('%d', U_DIMS_FREE), '_X', sprintf('%d', X_DIMS_FREE));
            if (isfield(Op, 'reuse_policy') && (Op.reuse_policy) && isfile(strcat(save_dir, '/', subsystem_id, '.mat')))
                
                decomposition = load(strcat(save_dir, '/', subsystem_id, '.mat'));
                Xinit = decomposition.Xinit;
                Uinit = decomposition.Uinit;
                Cinit = decomposition.Cinit;
                
                X_DDP = decomposition.X_DDP;
                X_Final = decomposition.X_Final;
                k_DDP = decomposition.k_DDP;
                K_DDP = decomposition.K_DDP;
            else
                [X_Final, U_Final, X_DDP, k_DDP, K_DDP, ...
                 ~, ~, ~, ~, Xinit, Uinit, Cinit] = get_ilqg_trajectory_multitraj_parallel(sys_, Op, starts_unique, ...
                                                            -K_, sub_policies_LQR_, sub_policies_DDP_);
                
                save(strcat(save_dir, '/', subsystem_id, '.mat'), 'X_Final', 'U_Final', 'X_DDP', 'k_DDP', 'K_DDP', 'Xinit', 'Uinit', 'Cinit', 'sys_');
            end
            
            sub_policies_DDP = cat(1, sub_policies_DDP, ...
                                   {U_DIMS_FREE, X_DIMS_FREE, k_DDP, K_DDP, X_DDP, X_Final});
            initial_trajectories = cat(1, initial_trajectories, ...
                                   {U_DIMS_FREE, X_DIMS_FREE, Xinit, Uinit, Cinit});

            % Update the parent node
            parent_input = leaf_nodes{ii, 5};
            parent_node_id = find(cellfun(@(x) isempty(setdiff(x, parent_input)) ...
                                               && isempty(setdiff(parent_input, x)), ...
                                  action_tree(:, 1)));
            assert(length(parent_node_id)==1, 'Can only be one parent node ID');
            parent_node = action_tree(parent_node_id, :);
            assert(all(~(parent_node{2} .* leaf_nodes{ii, 2})), 'Parent-child cannot have state overlap');
            
            parent_node{1, 3} = cat(1, parent_node{1, 3}, sub_policies_LQR);
            parent_node{1, 4} = cat(1, parent_node{1, 4}, sub_policies_DDP);
            
            % Find and delete the leaf_node in the list of children
            children_list = parent_node{end};
            childID = cellfun(@(x) isempty(setdiff(x, leaf_nodes{ii, 1})) ...
                                   && isempty(setdiff(leaf_nodes{ii, 1}, x)), ...
                              children_list);
            children_list(childID) = [];
            parent_node{end} = children_list;

            action_tree(parent_node_id, :) = parent_node;
        end
        action_tree(leaf_node_ids, :) = [];
    end
    
end