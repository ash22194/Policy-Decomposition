function [policies, value, info] = dp_decomposition_gpu(sys, Op, p, s)
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
    Op.save_dir = save_dir;
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
        
        sub_policies = cell(0, 4);
        action_tree(end+1, :) = {curr_node, curr_state, ...
                                 sub_policies, curr_parent, curr_children};
    end

%% Compute DP solutions for subsystems
    
    while (~isempty(action_tree))
        % Find leaf nodes
        leaf_node_ids = cellfun(@(x) isempty(x), action_tree(:,end));
        leaf_nodes = action_tree(leaf_node_ids, :);
    
        if (size(leaf_nodes, 1)==1 && all(leaf_nodes{1,1}==0))
            if (isfield(Op, 'reuse_policy') && Op.reuse_policy && isfile(strcat(save_dir, '/final.mat')))
            	load(strcat(save_dir, '/final.mat'), 'policies', 'value', 'info');
            	return;
            else
            	% Compute the final value function
            	sys_ = sys;
            	sys_.X_DIMS_FREE = (1:sys_.X_DIMS)';
            	sys_.X_DIMS_FIXED = [];
            	sys_.U_DIMS_FREE = [];
            	sys_.U_DIMS_CONTROLLED = (1:sys_.U_DIMS)';
            	sys_.U_DIMS_FIXED = [];
            	sub_policies = leaf_nodes{1, 3};
                
                if (isfield(Op, 'return_value_function') && (Op.return_value_function))
                    [value, info] = policy_evaluation_gpu(sys_, Op, sub_policies);
                    value = gather(value);
                    info.state_grid = cellfun(@(x) gather(x), info.state_grid, 'UniformOutput', false);
                    info.time_policy_update = 0;
                    info.time_policy_eval = info.time_total;
                else
                    value = 0;
                    info.time_toal = 0;
                    info.time_policy_update = 0;
                    info.time_policy_eval = 0;
                end

%             	policies = cell(sys.U_DIMS, 1);
                for ss=1:1:size(sub_policies, 1)
                    sub_policies{ss, 3} = cellfun(@(x) gather(x), sub_policies{ss, 3}, 'UniformOutput', false);
                    sub_policies{ss, 4}.state_grid = cellfun(@(x) gather(x), sub_policies{ss, 4}.state_grid, 'UniformOutput', false);
                    sub_policies{ss, 4}.value = gather(sub_policies{ss, 4}.value);
                end
                policies = sub_policies;
            	
            	for uu=1:1:size(sub_policies, 1)
                	info.time_total = info.time_total + sub_policies{uu,4}.time_total;
                	info.time_total = info.time_policy_update + sub_policies{uu,4}.time_policy_update;
                    info.time_total = info.time_policy_eval + sub_policies{uu,4}.time_policy_eval;
                    
%                 	U_SUBDIM = sub_policies{uu,1};
%                 	X_SUBDIM = sub_policies{uu,2};
%                 	X_SUBDIM_BAR = 1:sys_.X_DIMS;
%                 	X_SUBDIM_BAR(X_SUBDIM) = [];
% 
%                 	subpolicy_size = Op.num_points;
%                 	subpolicy_size(X_SUBDIM_BAR) = 1;
%                 	subpolicy_newsize = Op.num_points;
%                 	subpolicy_newsize(X_SUBDIM) = 1;
% 
%                 	policies(U_SUBDIM) = cellfun(@(x) repmat(reshape(x, subpolicy_size), subpolicy_newsize), sub_policies{uu,3}, 'UniformOutput', false);
            	end
            	save(strcat(save_dir, '/final.mat'), 'sys', 'action_tree', 'policies', 'value', 'info', 'p', 's', '-v7.3', '-nocompression');
            	return;
            end
        end
        
        % Compute DP solutions for all leaf nodes and update the action tree
        for ii=1:1:size(leaf_nodes,1)
            U_DIMS_FREE = leaf_nodes{ii, 1};
            U_DIMS_CONTROLLED = [];
            X_DIMS_FREE = find(leaf_nodes{ii, 2})';
            
            sub_policies = leaf_nodes{ii, 3};
            NUM_SUBSYS = size(sub_policies, 1);
            for jj=1:1:NUM_SUBSYS
                U_DIMS_CONTROLLED = [U_DIMS_CONTROLLED; sub_policies{jj, 1}];
                X_DIMS_FREE = [X_DIMS_FREE; sub_policies{jj, 2}];
            end
            X_DIMS_FREE = sort(unique(X_DIMS_FREE));
            
            sys_ = sys;
            sys_.X_DIMS_FREE = X_DIMS_FREE;
            sys_.X_DIMS_FIXED = linspace(1, sys_.X_DIMS, sys_.X_DIMS)';
            sys_.X_DIMS_FIXED(X_DIMS_FREE) = [];
            sys_.U_DIMS_CONTROLLED = U_DIMS_CONTROLLED;
            sys_.U_DIMS_FREE = U_DIMS_FREE;
            sys_.U_DIMS_FIXED = linspace(1, sys_.U_DIMS, sys_.U_DIMS)';
            sys_.U_DIMS_FIXED([U_DIMS_CONTROLLED; U_DIMS_FREE]) = [];
            
            subsystem_id = strcat('U', sprintf('%d', U_DIMS_FREE), '_X', sprintf('%d', X_DIMS_FREE));
            if (isfield(Op, 'reuse_policy') && (Op.reuse_policy) && isfile(strcat(save_dir, '/', subsystem_id, '.mat')))
                decomposition = load(strcat(save_dir, '/', subsystem_id, '.mat'));
                policy = decomposition.policy;
                info = decomposition.info;
            else
                [policy, info] = get_dp_solution_gpu(sys_, Op, sub_policies);
                save(strcat(save_dir, '/', subsystem_id, '.mat'), 'policy', 'info', 'sys_', '-v7.3', '-nocompression');
                if (isfile(strcat(save_dir, '/', subsystem_id, '_interm.mat')))
                    delete(strcat(save_dir, '/', subsystem_id, '_interm.mat'));
                end
            end
            sub_policies = cat(1, sub_policies, ...
                               {U_DIMS_FREE, X_DIMS_FREE, policy, info});

            % Update the parent node
            parent_input = leaf_nodes{ii, 4};
            parent_node_id = find(cellfun(@(x) isempty(setdiff(x, parent_input)) ...
                                               && isempty(setdiff(parent_input, x)), ...
                                  action_tree(:, 1)));
            assert(length(parent_node_id)==1, 'Can only be one parent node ID');
            parent_node = action_tree(parent_node_id, :);
            assert(all(~(parent_node{2} .* leaf_nodes{ii, 2})), 'Parent-child cannot have state overlap');
            
            parent_node{1, 3} = cat(1, parent_node{1, 3}, sub_policies);
            
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
