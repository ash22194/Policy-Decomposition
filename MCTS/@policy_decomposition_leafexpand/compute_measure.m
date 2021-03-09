function measure = compute_measure(de)
    
    if (de.nodelist.isKey(de.decomposition_key))
        measure = de.nodelist(de.decomposition_key).measure;
        return;
    end
    
    sys_ = de.sys;
    action_tree_ = de.action_tree;
    action_tree_(:,5:6) = cell(size(action_tree_,1),2);
    for jj=1:1:size(action_tree_,1)
        action_tree_{jj, 5} = zeros(sys_.U_DIMS, 1);
        action_tree_{jj, 6} = zeros(sys_.U_DIMS, sys_.X_DIMS);
        if (action_tree_{jj,1}~=0)
            action_tree_{jj, 5}(action_tree_{jj,1}) = sys_.u0(action_tree_{jj,1}, 1);
        end
    end
    
    err_compute = 0;
    while (~isempty(action_tree_))
        % Find leaf nodes
        leaf_node_ids = cellfun(@(x) isempty(x), action_tree_(:,4));
        leaf_nodes = action_tree_(leaf_node_ids, :);

        if (size(leaf_nodes, 1)==1 && all(leaf_nodes{1,1}==0))
            % Compute LQR error measure
            K = leaf_nodes{1,6};
            try
                S = lyap((sys_.A - sys_.B*K - sys_.lambda_/2*eye(size(sys_.A,1)))', ...
                         K'*sys_.R*K + sys_.Q);
            catch ME
                S = -eye(sys_.X_DIMS);
            end

            if (any(eig(S) < 0))
                err_lqr = inf;
                err_compute = 1;
                
            else
                err_lqr = sys_.err_lqr_func(S, sys_.state_bounds(:,1)-sys_.l_point, sys_.state_bounds(:,2)-sys_.l_point)/sys_.da;
                
                assert(err_lqr >= -1e-13, 'LQR error measure cannot be negative');
                % Calculate compute fraction
                NS = sys_.num_points;
                NA = sys_.num_action_samples;
                M = sys_.max_iter;
                MP = sys_.max_policy_iter;
                interp_complexity = 2^length(NS); % Gridded Interpolation
                step_complexity = 4 * length(NS); % RK4 integration
                sample_complexity = 1; % randomly sample action
                action_update_complexity = 2;
                policy_eval_compute = prod(NS) * (MP * interp_complexity + step_complexity);
                policy_update_compute = prod(NS) * prod(NA) ...
                                        * (sample_complexity + step_complexity ...
                                           + interp_complexity + action_update_complexity);
                joint_compute = M * (policy_eval_compute + policy_update_compute);

                err_compute = err_compute / joint_compute;
            end
            de.lqr_measure = err_lqr;
            de.compute_fraction = err_compute;
            de.measure = sys_.measure_func(err_lqr, err_compute);
            measure = de.measure;
            
            return;
        end
        
        for ii=1:1:size(leaf_nodes,1)
            % Calculate LQR controller
            u0_ = leaf_nodes{ii, 5};
            K = leaf_nodes{ii, 6};
            fxfu = sys_.fxfu_func(sys_.l_point, u0_);
            
            A_ = fxfu(:,1:sys_.X_DIMS);
            B_ = fxfu(:,(sys_.X_DIMS+1):end);
            Q_ = sys_.Q;
            R_ = sys_.R;

            A_ = A_ - B_*K;
            A_ = A_(logical(leaf_nodes{ii, 2}), logical(leaf_nodes{ii, 2}));
            B_ = B_(logical(leaf_nodes{ii, 2}), leaf_nodes{ii, 1});
            Q_ = Q_ + K'*R_*K;
            Q_ = Q_(logical(leaf_nodes{ii, 2}), logical(leaf_nodes{ii, 2}));
            R_ = R_(leaf_nodes{ii, 1}, leaf_nodes{ii, 1});

            try
                [K_, ~, ~] = lqr(A_ - eye(size(A_, 1))*sys_.lambda_/2, B_, ...
                                 Q_, R_, zeros(size(A_, 1), size(B_, 2)));
            catch ME
                de.lqr_measure = inf;
                de.compute_fraction = 1;
                de.measure = sys_.measure_func(de.lqr_measure, de.compute_fraction);
                measure = de.measure;
                
                return;
            end
            K(leaf_nodes{ii, 1}, logical(leaf_nodes{ii, 2})) = K_;
            
            % Calculate compute complexity
            NS = sys_.num_points(logical(leaf_nodes{ii,2}));
            NA = sys_.num_action_samples(leaf_nodes{ii,1});
            M = sys_.max_iter;
            MP = sys_.max_policy_iter;
            interp_complexity = 2^length(NS); % Gridded Interpolation
            step_complexity = 4 * length(NS); % RK4 integration
            sample_complexity = 1; % randomly sample action
            action_update_complexity = 2;
            subpolicy_eval_compute = prod(NS) * (MP * interp_complexity + step_complexity);
            subpolicy_update_compute = prod(NS) * prod(NA) ...
                                       * (sample_complexity + step_complexity ...
                                          + interp_complexity + action_update_complexity);
                                      
            err_compute = err_compute + M * (subpolicy_eval_compute + subpolicy_update_compute);
            
            % Update parent
            parent_input = leaf_nodes{ii, 3};
            parent_node_id = find(cellfun(@(x) isempty(setdiff(x, parent_input)) && isempty(setdiff(parent_input, x)),...
                                  action_tree_(:, 1)));
            assert(length(parent_node_id)==1, 'Invalid Parent Node ID');
            parent_node = action_tree_(parent_node_id, :);

            assert(all(~(parent_node{2} .* leaf_nodes{ii, 2})), 'Invalid state overlap');
            parent_node{2} = parent_node{2} + leaf_nodes{ii, 2};
            parent_node{5} = parent_node{5} + leaf_nodes{ii, 5};
            parent_node{6} = parent_node{6} + K;

            % Find and delete the leaf_node in the list of children
            children_list = parent_node{4};
            childID = cellfun(@(x) fastsetequal(x, leaf_nodes{ii, 1}), children_list);
            children_list(childID) = [];
            parent_node{4} = children_list;

            action_tree_(parent_node_id, :) = parent_node;
        end
        action_tree_(leaf_node_ids, :) = [];
    end
end