function mutationChildren = mutationfunctionstatemasked(sys, parents, options, nvars, ...
                                  FitnessFcn, state, thisScore, thisPopulation)
    
    % Randomly pick candidates for state manipulation
    set_state_manip = rand(length(parents), 1) > 0.5;
    state_mutation_children = round(thisPopulation(parents(set_state_manip)', :));
    
    for ii=1:1:size(state_mutation_children)
        current_child = state_mutation_children(ii, :);
        p = reshape(current_child(1:2*sys.U_DIMS), sys.U_DIMS, 2);
        s = reshape(current_child((2*sys.U_DIMS + 1):end), sys.U_DIMS, sys.X_DIMS);
        if (rand() > 0.5)
            % 1) Swap state allocation for two actions
            action_matrix = zeros(sys.U_DIMS, sys.U_DIMS);
            while(~all(action_matrix, 'all'))
                a1 = randi([1,sys.U_DIMS], 1);
                a1 = find(all(p(a1,:) == p, 2));
                is_a1_leaf = ~any(a1==p(:,1)', 'all');

                a2 = a1(1);
                while(any(a2==a1))
                    a2 = randi([1,sys.U_DIMS], 1);
                end
                a2 = find(all(p(a2,:) == p, 2));
                is_a2_leaf = ~any(a2==p(:,1)', 'all');
                
                action_matrix(a1,a1) = 1;
                action_matrix(a1,a2) = 1;
                action_matrix(a2,a2) = 1;
                action_matrix(a2,a1) = 1;
                
                s1 = s(a1(1), :);
                s2 = s(a2(1), :);
                
                if (~((is_a1_leaf && (sum(s2)<1)) || (is_a2_leaf && (sum(s1)<1))))
                    s(a1, :) = repmat(s2, length(a1), 1);
                    s(a2, :) = repmat(s1, length(a2), 1);
                    break;
                end
            end
        else
            % 2) Change state variable(s) from one action to another
            state_masked = linspace(1, size(sys.X_DIMS_MASKED, 1), size(sys.X_DIMS_MASKED, 1));
            while(~isempty(state_masked))
                s1 = randi([1, length(state_masked)], 1);
                s1_ = state_masked(s1);
                state_masked(s1) = [];
                
                s1 = logical(sys.X_DIMS_MASKED(s1_,:));
                a1 = all(s(:, s1), 2);
                a1 = find(a1);
                sa1 = logical(s(a1(1),:));
                
                is_a1_leaf = ~any(a1 == p(:,1)', 'all');
                is_a1_empty_leaf = is_a1_leaf && (sum(sa1 - s1) < 1);
                if (~is_a1_empty_leaf)
                    break;
                end
            end
            
            if (~is_a1_empty_leaf)
                a2 = randi([1, sys.U_DIMS], 1);
                while (any(a1==a2))
                    a2 = randi([1, sys.U_DIMS], 1);
                end
                a2 = all(p(a2, :)==p, 2);
                s(a2, s1) = 1;
                s(a1, s1) = 0;
            end
            
        end
        [c, c_eq] = constraints(p, s);
        if (any(c_eq~=0) || any(c > 0))
            disp('Invalid state mutation');
        else
            current_child((2*sys.U_DIMS + 1):end) = reshape(s, 1, sys.U_DIMS*sys.X_DIMS);
            state_mutation_children(ii, :) = current_child;
        end
    end
    
    % Remaining cadidates for action tree manipulation
    set_action_manip = ~set_state_manip;
    action_mutation_children = thisPopulation(parents(set_action_manip)', :);
    for ii=1:1:size(action_mutation_children, 1)
        current_child = action_mutation_children(ii, :);
        p = reshape(current_child(1:2*sys.U_DIMS), sys.U_DIMS, 2);
        s = reshape(current_child((2*sys.U_DIMS + 1):end), sys.U_DIMS, sys.X_DIMS);
        
        if (rand() < 0.5)
            % 1) Move subtree
            actions = linspace(1, sys.U_DIMS, sys.U_DIMS);
            while (~isempty(actions))
                aroot_id = randi([1, length(actions)], 1);
                aroot = actions(aroot_id);
                aroot_coupled = find(all(p(aroot,:) == p, 2));
                
                actions(any(aroot_coupled==actions, 1)) = [];
                root_parent = p(aroot, 1);
                num_root_parent_states = 0;
                if (root_parent~=0)
                    root_parent = find(all(p(root_parent,:) == p, 2));
                    num_root_parent_states = sum(s(root_parent(1),:));
                end
                
                num_root_parent_children = sum(any(root_parent == p(:,1)', 1));
                if (~(((num_root_parent_children - length(aroot_coupled))<1) ...
                      && (num_root_parent_states < 1)))

                    aroot_subtree = aroot_coupled;
                    aroot_subtree_size = 0;
                    loop_count = 0;
                    while (aroot_subtree_size < length(aroot_subtree))
                        loop_count = loop_count + 1;
                        aroot_subtree_size = length(aroot_subtree);
                        aroot_subtree = sort(unique([aroot_subtree; ...
                                                     find(any(p(:,1)==aroot_subtree', 2))]));
                        assert(loop_count<=sys.U_DIMS, 'Mutation error!');
                    end
                    new_parents = linspace(0, sys.U_DIMS, sys.U_DIMS+1);
                    new_parents(aroot_subtree+1) = [];
                    new_parent = new_parents(randi([1, length(new_parents)], 1));
                    if (new_parent~=0)
                        new_parent_coupled = find(all(p(new_parent,:) == p, 2));
                    else
                        new_parent_coupled = new_parent;
                    end

                    new_parent_children = linspace(1, sys.U_DIMS, sys.U_DIMS);
                    try
                        new_parent_children(p(any(p(:,1) == new_parent_coupled', 2), 2)) = [];
                    catch
                        disp('check!');
                    end
                    if (~isempty(new_parent_children))
                        p(aroot_coupled, 1) = new_parent;
                        p(aroot_coupled, 2) = new_parent_children(1);
                        if (any(p(:,1)==linspace(1,sys.U_DIMS,sys.U_DIMS)'))
                            disp('Loops!');
                        end
                        [c, c_eq] = constraints(p, s);
                        if (any(c_eq~=0) || any(c > 0))
                            disp('Invalid sub-tree mutation');
                        else
                            current_child(1:2*sys.U_DIMS) = reshape(p, 1, sys.U_DIMS*2);
                        end
                    end
                    break;
                end
            end
        else
            % 2) Couple decoupled actions
            % 3) Decouple coupled actions
            a1 = randi([1, sys.U_DIMS], 1);
            a2 = a1;
            while(a2==a1)
                a2 = randi([1,sys.U_DIMS], 1);
            end
            
            % Check if a1 or a2 are coupled
            a1coupled = all(p(a1,:) == p, 2);
            a2coupled = all(p(a2,:) == p, 2);
            
            if (all(a1coupled==a2coupled))
                % a1 and a2 are coupled, decouple a1
                a1coupled = find(a1coupled);
                a1childnodes = find(any(a1coupled==p(:,1)', 1));
                decouplea1index = rand(1, length(a1coupled)) > 0.5;
                while ((sum(decouplea1index)==length(a1coupled)) ...
                       || (sum(decouplea1index)==0))
                    decouplea1index = rand(1, length(a1coupled)) > 0.5;
                end
                decouplea1 = a1coupled(decouplea1index);
                a1coupled(decouplea1index) = [];
                newchildID = linspace(1, sys.U_DIMS, sys.U_DIMS);
                newchildID(p(any(p(:,1) == p(decouplea1, 1)', 2), 2)) = [];
                p(decouplea1, 2) = newchildID(1);
                
                % redistribute childnodes
                num_children_a1 = 0;
                num_children_decouplea1 = 0;
                while (~isempty(a1childnodes))
                    a1child = a1childnodes(1);
                    a1childcoupled = find(all(s(a1child,:)==s, 2));
                    if (rand() > 0.5)
                        num_children_a1 = num_children_a1 + 1;
                        p(a1childcoupled, 1) = min(a1coupled);
                        p(a1childcoupled, 2) = num_children_a1;
                    else
                        num_children_decouplea1 = num_children_decouplea1 + 1;
                        p(a1childcoupled, 1) = min(decouplea1);
                        p(a1childcoupled, 2) = num_children_decouplea1;
                    end
                    a1childnodes(any(a1childcoupled == a1childnodes, 1)) = [];
                end
                
                % redistribute state variables
                a1states = find(all((s(a1,:) - sys.X_DIMS_MASKED) >= 0, 2));
                decouplea1states = rand(1, length(a1states)) > 0.5;
                
                s(a1coupled, any(sys.X_DIMS_MASKED(a1states(decouplea1states), :), 1)) = 0;
                s(decouplea1, any(sys.X_DIMS_MASKED(a1states(~decouplea1states), :), 1)) = 0;
                
                if (~(((num_children_a1==0) && ((length(a1states) - sum(decouplea1states)) == 0)) ...
                    || ((num_children_decouplea1==0) && (sum(decouplea1states)==0))))
                    
                    [c, c_eq] = constraints(p, s);
                    if (any(c_eq~=0) || any(c > 0))
                        disp('Invalid decoupling mutation');
                    else
                        current_child(1:2*sys.U_DIMS) = reshape(p, 1, sys.U_DIMS*2);
                        current_child((2*sys.U_DIMS+1):end) = reshape(s, 1, sys.U_DIMS*sys.X_DIMS);
                    end
                end
                
            elseif (sum(a1coupled + a2coupled) < sys.U_DIMS)
                % a1 and a2 are decoupled, merge a2 into a1
                a1parent = p(a1, 1);
                a1coupled = find(a1coupled);
                a2coupled = find(a2coupled);
                while (a1parent~=0)
                    if (any(a1parent==a2coupled))
                        temp = a1;
                        a1 = a2;
                        a2 = temp;
                        
                        temp = a1coupled;
                        a1coupled = a2coupled;
                        a2coupled = temp;
                        break;
                    end
                    a1parent = p(a1parent, 1);
                end
                
                a2parent = p(a2, 1);
                num_a2parent_states = 0;
                if (a2parent~=0)
                    num_a2parent_states = sum(s(a2parent,:));
                    a2parent = find(all(p(a2parent, :) == p, 2));
                end
                num_a2parent_children = sum(any(a2parent == p(:,1)', 1));
                
                if (~(((num_a2parent_children - length(a2coupled)) < 1) ...
                    && (num_a2parent_states < 1)))
                    p(a2coupled,1) = p(a1,1);
                    p(a2coupled,2) = p(a1,2);
                    
                    a1children_parent = a1coupled(any(a1coupled == p(:,1)', 2));
                    a2children_parent = a2coupled(any(a2coupled == p(:,1)', 2));
                    
                    assert(length(a1children_parent) <= 1, 'a1children - Parent must be the same!');
                    assert(length(a2children_parent) <= 1, 'a2children - Parent must be the same!');
                    
                    if (~isempty(a2children_parent))
                        a2children = (p(:,1) == a2children_parent);
                        if (~isempty(a1children_parent))
                            p(a2children, 1) = a1children_parent;
                            a1children = (p(:,1) == a1children_parent);
                            p(a2children, 2) = p(a2children, 2) + max(p(a1children, 2));
                        end
                    end
                    
                    statescoupled = s(a1,:) | s(a2,:);
                    s(a1coupled, :) = repmat(statescoupled, length(a1coupled), 1);
                    s(a2coupled, :) = repmat(statescoupled, length(a2coupled), 1);
                end
                [c, c_eq] = constraints(p, s);
                if (any(c_eq~=0) || any(c > 0))
                    disp('Invalid coupling mutation');
                else
                    current_child(1:2*sys.U_DIMS) = reshape(p, 1, sys.U_DIMS*2);
                    current_child((2*sys.U_DIMS+1):end) = reshape(s, 1, sys.U_DIMS*sys.X_DIMS);
                end
            end
            
            if (any(p(:,1)==linspace(1,sys.U_DIMS,sys.U_DIMS)'))
                disp('Loops!');
            end
            
        end
        action_mutation_children(ii, :) = current_child;
    end
    mutationChildren = [state_mutation_children; action_mutation_children];
end
