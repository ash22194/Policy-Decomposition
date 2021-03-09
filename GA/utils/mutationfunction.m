function mutationChildren = mutationfunction(sys, parents, options, nvars, ...
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
            a1 = randi([1,sys.U_DIMS], 1);
            a2 = a1;
            while(a2==a1)
                a2 = randi([1,sys.U_DIMS], 1);
            end

            % Check if a1 or a2 are coupled
            a1coupled = all(s(a1,:) == s, 2);
            a2coupled = all(s(a2,:) == s, 2);

            s1 = s(a1coupled, :);
            s1 = s1(1,:);
            s2 = s(a2coupled, :);
            s2 = s2(1,:);

            s(a1coupled, :) = repmat(s2, sum(a1coupled), 1);
            s(a2coupled, :) = repmat(s1, sum(a2coupled), 1);
            
        else
            % 2) Change state variable(s) from one action to another
            s1 = randi([1,sys.X_DIMS], 1);
            s2 = s1;
            while(s2==s1)
                s2 = randi([1,sys.X_DIMS], 1);
            end
            
           a1 = logical(s(:,s1));
           a2 = logical(s(:,s2));
           s(a1, s1) = 0;
           s(a2, s2) = 0;
           s(a1, s2) = 1;
           s(a2, s1) = 1;
        end
        [c, c_eq] = constraints(p, s);
        if (any(c_eq~=0) || any(c > 0))
%             disp('Invalid state mutation');
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
            aroot = randi([1, sys.U_DIMS], 1);
            aroot_coupled = find(all(s(aroot,:) == s, 2));
            aroot_subtree = aroot_coupled;
            aroot_subtree_size = 0;
            loop_count = 0;
            while(aroot_subtree_size < length(aroot_subtree))
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
                new_parent_coupled = find(all(s(new_parent,:) == s, 2));
            else
                new_parent_coupled = new_parent;
            end
            
            new_parent_children = linspace(1, sys.U_DIMS, sys.U_DIMS);
            new_parent_children(p(any(p(:,1) == new_parent_coupled', 2), 2)) = [];
            if (~isempty(new_parent_children))
                p(aroot_coupled, 1) = new_parent;
                p(aroot_coupled, 2) = new_parent_children(1);
                if (any(p(:,1)==linspace(1,sys.U_DIMS,sys.U_DIMS)'))
                    disp('Loops!');
                end
                [c, c_eq] = constraints(p, s);
                if (any(c_eq~=0) || any(c > 0))
%                     disp('Invalid sub-tree mutation');
                else
                    current_child(1:2*sys.U_DIMS) = reshape(p, 1, sys.U_DIMS*2);
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
            a1coupled = all(s(a1,:) == s, 2);
            a2coupled = all(s(a2,:) == s, 2);
            
            if (all(a1coupled~=a2coupled))
                if ((sum(a1coupled) > 1) && (sum(s(a1,:)) > 1))
                    % Decouple a1
                    a1coupled = find(a1coupled);
                    decouplea1index = rand(1, length(a1coupled)) > 0.5;
                    while((sum(decouplea1index)==length(a1coupled)) ...
                          || (sum(decouplea1index)==0))
                        decouplea1index = rand(1, length(a1coupled)) > 0.5;
                    end
                    decouplea1 = a1coupled(decouplea1index);
                    a1coupled(decouplea1index) = [];

                    newchildID = linspace(1, sys.U_DIMS, sys.U_DIMS);
                    newchildID(p(any(p(:,1) == p(decouplea1, 1)', 2), 2)) = [];
                    p(decouplea1, 2) = newchildID(1);

                    a1states = find(s(a1, :));
                    decouplea1states = rand(1, length(a1states)) > 0.5;
                    while((sum(decouplea1states)==length(a1states)) ...
                          || (sum(decouplea1states)==0))
                        decouplea1states = rand(1, length(a1states)) > 0.5;
                    end
                    s(decouplea1, a1states(decouplea1states)) = 0;
                    s(a1coupled, a1states(~decouplea1states)) = 0;

                elseif ((sum(a1coupled) + sum(a2coupled)) < sys.U_DIMS)
                    % Couple a1 and a2
                    p(a2coupled,1) = p(a1,1);
                    p(a2coupled,2) = p(a1,2);
                    statescoupled = s(a1,:) || s(a2,:);
                    s(a1coupled || a2coupled, :) = statescoupled;
                    
                end
            end
            if (any(p(:,1)==linspace(1,sys.U_DIMS,sys.U_DIMS)'))
                disp('Loops!');
            end
            [c, c_eq] = constraints(p, s);
            if (any(c_eq~=0) || any(c > 0))
%                 disp('Invalid coupling-decoupling mutation');
            else
                current_child(1:2*sys.U_DIMS) = reshape(p, 1, sys.U_DIMS*2);
                current_child((2*sys.U_DIMS+1):end) = reshape(s, 1, sys.U_DIMS*sys.X_DIMS);
            end
        end
        action_mutation_children(ii, :) = current_child;
        
    end
    mutationChildren = [state_mutation_children; action_mutation_children];
end
