function [root_node, info] = run_mcts(root_node, max_iter, max_time, determ)
    
    root_node.compute_measure();
    
    % Expand nodes till the time runs out or the tree is fully explored
    ii = 0;
    curr_best_measure = 1;
    tic;
    while ((ii < max_iter) && (toc < max_time) && (root_node.subtree_unexplored))
        ii = ii + 1;
        curr_node = root_node;
        while(isa(curr_node, class(root_node)))
            try
                curr_node = curr_node.expand_best_child(determ);
            catch ME
                disp('Something is wrong!');
            end
        end
        
        if (root_node.childnode_best.measure < curr_best_measure)
            time_to_find_best = toc;
            num_nodes_explored = sum(cellfun(@(x) isa(x, class(root_node)), root_node.nodelist.values));
            curr_best_measure = root_node.childnode_best.measure;
            fprintf('Iter : %d, Nodes : %d, Best so far : %d\n', ...
                    ii, num_nodes_explored, curr_best_measure);
        end
    end
    
	num_nodes_explored = sum(cellfun(@(x) isa(x, class(root_node)), root_node.nodelist.values));
    fprintf('Best : %d, Time for Best : %d s, Number of nodes : %d\n', curr_best_measure, time_to_find_best, num_nodes_explored);
    
    info.best_measure = curr_best_measure;
    info.time_to_find_best = time_to_find_best;
    info.num_nodes_explored = num_nodes_explored;
end