clear;
close all;
clc;

%% 

restoredefaultpath();
system_name = 'biped2d';
addpath('utils');
addpath(strcat('../iLQG_boxQP/systems/', system_name));
load(strcat('data/', system_name, 'System.mat'));

% Add function to compute linearized dynamics
if (~isfield(sys, 'fxfu_func'))
    sys.fxfu_func = @(x, u) [dynx(sys, x, u), dynu(sys, x, u)];
    fxfu = sys.fxfu_func(sys.l_point, sys.u0);
    sys.A = fxfu(:, 1:sys.X_DIMS);
    sys.B = fxfu(:, (1+sys.X_DIMS):end);
end

% Add function to compute LQR metric
if (~isfield(sys, 'err_lqr_func'))
    sys.lambda_ = (1 - sys.gamma_) / sys.dt;
    [~, S_j, ~] = lqr(sys.A - (sys.lambda_ / 2) * eye(size(sys.A,1)), sys.B, sys.Q, sys.R, zeros(size(sys.B)));
    sys.S = sym('S', [sys.X_DIMS, sys.X_DIMS]);
    sys.S = tril(sys.S,0) + tril(sys.S,-1).';
    sys.a = sym('a', [sys.X_DIMS, 1]);
    sys.b = sym('b', [sys.X_DIMS, 1]);
    err_lqr = sys.x.'*sys.S*sys.x - sys.x.'*S_j*sys.x;
    for ii=1:1:sys.X_DIMS
        err_lqr = int(err_lqr, sys.x(ii), [sys.a(ii), sys.b(ii)]);
    end
    sys.err_lqr_func = matlabFunction(err_lqr, 'Vars', {sys.S, sys.a, sys.b});
    sys.da = prod(sys.state_bounds(:,2) - sys.state_bounds(:,1), 1);
end

sys.measure_func = @(err_lqr, err_compute) (1 - exp(-err_lqr)) .* err_compute;
% sys.measure_func = @(err_lqr, err_compute) (min(20.0, err_lqr) / 20.0) .* err_compute;

nvars = 2*sys.U_DIMS + sys.X_DIMS*sys.U_DIMS;
lb = [zeros(1, sys.U_DIMS), ones(1, sys.U_DIMS), zeros(1, sys.U_DIMS*sys.X_DIMS)];
ub = [sys.U_DIMS*ones(1, sys.U_DIMS*2), ones(1, sys.U_DIMS*sys.X_DIMS)];
A = [];
b = [];
Aeq = [];
beq = [];
nonlcon = @(x) constraints(reshape(x(1:2*sys.U_DIMS), sys.U_DIMS, 2), ...
                           reshape(x((2*sys.U_DIMS + 1):(2*sys.U_DIMS + sys.U_DIMS*sys.X_DIMS)), sys.U_DIMS, sys.X_DIMS));
IntCon = [];
options.Display = 'iter';
options.PopulationSize = 200;
options.CrossoverFraction = 0.0;
options.EliteCount = 0.2*options.PopulationSize;

options.CreationFcn = @(nvars, fitness_fcn, options) ...
                        generate_population(sys, options.PopulationSize);
options.CrossoverFcn = @(parents, options, nvars, fitness_fcn, unused, population) ...
                         crossoverfunction(sys, parents, options, nvars, fitness_fcn, unused, population);
options.MutationFcn = @(parents, options, nvars, fitness_fcn, state, score, population) ...
                         mutationfunction(sys, parents, options, nvars, fitness_fcn, state, score, population);
MaxGATime = 600;
MaxTotalTime = 600;
num_runs = 1;
ga_stats = zeros(num_runs, 7);
ga_decompositions = cell(num_runs, 2);

for rr=1:1:num_runs
    
    sys.decompositionlist = containers.Map();
    fun = @(x) computeJointMeasure(sys, reshape(x(1:2*sys.U_DIMS), sys.U_DIMS, 2), ...
                                   reshape(x((2*sys.U_DIMS + 1):(2*sys.U_DIMS + sys.U_DIMS*sys.X_DIMS)), sys.U_DIMS, sys.X_DIMS));
    ga_solutions = cell(0, 6); % x, err_joint, exitflag, output, population, scores
    tic;
    while((MaxTotalTime - toc) > 0)
        options.InitialPopulation = generate_population(sys, options.PopulationSize);
        options.MaxTime = min(MaxGATime, MaxTotalTime - toc);
        [ga_solutions{(end+1), :}] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,IntCon,options);
    end

    % Extract decomposition
    num_to_extract = 10;
    final_population = cell2mat(ga_solutions(:, end-1));
    final_scores     = cell2mat(ga_solutions(:, end));

    [~, unique_population_id] = extract_decompositions_from_population(sys, final_population);
    final_population = final_population(unique_population_id, :);
    final_scores     = final_scores(unique_population_id, :);
    [~, final_scores_order] = sort(final_scores);

    final_population = final_population(final_scores_order(1:min(num_to_extract, size(final_population, 1))), :);
    final_scores     = final_scores(final_scores_order(1:min(num_to_extract, size(final_scores, 1))), :);

    [cumm_population_encoding, cumm_population_ids] = unique(cell2mat(sys.decompositionlist.keys'), ...
                                                             'rows', 'stable');
    cumm_population_measure = cell2mat(sys.decompositionlist.values');
    cumm_population_measure = cumm_population_measure(cumm_population_ids,:);

    [~, best_population_ids] = sort(cumm_population_measure(:,1));
    best_population_ids = best_population_ids(1:num_to_extract);
    best_population_measure = cumm_population_measure(best_population_ids, :);
    best_population = cumm_population_encoding(best_population_ids, :);

    best_measure = min(best_population_measure(:,1));
    mean_measure = mean(best_population_measure(:,1));
    std_measure = std(best_population_measure(:,1));

    best_lqr_measure = min(best_population_measure(:,2));
    mean_lqr_measure = mean(best_population_measure(:,2));
    std_lqr_measure = std(best_population_measure(:,2));
    
    ga_stats(rr, 1) = best_measure;
    ga_stats(rr, 2) = mean_measure;
    ga_stats(rr, 3) = std_measure;
    
    ga_stats(rr, 4) = best_lqr_measure;
    ga_stats(rr, 5) = mean_lqr_measure;
    ga_stats(rr, 6) = std_lqr_measure;
    
    ga_stats(rr, 7) = size(sys.decompositionlist, 1);
    
    ga_decompositions{rr, 1} = best_population;
    ga_decompositions{rr, 2} = best_population_measure;
    
    disp('GA');
    disp(strcat('Best Measure : ', num2str(best_measure), ...
                ', Mean Measure : ', num2str(mean_measure), ...
                ', Std Measure : ', num2str(std_measure)));
    disp(strcat('Best LQR Measure : ', num2str(best_lqr_measure), ...
                ', Mean LQR Measure : ', num2str(mean_lqr_measure), ...
                ', Std LQR Measure : ', num2str(std_lqr_measure)));
    disp(strcat('', 'Number of nodes : ', num2str(size(sys.decompositionlist, 1))));
    
end

%% Random Sampling

random_stats = zeros(num_runs, 7);
random_decompositions = cell(num_runs, 3);
for rr=1:1:num_runs
    sys.decompositionlist = containers.Map();
    sampled_population = cell(0, 2);
    tic;
    while ((MaxTotalTime - toc) > 0)

        new_population = generate_population(sys, num_to_extract);
        new_population = num2cell(new_population, 2);
        new_population_measure = cellfun(@(x) computeJointMeasure(sys, ...
                                                                  reshape(x(1:(2*sys.U_DIMS)), sys.U_DIMS, 2), ...
                                                                  reshape(x((1+2*sys.U_DIMS):end), sys.U_DIMS, sys.X_DIMS)), ...
                                         new_population, 'UniformOutput', false);
        sampled_population = cat(1, sampled_population, cat(2, new_population, new_population_measure));
        if (size(sampled_population, 1) > 1e4)
            [~, unique_sampled_population_id] = extract_decompositions_from_population(sys, cell2mat(sampled_population(:,1)));
            sampled_population = sampled_population(unique_sampled_population_id, :);
        end
    end

    [~, unique_sampled_population_id] = extract_decompositions_from_population(sys, cell2mat(sampled_population(:,1)));
    sampled_population_measure = cell2mat(sampled_population(unique_sampled_population_id, 2));
    sampled_population = cell2mat(sampled_population(unique_sampled_population_id, 1));

    [sampled_population_measure, sampled_population_order] = sort(sampled_population_measure);
    sampled_population = sampled_population(sampled_population_order(1:min(num_to_extract, size(sampled_population, 1))), :);
    sampled_population_measure = sampled_population_measure(1:size(sampled_population, 1), :);

    sampled_population_err_lqr = zeros(num_to_extract, 1);
    for dd=1:1:num_to_extract
        sampled_population_err_lqr(dd) = computeLQRMeasure(sys, ...
                                                           reshape(sampled_population(dd, 1:(2*sys.U_DIMS)), sys.U_DIMS, 2), ...
                                                           reshape(sampled_population(dd, (1+(2*sys.U_DIMS)):end), sys.U_DIMS, sys.X_DIMS));
    end
    
    random_stats(rr, 1) = min(sampled_population_measure);
    random_stats(rr, 2) = mean(sampled_population_measure);
    random_stats(rr, 3) = std(sampled_population_measure);
    
    random_stats(rr, 4) = min(sampled_population_err_lqr);
    random_stats(rr, 5) = mean(sampled_population_err_lqr);
    random_stats(rr, 6) = std(sampled_population_err_lqr);
    
    random_stats(rr, 7) = size(sys.decompositionlist, 1);
    
    random_decompositions{rr, 1} = sampled_population;
    random_decompositions{rr, 2} = sampled_population_measure;
    random_decompositions{rr, 3} = sampled_population_err_lqr;
    
    disp('Random');
    disp(strcat('Best Measure : ', num2str(min(sampled_population_measure)), ...
                ', Mean Measure : ', num2str(mean(sampled_population_measure)), ...
                ', Std Measure : ', num2str(std(sampled_population_measure))));
    disp(strcat('Best LQR Measure : ', num2str(min(sampled_population_err_lqr)), ...
                ', Mean LQR Measure : ', num2str(mean(sampled_population_err_lqr)), ...
                ', Std LQR Measure : ', num2str(std(sampled_population_err_lqr))));
    disp(strcat('', 'Number of nodes : ', num2str(size(sys.decompositionlist, 1))));
end
sys.decompositionlist = [];

save(strcat('data/', system_name,'_GA_NoCrossover_RandomSampled_explqrobj',num2str(MaxTotalTime),'.mat'), ...
     'sys', 'ga_stats', 'ga_decompositions', 'random_stats', 'random_decompositions', 'MaxTotalTime', 'MaxGATime', 'num_to_extract');

%% Functions

function population = generate_population(sys, n)
    
    % Decide input coupling
    invalid = true(1,n);
    while (sum(invalid) > 0)
       r(:, invalid) = randi([1,sys.U_DIMS], sys.U_DIMS, sum(invalid));
       invalid = vecnorm(r - mean(r, 1)) < 1e-4;
    end
    p = zeros(sys.U_DIMS, 2, n);
    s = zeros(sys.U_DIMS, sys.X_DIMS, n);
    
    for ii=1:1:n
        rC = unique(r(:,ii));
        pseudo_inputs = cell(length(rC)+1, 1);
        pseudo_inputs{end, 1} = [0];
        for jj=1:1:length(rC)
            pseudo_inputs{jj} = find(r(:,ii)==rC(jj));
        end
        
        % Use random Prufer codes to generate a tree
        prufer_seq = randi([1, length(rC) + 1], length(rC)-1, 1);
        leaf_nodes = ~any(prufer_seq == linspace(1, length(rC) + 1, length(rC) + 1), 1);
        leaf_nodes = leaf_nodes(1:length(rC));
        
        node_list = linspace(1, length(rC) + 1, length(rC) + 1)';
        child_count = zeros(length(rC) + 1, 1);
        while(length(node_list) > 2)
            child = min(setdiff(node_list, prufer_seq));
            parent = prufer_seq(1);
            child_count(parent) = child_count(parent) + 1;
            
            node_list(node_list == child) = [];
            prufer_seq = prufer_seq(2:end);
            
            p(pseudo_inputs{child}, 1, ii) = min(pseudo_inputs{parent});
            p(pseudo_inputs{child}, 2, ii) = child_count(parent);
        end
        child = node_list(1);
        parent = node_list(2);
        child_count(parent) = child_count(parent) + 1;
        p(pseudo_inputs{child}, 1, ii) = pseudo_inputs{parent}(1);
        p(pseudo_inputs{child}, 2, ii) = child_count(parent);
        
        % Random state assignments for the inputs
        state = linspace(1, sys.X_DIMS, sys.X_DIMS);
        for jj=1:1:length(rC)
            is_leaf_node = leaf_nodes(jj);
            num_leaf_nodes_left = sum(leaf_nodes(jj:end));
            num_leaf_nodes_left = num_leaf_nodes_left - double(is_leaf_node);
            if (~is_leaf_node)
                k = randi([0, length(state)-num_leaf_nodes_left], 1);
            else
                k = randi([1, length(state)-num_leaf_nodes_left], 1);
            end
            
            if (k>0)
                sub_state = nchoosek(linspace(1,length(state),length(state)), k);
                sub_state = sub_state(randi([1, size(sub_state,1)],1),:);
                sub_state_ = state(sub_state);
                state(sub_state) = [];

                s(r(:,ii)==rC(jj), sub_state_, ii) = 1;
            end
        end
        s(r(:,ii)==rC(jj), state, ii) = 1;
        
        if (any(p(:,1,ii) == linspace(1, sys.U_DIMS, sys.U_DIMS)'))
            disp('Loops!');
        end
        [c, c_eq] = constraints(p(:,:,ii), s(:,:,ii));
        if (any(c_eq~=0) || any(c > 0))
            disp('Invalid sample');
        end
    end
    
    population = [reshape(p, 2*sys.U_DIMS, n)', reshape(s, sys.U_DIMS*sys.X_DIMS, n)'];

end
