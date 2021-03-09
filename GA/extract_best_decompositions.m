clear;
clc;
close all;

%% Load
system_name = 'manipulator4dof';
MaxTime = 600;
num2extract = 5;
addpath('utils');
load(strcat('../data/', system_name, ...
            '_GA_NoCrossover_RandomSampled_explqrobj', num2str(MaxTime), ...
            '.mat'));

% GA 
population_ga = cell2mat(ga_decompositions(:,1));
population_ga = double(population_ga) - 48;
[~, u_id_ga] = unique(population_ga, 'rows', 'stable');
population_ga = population_ga(u_id_ga, :);

population_ga_measure = cell2mat(ga_decompositions(:,2));
population_ga_measure = population_ga_measure(u_id_ga,:);

[~, population_ga_sorted] = sort(population_ga_measure(:,1));
population_ga_sorted = population_ga_sorted(1:min(num2extract, ...
                                                  length(population_ga_sorted)));
population_ga = population_ga(population_ga_sorted, :);
population_ga_measure = population_ga_measure(population_ga_sorted,:);

u_xp_ga = zeros(length(population_ga_sorted), sys.U_DIMS*(2 + sys.X_DIMS));
[u_xp_ga(:, 1:(2*sys.U_DIMS)), ...
 u_xp_ga(:, (1+2*sys.U_DIMS):end)] = decode_bigbinary(sys, population_ga);

% Random

population_random = cell2mat(random_decompositions(:,1));
[~, u_id_random] = extract_decompositions_from_population(sys, population_random);
population_random = population_random(u_id_random, :);

population_random_measure = cell2mat(random_decompositions(:,2));
population_random_measure = population_random_measure(u_id_random, :);

[~, population_random_sorted] = sort(population_random_measure(:,1));
population_random_sorted = population_random_sorted(1:min(num2extract, ...
                                                  length(population_random_sorted)));

population_random = population_random(population_random_sorted, :);
population_random_measure = population_random_measure(population_random_sorted, :);

u_xp_random = population_random;

save(strcat('../data/', system_name, '_', ...
            num2str(num2extract),'candidates_GA_Random.mat'), 'u_xp_ga', 'u_xp_random');