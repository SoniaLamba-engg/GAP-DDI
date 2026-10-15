% Leak-free GAP-DDI evaluation: PPXA with per-fold embedding similarities.
% Run from "GAP-DDI_2 (PPXA + Prediction)" after make_fold_sims.py has created
% ../GAP-DDI_1(Feature Extraction)/fold_sims/<dataset>/
clear all
addpath(genpath('helper_functions'));
global pp mu1 mu2 lamda
rawdir  = '../GAP-DDI_1(Feature Extraction)/data_raw/';
simdir  = '../GAP-DDI_1(Feature Extraction)/fold_sims/';
datasets = {'Fdatasets','Cdatasets','MainDataset'};
nRep = 10; nFold = 10;
results = [];   % columns: dataset, repeat, fold, AUC, AUPR, runtime

for ds = 1:3
    name = datasets{ds};
    if ds == 3
        Y = readmatrix([rawdir name '/drug_dis.csv'])';
    else
        Y = load([rawdir name '/DiDrA.txt']);
    end
    getParameters('ppxa', min(ds,2));   % Main uses the same values as F/C (Table 3)
    for r = 1:nRep
        for i = 1:nFold
            p  = sprintf('%s%s/rep%d_fold%d_', simdir, name, r, i);
            St = load([p 'DrugSim.txt']);  Sd = load([p 'DiseaseSim.txt']);
            test_ind = load([p 'test_ind.txt']);
            y2 = Y; y2(test_ind) = 0;
            tic; y3 = ppxa(y2, Sd, St, test_ind); t = toc;
            ev = find(y2 == 0);
            auc  = calculate_auc (y3(ev), Y(ev), 0);
            aupr = calculate_aupr(y3(ev), Y(ev), 0);
            results = [results; ds r i auc aupr t];
            fprintf('%s rep %d fold %d  AUPR %.4f\n', name, r, i, aupr);
        end
    end
end
T = array2table(results, 'VariableNames', {'Dataset','Repeat','Fold','AUROC','AUPR','Runtime_s'});
T.Dataset = categorical(T.Dataset, 1:3, {'F','C','Main'});
writetable(T, 'fold_results_leakfree.csv');
disp(groupsummary(T, 'Dataset', {'mean','std'}, {'AUPR','AUROC'}));
