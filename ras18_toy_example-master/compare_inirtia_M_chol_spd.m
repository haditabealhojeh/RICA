%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code compares the results of learning and reproducing inirtia
% matrix profiles using GMM7GMR based Cholesky decomposition and GMM/GMR
% based on metrics of Riemannian manifold.
%
% Writing code takes time. Polishing it and making it available to others 
% takes longer! If some parts of the code were useful for your research of 
% for a better understanding of the algorithms, please reward the authors 
% by citing the related publications, and consider making your own research
% available in this way.
%
% @article{abudakka2018force,
% 	title={Force-based variable impedance learning for robotic 
%          manipulation},
%   author={Abu-Dakka, Fares J and Rozo, Leonel and Caldwell, Darwin G},
%   journal={Robotics and Autonomous Systems},
%   volume={109},
%   pages={156--167},
%   year={2018}
% }
%
% Copyright (c) 2017-18 Fares J. Abu-Dakka
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
close all;
clear all;
clc
for nstat = 2:4
    clear modelCHOL modelPD model_r tmpModel
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('GMM components G = %d \n',nstat);
    % GMM parameters on CHOLesky decomposition
    model_r.nbStates  = nstat;    % Number of Gaussians in the model
    model_r.dt        = 0.02; % Time step
    model_r.nbSamples = 5;    % Number of demonstrations
    reprosCHOL          = 6:6;  % reproduction
    % Regularization of covariance
    model_r.params_diagRegFact = 1E-2;
    
    % GMM parametes on SPD manifolds
    model_s.nbStates    = nstat; %Number of Gaussians
    model_s.nbVar       = 15;    % Dimension of the manifold/tangent space (1D input + 2^2 output)
    model_s.dt          = 2E-2; % Time step duration
    model_s.nbIter      = 10;   % Number of iteration for the Gauss Newton algorithm (Riemannian manifold)
    model_s.nbSamples   = 5;    % Number of demonstrations
    model_s.nbIterEM    = 10;   % Number of iteration for the EM algorithm
    reprosMan           = 6:6;  % reproduction
    % Regularization of covariance
    model_s.params_diagRegFact = 1E-2;
    %Dimension of the output covariance
    model_s.nbVarCovOut = model_s.nbVar + model_s.nbVar*(model_s.nbVar-1)/2;
    
    clrmapCHOL_states = lines(model_r.nbStates);
    clrmapMAN_states = lines(model_s.nbStates);
    clrmap_demos = lines(model_r.nbSamples);
    clrmap_summer = summer(max([length(reprosCHOL) length(reprosMan)]));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Load demonstrations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    path(path,'manifolds');
    load('data/DataM_X_01.mat');
    
    %% Learning: GMM on CHOLesky decomposition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t1 = clock;tic;
    model_r = init_GMM_kbins(DataM(:,:), model_r, model_r.nbSamples);%Initialization of GMM
    model_r = EM_GMM(DataM(:,:), model_r);%Training of a GMM with an expectation-maximization
    t2 = clock; time_Chol_GMM(nstat) = toc;
    etime_Chol_GMM(nstat) = etime(t2,t1);
    modelCHOL(nstat-1) = model_r;
    
    %% Reproduction: CHOLesky decomposition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    in     = 1:8;   % Input variables
    out    = 9:36;   % Output variables
    for j = reprosCHOL
    t1 = clock;tic;
        tmpModel(j) = GMR(model_r,[t(1:(nbData)); Y(:,:,j)'],in, out);
    t2 = clock; time_Chol_GMR(nstat) = toc;
    etime_Chol_GMR(nstat) = etime(t2,t1);
    end
    
    %% Learning: GMM on SPD Manifolds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialisation of the covariance transformation for GMM over man. ellips.
    [covOrder4to2, covOrder2to4] = set_covOrder4to2(model_s.nbVar);
    
    disp('Learning GMM2 (K ellipsoids)...');
    in     = 1:8;%1:3;               % Input variables
    out    = 9:model_s.nbVar;   % Output variables
    
    % Initialisation on the manifold
    clear model_r;
    t1 = clock;tic;
    model_s = spd_init_GMM_kbins(X, model_s, model_s.nbSamples); % Model for Man. Ellips.
    [model_s, GAMMA, L] = spd_EM_GMM(X, model_s, nbData,in,out);
    t2 = clock; time_Man_GMM(nstat) = toc;
    etime_Man_GMM(nstat) = etime(t2,t1);
    modelPD(nstat-1) = model_s;

    %% GMR (version with single optimization loop)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    xInSPD = zeros(8,8,nbData);
    for j = reprosMan
        for i=1:nbData
            xInSPD(1,1,i) = t(i);
            xInSPD(2,2,i) = Y(i,1,j);
            xInSPD(3,3,i) = Y(i,2,j);
            xInSPD(4,4,i) = Y(i,3,j);
            xInSPD(5,5,i) = Y(i,4,j);
            xInSPD(6,6,i) = Y(i,5,j);
            xInSPD(7,7,i) = Y(i,6,j);
            xInSPD(8,8,i) = Y(i,7,j);
        end
    t1 = clock;tic;
        newModelPD(j) = spd_GMR(modelPD(nstat-1), xInSPD, in, out);
    t2 = clock; time_Man_GMR(nstat) = toc;
    etime_Man_GMR(nstat) = etime(t2,t1);
    end
    clear model_s;
    %% Plotting resulting model and reproductions
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = reprosCHOL
        %M(:,:,:,j) = wam.inertia(Y(:,:,j));
        for i = 1:nbData
            Mc = zeros(7,7);
            Mc(1,1:7) = tmpModel(j).Mu(1:7,i);
            Mc(2,2:7) = tmpModel(j).Mu(8:13,i);
            Mc(3,3:7) = tmpModel(j).Mu(14:18,i);
            Mc(4,4:7) = tmpModel(j).Mu(19:22,i);
            Mc(5,5:7) = tmpModel(j).Mu(23:25,i);
            Mc(6,6:7) = tmpModel(j).Mu(26:27,i);
            Mc(7,7)   = tmpModel(j).Mu(28,i);
            
            Mn = Mc' * Mc;
            chol_errorCHOL(i,j) = cholesky_dist(M(:,:,i,j),Mn);
            Log_EuclideanCHOL(i,j) = Log_Euclidean_dist(M(:,:,i,j),Mn);
            Affine_InvariantCHOL(i,j) = Affine_invariant_dist(M(:,:,i,j),Mn);
            Log_DeterminantCHOL(i,j) = Log_Determinant_dist(M(:,:,i,j),Mn);
            
            Ms = newModelPD(j).Mu(:,:,i);
            chol_errorSPD(i,j) = cholesky_dist(M(:,:,i,j),Ms);
            Log_EuclideanSPD(i,j) = Log_Euclidean_dist(M(:,:,i,j),Ms);
            Affine_InvariantSPD(i,j) = Affine_invariant_dist(M(:,:,i,j),Ms);
            Log_DeterminantSPD(i,j) = Log_Determinant_dist(M(:,:,i,j),Ms);
        end
        chol_eCHOL(j) = mean(chol_errorCHOL(:,j).^2);
        logE_eCHOL(j) = mean(Log_EuclideanCHOL(:,j).^2);
        AffI_eCHOL(j) = mean(Affine_InvariantCHOL(:,j).^2);
        logD_eCHOL(j) = mean(Log_DeterminantCHOL(:,j).^2);
        
        chol_eSPD(j) = mean(chol_errorSPD(:,j).^2);
        logE_eSPD(j) = mean(Log_EuclideanSPD(:,j).^2);
        AffI_eSPD(j) = mean(Affine_InvariantSPD(:,j).^2);
        logD_eSPD(j) = mean(Log_DeterminantSPD(:,j).^2);
        
    end
    %Data to be plotted as the error bars
    if (length(reprosCHOL) > 1)
    model_error(:,:,nstat) = [ 
        std(std(Log_EuclideanSPD(:,reprosCHOL).^2))      std(std(Log_EuclideanCHOL(:,reprosCHOL).^2));
        std(std(Affine_InvariantSPD(:,reprosCHOL).^2))   std(std(Affine_InvariantCHOL(:,reprosCHOL).^2));
        std(std(Log_DeterminantSPD(:,reprosCHOL).^2))    std(std(Log_DeterminantCHOL(:,reprosCHOL).^2))];
    else
    model_error(:,:,nstat) = [ std(Log_EuclideanSPD(:,j).^2)      std(Log_EuclideanCHOL(:,j).^2);
                               std(Affine_InvariantSPD(:,j).^2)   std(Affine_InvariantCHOL(:,j).^2);
                               std(Log_DeterminantSPD(:,j).^2)    std(Log_DeterminantCHOL(:,j).^2)];
    end    
    colorss = lines(modelCHOL(nstat-1).nbStates);
    
    err(:,:,nstat)=[mean(logE_eSPD(reprosCHOL)) mean(logE_eCHOL(reprosCHOL));
                    mean(AffI_eSPD(reprosCHOL)) mean(AffI_eCHOL(reprosCHOL));
                    mean(logD_eSPD(reprosCHOL)) mean(logD_eCHOL(reprosCHOL))];
end
%% plot bar with barerror
comp_chol_spd_inertia = figure('position',[200,600,450,110]);
model_series = err;
mErr = [err(1,:,2); err(1,:,3); err(1,:,4)];
mStd = [model_error(1,:,2); model_error(1,:,3); model_error(1,:,4)];
% Creating axes and the bar graph
%handel_bar = figure('position',[2000,6000,450,110]);ax = axes;
%figure(comp_chol_spd_inertia);
%ax = subplot(1,2,[1 2]);
ax = axes;
h = bar(mErr,'BarWidth',1);
% Set color for each bar face
h(1).FaceColor = colorss(1,:);
h(2).FaceColor = colorss(2,:);
% Properties of the bar graph as required
ax.YGrid = 'on';
ax.GridLineStyle = '-';
xticks(ax,[1 2 3]);
% Naming each of the bar groups
%xticklabels(ax,{ 'Log Euclidean';'Affine invariant';'Log Determinant'});
xticklabels(ax,{ '2';'3';'4'});
% Y labels
ylabel ({'dist. error'}, 'Fontsize', 12, 'Interpreter', 'Latex');
xlabel('$\#$ of states  $G$', 'Fontsize', 12, 'Interpreter', 'Latex')
% Creating a legend and placing it outside the bar plot
%lg = legend('SPD','CHOL','AutoUpdate','off');set(lg,'FontSize',8);
lg = legend('GMR on $\bf{\mathcal{S}}_+^{m}$','GMR on Chol.','AutoUpdate','off');
set(lg, 'Fontsize', 12, 'Interpreter', 'Latex');
%lg.Location = 'BestOutside';
%lg.Orientation = 'Horizontal';
hold on;
% Finding the number of groups and the number of bars in each group
ngroups = size(mErr, 1);
nbars = size(mErr, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, mErr(:,i), mStd(:,i), 'k', 'linestyle', 'none');
end
%yaxis(0, max(max(mErr))+2*max(max(mStd)))
yaxis(0, .15);xaxis(.5, 3.5)
set(ax,'FontSize',12)
comp_chol_spd_inertia.CurrentAxes.TickLabelInterpreter = 'latex';
print(comp_chol_spd_inertia,'-depsc2','-r300',['fig_cp_spd_chol_metric_7D']);