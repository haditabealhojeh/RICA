%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code compares the computation time between Cholesky-Based and
% Riemannian-based GMM/GMR. We analyse the effect of the SPD dimensiones on
% the computational time.
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
dbstop if error
%% Problem parameters
repros       = 1;           % Reproduction
demons       = 1:1;         % Demonstrations to train the model
dt           = 1E-1;        % Time step
%dt = 2/50;


random = @(a,b,m,n)  a + (b-a).*rand(m,n);

nbData = 100; % Number of datapoints
nbSamples = 1; % Number of demonstrations
nbIter = 5; % Number of iteration for the Gauss Newton algorithm
nbIterEM = 1; % Number of iteration for the EM algorithm
nbDrawingSeg = 60; %Number of segments used to draw ellipsoids

elapsed_t = zeros(25,6);
elapsed_t_GMM_CHOL = zeros(25,50);
elapsed_t_GMR_CHOL = zeros(25,50);
elapsed_t_GMM_SPD = zeros(10,20);
elapsed_t_GMR_SPD = zeros(10,20);

for dims = 2:20
    %% GMM parameters on CHOLesky decomposition
    if exist('modelCHOL'),        clear modelCHOL;      end
    modelCHOL.nbStates  = 3;    % Number of Gaussians in the model
    modelCHOL.dt        = 0.02; % Time step
    modelCHOL.nbSamples = demons(end);    % Number of demonstrations
    reprosCHOL          = repros;  % reproduction
    % Regularization of covariance
    modelCHOL.params_diagRegFact = 1E-2;
    
    %% GMM parametes on SPD manifolds
    if exist('modelPD'),        clear modelPD;      end
    if exist('X'),              clear X;            end
    modelPD.nbStates  = 3;          % Number of Gaussians in SPD manifold model
    modelPD.nbVar     = dims+1;          % Dimension of the manifold and tangent
    modelPD.dt        = dt;         % Time step duration
    modelPD.nbIter    = 10;         % No. of iterations for the Gaussian Newton
    % algorithm (Riemannian manifold)
    modelPD.nbSamples = demons(end);% Number of demonstrations
    modelPD.nbIterEM  = 10;         % Number of iteration for the EM algorithm
    reprosMan         = repros;     % reproduction
    % Regularization of covariance
    modelPD.params_diagRegFact = 1E-2;
    % Dimension of the output covariance
    modelPD.nbVarCovOut = modelPD.nbVar + modelPD.nbVar*(modelPD.nbVar-1)/2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Generating input data
    disp('Generating data...');
    dt=0.02; dur=2; tt=linspace(0.02,dur,nbData);
    
    xIn(1,:) = 1:nbData;
    
    S1 = generateSPDmatrix(dims)*4;
    S2 = generateSPDmatrix(dims)*2;
    S3 = generateSPDmatrix(dims);
    
    s12 = logmap(S2,S1);
    s23 = logmap(S3,S2);
    
    t = 0:2/nbData:1-2/nbData;
    g12 = geodesic(s12,S1,t);
    g23 = geodesic(s23,S2,t);
    
    X = zeros(modelPD.nbVar, modelPD.nbVar, nbData*nbSamples);
    X(1,1,:) = reshape(repmat(xIn,1,nbSamples),1,1,nbData*nbSamples);
    X(2:end,2:end,:) = cat(3,g12,g23);
    for t = 1:nbData
        noise = .2*randn(dims,dims);
        X(2:end,2:end,t) = X(2:end,2:end,t) + (noise+noise');
    end
    
    for i = 1 : nbData
        [Mc,p] = chol(X(2:end,2:end,i));
        DataM(1,i) = tt(i);
        mask = triu(true(size(Mc)),0);
        Mv = Mc(mask);
        DataM(2:1+dims+dims*(dims-1)/2,i) = Mv;
    end
    
    for cnt = 1:50
        %% Learning: GMM on CHOLesky decomposition
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t1 = clock;tic;
        modelCHOL = init_GMM_kbins(DataM(:,:), modelCHOL, modelCHOL.nbSamples);%Initialization of GMM
        modelCHOL = EM_GMM(DataM(:,:), modelCHOL);
        t2 = clock;
        elapsed_t(dims,1) = elapsed_t(dims,1) + toc;
        elapsed_t_GMM_CHOL(dims,cnt) = etime(t2,t1);
        etime_Chol_GMM(dims) = etime(t2,t1);
        
        %% Reproduction: CHOLesky decomposition
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        in     = 1;                         % Input variables
        out    = 2:1+dims+dims*(dims-1)/2;  % Output variables
        
        tic;t1 = clock;
        for j = reprosCHOL
            tmpModel(j) = GMR(modelCHOL,tt(1:nbData),in, out);
        end
        
        t2 = clock;
        elapsed_t(dims,2) = elapsed_t(dims,2) + toc;
        elapsed_t_GMR_CHOL(dims,cnt) = etime(t2,t1);
    end
    if dims <= 20
        [covOrder4to2, covOrder2to4] = set_covOrder4to2(modelPD.nbVar);
        [covOrder4to2out, ~] = set_covOrder4to2(modelPD.nbVar-1);
        tensor_diagRegFact_mat = eye(modelPD.nbVar + modelPD.nbVar * (modelPD.nbVar-1)/2);
        tensor_diagRegFact = covOrder2to4(tensor_diagRegFact_mat);
        
        for cnt = 1:20
            printt = ['dimension = ' num2str(dims) ',    run no. = ' num2str(cnt)];
            disp(printt);
            
            %% Learning: GMM on SPD Manifolds
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Initialisation of the covariance transformation for
            % GMM over SPD ellips.
            
            in     = 1;                 % Input variables
            out    = 2:modelPD.nbVar;   % Output variables
            
            tic;t1 = clock;
            % GMM initialisation on the manifold
            modelPD = spd_init_GMM_kbins(X, modelPD, modelPD.nbSamples);
            [modelPD, GAMMA, L] = spd_EM_GMM(X, modelPD, nbData,in,out);
            t2 = clock;
            elapsed_t(dims,3) = elapsed_t(dims,3) + toc;
            elapsed_t_GMM_SPD(dims,cnt) = etime(t2,t1);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% GMR on SPD (version with single optimization loop)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            xIntest = linspace(1,100,100);
            xInSPD = zeros(1,1,length(xIntest));
            xInSPD(1,1,:) = reshape(repmat(xIntest,1,1),1,1,length(xIntest));%reshape(repmat(xIn,1,1),1,1,nbData);
            tic;t1 = clock;
            for j = reprosMan
                newModelPD(j) = spd_GMR(modelPD, xInSPD, in, out);
            end
            t2 = clock;
            elapsed_t(dims,4) = elapsed_t(dims,4) + toc;
            elapsed_t_GMR_SPD(dims,cnt) = etime(t2,t1);
        end

    end
end
%save('data/elapsed_time10.mat', 'elapsed_t_GMR_CHOL', 'elapsed_t_GMR_SPD'); 
elapsed_time=figure;
x = [2:20];
yCHOL = mean(elapsed_t_GMR_CHOL(2:20,:)'/100);
errCHOL = std(elapsed_t_GMR_CHOL(2:20,:)'/100);
ySPD = mean(elapsed_t_GMR_SPD(2:end,:)'/100);
errSPD = std(elapsed_t_GMR_SPD(2:end,:)'/100);

subplot(211);
hold on
errorbar(x,yCHOL,errCHOL,'-s','MarkerSize',3,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')
axis([1,21,1e-5,17e-4]);box on
set(gca, 'FontSize', 8);
elapsed_time.CurrentAxes.TickLabelInterpreter = 'latex';
ylabel('avg. comp. time','Interpreter','latex','FontSize',10);
xlabel('Space dimension','Interpreter','latex','FontSize',10);
subplot(212);
hold on
errorbar([2:1:length(ySPD)+1],ySPD,errSPD,'-s','MarkerSize',3,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')
axis([1,21,-.1,4.2]);box on
set(gca, 'FontSize', 8);
elapsed_time.CurrentAxes.TickLabelInterpreter = 'latex';
ylabel('avg. comp. time','Interpreter','latex','FontSize',10);
xlabel('Space dimension','Interpreter','latex','FontSize',10);

% print(elapsed_time,'-depsc2','-r600',['elapsed_time']);
% print(elapsed_time,'-dpng','-r600',['elapsed_time']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot computational time
