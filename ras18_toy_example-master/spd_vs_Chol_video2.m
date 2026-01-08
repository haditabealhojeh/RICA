%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code show a comparison between GMR reproduction based-Cholesky vs 
% based-Riemannian metric, and show the result in the cone space of SPD.
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
path(path,'manifolds');
%% Problem parameters
repros       = 1;           % Reproduction
demons       = 1:1;         % Demonstrations to train the model
dt           = 1E-1;        % Time step
%dt = 2/50;
% GMM parameters on CHOLesky decomposition
modelCHOL.nbStates  = 1;            % Number of Gaussians in CHOL model
modelCHOL.dt        = dt;           % Time step
modelCHOL.nbSamples = demons(end);  % Number of demonstrations
reprosCHOL          = repros;       % reproduction
% Regularization of covariance
modelCHOL.params_diagRegFact = 1E-4;

% GMM parametes on SPD manifolds
modelPD.nbStates  = 1;          % Number of Gaussians in SPD manifold model
modelPD.nbVar     = 3;          % Dimension of the manifold and tangent
modelPD.dt        = dt;         % Time step duration
modelPD.nbIter    = 10;         % No. of iterations for the Gaussian Newton
% algorithm (Riemannian manifold)
modelPD.nbSamples = demons(end);% Number of demonstrations
modelPD.nbIterEM  = 10;         % Number of iteration for the EM algorithm
reprosMan         = repros;     % reproduction
% Regularization of covariance
modelPD.params_diagRegFact = 1E-4;
% Dimension of the output covariance
modelPD.nbVarCovOut = modelPD.nbVar + modelPD.nbVar*(modelPD.nbVar-1)/2;

% Colors
clrmapCHOL_states   = lines(modelCHOL.nbStates);
clrmapMAN_states    = lines(modelPD.nbStates);
clrmap_demos        = lines(modelCHOL.nbSamples);
clrmap_summer       = summer(max([length(reprosCHOL) length(reprosMan)]));

random = @(a,b,m,n)  a + (b-a).*rand(m,n);

nbData = 20; % Number of datapoints
nbSamples = 1; % Number of demonstrations
nbIter = 5; % Number of iteration for the Gauss Newton algorithm
nbIterEM = 1; % Number of iteration for the EM algorithm
nbDrawingSeg = 60; %Number of segments used to draw ellipsoids


% Initialisation of the covariance transformation
[covOrder4to2, covOrder2to4] = set_covOrder4to2(modelPD.nbVar);
[covOrder4to2out, ~] = set_covOrder4to2(modelPD.nbVar-1);

% Tensor regularization term
tensor_diagRegFact_mat = eye(modelPD.nbVar + modelPD.nbVar * (modelPD.nbVar-1)/2);
tensor_diagRegFact = covOrder2to4(tensor_diagRegFact_mat);
%% Generate covariance data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xIn(1,:) = 1:nbData;
DataK(1,:) = reshape(repmat(xIn,1,modelPD.nbSamples),1,...
    (nbData)*modelPD.nbSamples);

S1 = [15,7;7,17]*.5;[15,7;7,17];%[500,0;0,100];
R = [cosd(45) -sind(45); sind(45) cosd(45)];
S2 = R'*S1*R;
R = [cosd(-45) -sind(-45); sind(-45) cosd(-45)];
S3 = R'*S1*R;
s12 = logmap(S2,S1);
s23 = logmap(S3,S2);

t = linspace(0,1.9,20);0:2/nbData:1-2/nbData;
g12 = geodesic(s12,S1,t);
g23 = geodesic(s23,S2,t);

X = zeros(modelPD.nbVar, modelPD.nbVar, nbData*nbSamples);
X(1,1,:) = reshape(repmat(xIn,1,nbSamples),1,1,nbData*nbSamples);
X(2:end,2:end,:) = g12;cat(3,g12,g23);
for t = 1:nbData
	noise = .2*randn(2,2);
	%X(2:end,2:end,t) = X(2:end,2:end,t) + (noise+noise');
    [Kchol,p] = chol(X(2:end,2:end,t));
    DataK(2:4,t) = [Kchol(1,1); Kchol(1,2); Kchol(2,2)];
end

%% Set up the movie.
fps = 30;
writerObj1 = VideoWriter('spd_vs_chol2.avi'); % Name it.
writerObj1.FrameRate = fps; % How many frames per second.
writerObj1.Quality  = 100; % How many frames per second.
open(writerObj1);
loops = 100;
%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(modelPD.nbStates);

% Manifold plot
f2 = figure('position',[10,10,1820,680],'color',[1 1 1]); hold on; axis off; axis equal;
%f2 = figure('color',[1 1 1]); hold on; axis off; axis equal;
% Plot the SPD convex cone
r = 20;
phi = 0:0.1:2*pi+0.1;
x = [zeros(size(phi)); r.*ones(size(phi))];
y = [zeros(size(phi));r.*sin(phi)];
z = [zeros(size(phi));r/sqrt(2).*cos(phi)];

h = mesh(x,y,z,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])

h = plot3(x(2,:),y(2,:),z(2,:),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(x(:,63),y(:,63),z(:,63),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,62,[0,0,0])
h = plot3(x(:,40),y(:,40),z(:,40),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,25,[0,0,0])

% Settings
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off
% xlabel('\alpha'); ylabel('\gamma'); zlabel('\beta');
view(70,12);
view(-150,-10);

pause( 0.1 );
for lll = 1:10
	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
	writeVideo(writerObj1, frame);
end

% Plot datapoints on the manifold
ll = floor(linspace(3,18,8));nn = length(ll);
pdata = plot3(reshape(X(2,2,ll),[nn,1]), reshape(X(3,3,ll),[nn,1]), reshape(X(3,2,ll),[nn,1]), '.','markersize',15,'color',[0.1 0.1 0.1]);
p1st = plot3(reshape(X(2,2,1),[1,1]), reshape(X(3,3,1),[1,1]), reshape(X(3,2,1),[1,1]), '^','markersize',8,'MarkerFaceColor',[0 0 0],'color',[0.1 0.1 0.1]);
plast = plot3(reshape(X(2,2,20),[1,1]), reshape(X(3,3,20),[1,1]), reshape(X(3,2,20),[1,1]), '^','markersize',8,'MarkerFaceColor',[0 0 0],'color',[0.1 0.1 0.1]);

h_leg = legend([pdata(1)],'Data points $\mathbf{A}_t$',...
    'Location','northwest');
set(h_leg,'Interpreter','latex','FontSize',20);
legend('boxoff');
print(f2,'-dpng','-r600','f2_1');
% pause( 0.1 );
% for lll = 1:10
% 	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% 	writeVideo(writerObj1, frame);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot geodesics from old to new means
p_geo = [];
U = logmap(S1,X(2:3,2:3,20));
Up = U + repmat(X(2:3,2:3,20),[1,1,nbData]);
umsh = bsxfun(@times,U,reshape(linspace(0,1,nbDrawingSeg),1,1,60));
msh = expmap(umsh, X(2:3,2:3,20));
p_geo = [p_geo plot3(reshape(msh(1,1,:),[nbDrawingSeg,1]), reshape(msh(2,2,:),[nbDrawingSeg,1]), reshape(msh(2,1,:),[nbDrawingSeg,1]), '-','linewidth',2,'color',[0.6 0.6 0.6])];

h_leg = legend([pdata(1) p_geo(1)],...
   'Data points $\mathbf{A}_t$', 'Geodesic $\rho$',...
   'Location','northwest');
set(h_leg,'Interpreter','latex','FontSize',20);
print(f2,'-dpng','-r600','f2_2');

pause( 0.1 );
for lll = 1:loops
	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
	writeVideo(writerObj1, frame);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learning: GMM on SPD Manifolds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialisation of the covariance transformation for
% GMM over SPD ellips.
[covOrder4to2, covOrder2to4] = set_covOrder4to2(modelPD.nbVar);
[covOrder4to2out, ~] = set_covOrder4to2(modelPD.nbVar-1);

disp('Learning GMM2 (K ellipsoids)...');
in     = 1;               % Input variables
out    = 2:modelPD.nbVar;	% Output variables

tic;
% GMM initialisation on the manifold
modelPD = spd_init_GMM_kbins(X, modelPD, modelPD.nbSamples);
[modelPD, GAMMA, L] = spd_EM_GMM(X, modelPD, nbData,in,out);
elapsed_t_Man(1) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GMR on SPD (version with single optimization loop)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xIntest = [5.5 15.5 22 24];linspace(1,30,nbData) * 1;
disp('Regression...');
xInSPD = zeros(1,1,length(xIntest));%zeros(3,3,nbData);
xInSPD(1,1,:) = reshape(repmat(xIntest,1,1),1,1,length(xIntest));%reshape(repmat(xIn,1,1),1,1,nbData);
in     = 1;
out    = 2:3;
tic;
for j = reprosMan
%     for i=1:(nbData-rdp)
%         xInSPD(2,2,i) = dD(j).avgFe(1,i+rdp);
%         xInSPD(3,3,i) = dD(j).avgFe(2,i+rdp);
%     end
    newModelPD(j) = spd_GMR(modelPD, xInSPD, in, out);
end
elapsed_t_Man(2) = toc/length(reprosMan);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learning: GMM on CHOLesky decomposition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
% GMM initialization and training with an expectation-maximization
% Eq. (5)
modelCHOL = init_GMM_kbins(DataK(:,:), modelCHOL, modelCHOL.nbSamples);
modelCHOL = EM_GMM(DataK(:,:), modelCHOL);
elapsed_t_CHOL(1) = toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:modelCHOL.nbStates
    figure(f2);
    muu = zeros(2,2);
    muu(1,1) = modelCHOL.Mu(2,i);
    muu(1,2) = modelCHOL.Mu(3,i);
    muu(2,2) = modelCHOL.Mu(4,i);
    mu(:,:,i) = muu' * muu;
end
%% Draw Mu CHOL
for i=1:modelCHOL.nbStates
    figure(f2);
    pmuChol(i) = plot3(mu(1,1,i),mu(2,2,i),mu(1,2,i), '.','markersize',30,'color',[1 0 0]);
    %plotGMM3D(MU, modelPD.Sigma, [0.3 0.3 0.3], alpha, dispOpt)
end
h_leg = legend([pdata(1) p_geo(1) pmuChol(1)],...
   'Data points $\mathbf{A}_t$', 'Geodesic $\rho$',...
   'Mean for Cholesky representation $\mathbf{M}_{chol}$',...
   'Location','northwest');
set(h_leg,'Interpreter','latex','FontSize',20);
print(f2,'-dpng','-r600','f2_3');

pause( 0.1 );
for lll = 1:loops
	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
	writeVideo(writerObj1, frame);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Draw Mu SPD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p1 = []; p2 = []; p3 = [];
for i=1:modelPD.nbStates
    figure(f2);
    p2 = [p2 plot3(modelPD.MuMan(2,2,i),modelPD.MuMan(3,3,i),...
        modelPD.MuMan(3,2,i), '.','markersize',30,'color',clrmap(i,:))];
end
pmuSpd = p2;
h_leg = legend([pdata(1) p_geo(1) pmuChol(1) pmuSpd(1)],...
   'Data points $\mathbf{A}_t$', 'Geodesic $\rho$',...
   'Mean for Cholesky representation $\mathbf{M}_{chol}$',...
   'Mean for Riemannian manifold $\mathbf{M}_{spd}$',...
   'Location','northwest');
set(h_leg,'Interpreter','latex','FontSize',20);
print(f2,'-dpng','-r600','f2_4');

pause( 0.1 );
for lll = 1:loops
	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
	writeVideo(writerObj1, frame);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reproduction: GMR on CHOLesky decomposition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in     = 1;   % Input variables
out    = 2:4;   % Output variables
tic;
for j = reprosCHOL
    tmpModel(j) = GMR(modelCHOL,xIntest,in, out);         % Eq. (8)
end
elapsed_t_CHOL(2) = toc/length(reprosCHOL);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(xIntest)%nbData
    figure(f2);
    muu = zeros(2,2);
    muu(1,1) = tmpModel(j).Mu(1,i);
    muu(1,2) = tmpModel(j).Mu(2,i);
    muu(2,2) = tmpModel(j).Mu(3,i);
    muO(:,:,i) = muu' * muu;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot new estimation of the output
figure(f2)
for t=1:length(xIntest)%nbData
    gmr_c = plot3(muO(1,1,t),muO(2,2,t),muO(2,1,t), '.','markersize',25,'color',[0.5 .0 .8]);
end
h_leg = legend([pdata(1) p_geo(1) pmuChol(1) pmuSpd(1) gmr_c(1)],...
   'Data points $\mathbf{A}_t$', 'Geodesic $\rho$',...
   'Mean for Cholesky representation $\mathbf{M}_{chol}$',...
   'Mean for Riemannian manifold $\mathbf{M}_{spd}$',...
   'New data using GMR on Cholesky $\hat{\mathbf{A}}_{chol}$',...
   'Location','northwest');
set(h_leg,'Interpreter','latex','FontSize',20);
print(f2,'-dpng','-r600','f2_5');

pause( 0.1 );
for lll = 1:loops
	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
	writeVideo(writerObj1, frame);
end

% Plot new estimation of the output
figure(f2)
for t=1:length(xIntest)%nbData
    gmr_s = plot3(newModelPD(j).Mu(1,1,t),newModelPD(j).Mu(2,2,t),newModelPD(j).Mu(2,1,t), '.','markersize',25,'color',[0 .8 .2]);
end
h_leg = legend([pdata(1) p_geo(1) pmuChol(1) pmuSpd(1) gmr_c(1) gmr_s(1)],...
   'Data points $\mathbf{A}_t$', 'Geodesic $\rho$',...
   'Mean for Cholesky representation $\mathbf{M}_{chol}$',...
   'Mean for Riemannian manifold $\mathbf{M}_{spd}$',...
   'New data using GMR on Cholesky $\hat{\mathbf{A}}_{chol}$',...
   'New data using GMR on Riemannian manifold $\hat{\mathbf{A}}_{spd}$','Location','northwest');
set(h_leg,'Interpreter','latex','FontSize',20);
print(f2,'-dpng','-r600','f2_6');

pause( 0.1 );
for lll = 1:loops
	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
	writeVideo(writerObj1, frame);
end
% h_leg = legend([pdata(1) pmuSpd(1) pmuChol(1) gmr_s(1) gmr_c(1) p_geo(1)],...
%    '$\mathbf{A}_t$','$\mathbf{M}_{spd}$', '$\mathbf{M}_{chol}$',...
%    '$\hat{\mathbf{A}}_{spd}$', '$\hat{\mathbf{A}}_{chol}$', '$\rho$', 'Location','northeast');
% set(h_leg,'Interpreter','latex','FontSize',11);

% Plot geodesics from old to new means
p_geo = [];
U = logmap(S1,newModelPD(j).Mu(:,:,end));
Up = U + repmat(newModelPD(j).Mu(:,:,end),[1,1,nbData]);
umsh = bsxfun(@times,U,reshape(linspace(0,1,nbDrawingSeg),1,1,60));
msh = expmap(umsh, newModelPD(j).Mu(:,:,end));
p_geo = [p_geo plot3(reshape(msh(1,1,:),[nbDrawingSeg,1]), reshape(msh(2,2,:),[nbDrawingSeg,1]), reshape(msh(2,1,:),[nbDrawingSeg,1]), '-','linewidth',2,'color',[0.6 0.6 0.6])];

pause( 0.1 );
for lll = 1:loops
	frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
	writeVideo(writerObj1, frame);
end


close(writerObj1); % Saves the movie.

print(f2,'-dpng','-r600','f2_7.png');
% print(f2,'-depsc2','-r300',['fig_spdCone']);

