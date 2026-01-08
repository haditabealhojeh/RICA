function model = spd_init_GMM_kbins(Data, model, nbSamples, spdDataId)
% Initialization of Gaussian Mixture Model (GMM) parameters by clustering 
% an ordered dataset into equal bins for SPD data on Riemannian manifold.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Jaquier17IROS,
%   author="Jaquier, N. and Calinon, S.",
%   title="Gaussian Mixture Regression on Symmetric Positive Definite Matrices Manifolds: 
%	    Application to Wrist Motion Estimation with s{EMG}",
%   year="2017",
%	  booktitle = "{IEEE/RSJ} Intl. Conf. on Intelligent Robots and Systems ({IROS})",
%	  address = "Vancouver, Canada"
% }
% 
% Copyright (c) 2017 Idiap Research Institute, http://idiap.ch/
% Written by Noémie Jaquier and Sylvain Calinon
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

nbData = size(Data,3) / nbSamples;
if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-2; %Optional regularization term to avoid numerical instability
end
e0tensor = zeros(model.nbVar, model.nbVar, model.nbVar, model.nbVar);
for i = 1:model.nbVar
	e0tensor(i,i,i,i) = 1;
end
% Delimit the cluster bins for the first demonstration
tSep = round(linspace(0, nbData, model.nbStates+1));

% Compute statistics for each bin
for i=1:model.nbStates
	id=[];
	for n=1:nbSamples
		id = [id (n-1)*nbData+[tSep(i)+1:tSep(i+1)]];
	end
	model.Priors(i) = length(id);
	
	% Mean computed on SPD manifold for parts of the data belonging to the
	% manifold
	if nargin < 4
		model.MuMan(:,:,i) = spdMean(Data(:,:,id));
	else
		model.MuMan(:,:,i) = mean(Data(:,:,id),3);
		if iscell(spdDataId)
			for c = 1:length(spdDataId)
				model.MuMan(spdDataId{c},spdDataId{c},i) = spdMean(Data(spdDataId{c},spdDataId{c},id),3);
			end
		else
			model.MuMan(spdDataId,spdDataId,i) = spdMean(Data(spdDataId,spdDataId,id),3);
		end
	end
	
	% Parts of data belonging to SPD manifold projected to tangent space at
	% the mean to compute the covariance tensor in the tangent space
	DataTgt = zeros(size(Data(:,:,id)));
	if nargin < 4
		DataTgt = logmap(Data(:,:,id),model.MuMan(:,:,i));
	else
		DataTgt = Data(:,:,id);
		if iscell(spdDataId)
			for c = 1:length(spdDataId)
				DataTgt(spdDataId{c},spdDataId{c},:) = logmap(Data(spdDataId{c},spdDataId{c},id),model.MuMan(spdDataId{c},spdDataId{c},i));
			end
		else
			DataTgt(spdDataId,spdDataId,:) = logmap(Data(spdDataId,spdDataId,id),model.MuMan(spdDataId,spdDataId,i));
		end
	end

	model.Sigma(:,:,:,:,i) = computeCov(DataTgt) + e0tensor.*model.params_diagRegFact;
	
end
model.Priors = model.Priors / sum(model.Priors);
end
%% Functions
%%%%%%%%%%%%%%%%%
function M = spdMean(setS, nbIt)
% Mean of SPD matrices on the manifold
if nargin == 1
    nbIt = 10;
end
M = setS(:,:,1);

for i=1:nbIt
    L = zeros(size(setS,1),size(setS,2));
    for n = 1:size(setS,3)
        L = L + logm(M^-.5 * setS(:,:,n)* M^-.5);
    end
    M = M^.5 * expm(L./size(setS,3)) * M^.5;
end
end

function U = logmap(X,S)
% Logarithm map (SPD manifold)
N = size(X,3);
for n = 1:N
    U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
end
end

function S = computeCov(Data)
% Compute the 4th-order covariance of matrix data.
d = size(Data,1);
N = size(Data,3);

Data = Data-repmat(mean(Data,3),1,1,N);

S = zeros(d,d,d,d);
for i = 1:N
    S = S + outerprod(Data(:,:,i),Data(:,:,i));
end
S = S./(N-1);
end