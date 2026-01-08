function [modelPD, GAMMA, L] = spd_EM_GMM(X, modelPD, nbData,in,out)
% GMM for SPD data on Riemannian manifold.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please cite the related publications.
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

% Initialisation of the covariance transformation for GMM over man. ellips.
[covOrder4to2, covOrder2to4] = set_covOrder4to2(modelPD.nbVar);
% Tensor regularization term
tensor_diagRegFact_mat = eye(modelPD.nbVar + modelPD.nbVar * (modelPD.nbVar-1)/2);
tensor_diagRegFact = covOrder2to4(tensor_diagRegFact_mat);

modelPD.Mu = zeros(size(modelPD.MuMan));
%in=1:3; % Input dimension
%out=4:modelPD.nbVar; % Output dimension
id = cov_findNonZeroID(in, out, 1, 0);

L = zeros(modelPD.nbStates, nbData*modelPD.nbSamples);
Xts = zeros(modelPD.nbVar, modelPD.nbVar, nbData*modelPD.nbSamples, modelPD.nbStates);
% EM for PD matrices manifold
for nb=1:modelPD.nbIterEM
    %fprintf('.');
    
    % E-step
    for i=1:modelPD.nbStates
        Xts(in,in,:,i) = X(in,in,:)-repmat(modelPD.MuMan(in,in,i),1,1,nbData*modelPD.nbSamples);
        Xts(out,out,:,i) = logmap(X(out,out,:), modelPD.MuMan(out,out,i));
        
        % Compute probabilities using the reduced form (computationally
        % less expensive than complete form)
        xts = symMat2Vec(Xts(:,:,:,i));
        MuVec = symMat2Vec(modelPD.Mu(:,:,i));
        SigmaVec = covOrder4to2(modelPD.Sigma(:,:,:,:,i) + tensor_diagRegFact.*modelPD.params_diagRegFact);
        
        % 		L(i,:) = model.Priors(i) * gaussPDF2(xts, MuVec, SigmaVec);
        L(i,:) = modelPD.Priors(i) * gaussPDF(xts(id,:), MuVec(id,:), SigmaVec(id,id));
        
    end
    GAMMA = L ./ repmat(sum(L,1)+realmin, modelPD.nbStates, 1);
    H = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*modelPD.nbSamples);
    % M-step
    for i=1:modelPD.nbStates
        % Update Priors
        modelPD.Priors(i) = sum(GAMMA(i,:)) / (nbData*modelPD.nbSamples);
        
        % Update MuMan
        for n=1:modelPD.nbIter
            uTmpTot = zeros(modelPD.nbVar,modelPD.nbVar);
            uTmp = zeros(modelPD.nbVar,modelPD.nbVar,nbData*modelPD.nbSamples);
            uTmp(in,in,:) = X(in,in,:)-repmat(modelPD.MuMan(in,in,i),1,1,nbData*modelPD.nbSamples);
            uTmp(out,out,:) = logmap(X(out,out,:), modelPD.MuMan(out,out,i));
            
            for k = 1:nbData*modelPD.nbSamples
                uTmpTot = uTmpTot + uTmp(:,:,k) .* H(i,k);
            end
            modelPD.MuMan(in,in,i) = uTmpTot(in,in) + modelPD.MuMan(in,in,i);
            modelPD.MuMan(out,out,i) = expmap(uTmpTot(out,out), modelPD.MuMan(out,out,i));
        end
        
        % Update SigmaMan
        modelPD.Sigma(:,:,:,:,i) = zeros(modelPD.nbVar,modelPD.nbVar,modelPD.nbVar,modelPD.nbVar);
        for k = 1:nbData*modelPD.nbSamples
            modelPD.Sigma(:,:,:,:,i) = modelPD.Sigma(:,:,:,:,i) + H(i,k) .* outerprod(uTmp(:,:,k),uTmp(:,:,k));
        end
        modelPD.Sigma(:,:,:,:,i) = modelPD.Sigma(:,:,:,:,i) + tensor_diagRegFact.*modelPD.params_diagRegFact;
    end
end

% Eigendecomposition of Sigma
for i=1:modelPD.nbStates
    [~, modelPD.V(:,:,:,i), modelPD.D(:,:,i)] = covOrder4to2(modelPD.Sigma(:,:,:,:,i));
end

fprintf('\n');

end

%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(U,S)
% Exponential map (SPD manifold)
N = size(U,3);
for n = 1:N
    X(:,:,n) = S^.5 * expm(S^-.5 * U(:,:,n) * S^-.5) * S^.5;
end
end

function U = logmap(X,S)
% Logarithm map (SPD manifold)
N = size(X,3);
for n = 1:N
    U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
end
end

function id = cov_findNonZeroID(in, out, isVec_in, isVec_out)

if nargin == 2
    isVec_in = 0;
    isVec_out = 0;
end

numberMat = zeros(out(end));
if ~isVec_in
    numberMat(in,in) = ones(length(in));
else
    numberMat(in,in) = eye(length(in));
end
if ~isVec_out
    numberMat(out,out) = ones(length(out));
else
    numberMat(out,out) = eye(length(out));
end

numberVec = logical(symMat2Vec(numberMat))';
numberVec = numberVec.*[1:length(numberVec)];

id = nonzeros(numberVec)';

end

function v = symMat2Vec(S)
% Reduced vectorisation of a symmetric matrix.
[d,~,N] = size(S);

v = zeros(d+d*(d-1)/2,N);
for n = 1:N
    v(1:d,n) = diag(S(:,:,n));
    
    row = d+1;
    for i = 1:d-1
        v(row:row + d-1-i,n) = sqrt(2).*S(i+1:end,i,n);
        row = row + d-i;
    end
end
end

function [covOrder4to2, covOrder2to4] = set_covOrder4to2(dim)
% Set the factors necessary to simplify a 4th-order covariance of symmetric
% matrix to a 2nd-order covariance. The dimension ofthe 4th-order
% covariance is dim x dim x dim x dim.Return the functions covOrder4to2 and
% covOrder2to4. This function must be called one time for each
% covariance's size.
newDim = dim+dim*(dim-1)/2;

% Computation of the indices and coefficients to transform 4th-order
% covariances to 2nd-order covariances
sizeS = [dim,dim,dim,dim];
sizeSred = [newDim,newDim];
id = [];
idred = [];
coeffs = [];

% left-up part
for k = 1:dim
    for m = 1:dim
        id = [id,sub2ind(sizeS,k,k,m,m)];
        idred = [idred,sub2ind(sizeSred,k,m)];
        coeffs = [coeffs,1];
    end
end

% right-down part
row = dim+1; col = dim+1;
for k = 1:dim-1
    for m = k+1:dim
        for p = 1:dim-1
            for q = p+1:dim
                id = [id,sub2ind(sizeS,k,m,p,q)];
                idred = [idred,sub2ind(sizeSred,row,col)];
                coeffs = [coeffs,2];
                col = col+1;
            end
        end
        row = row + 1;
        col = dim+1;
    end
end

% side-parts
for k = 1:dim
    col = dim+1;
    for p = 1:dim-1
        for q = p+1:dim
            id = [id,sub2ind(sizeS,k,k,p,q)];
            idred = [idred,sub2ind(sizeSred,k,col)];
            id = [id,sub2ind(sizeS,k,k,p,q)];
            idred = [idred,sub2ind(sizeSred,col,k)];
            coeffs = [coeffs,sqrt(2),sqrt(2)];
            col = col+1;
        end
    end
end

% Computation of the indices and coefficients to transform eigenvectors to
% eigentensors
sizeV = [dim,dim,newDim];
sizeVred = [newDim,newDim];
idEig = [];
idredEig = [];
coeffsEig = [];

for n = 1:newDim
    % diagonal part
    for j = 1:dim
        idEig = [idEig,sub2ind(sizeV,j,j,n)];
        idredEig = [idredEig,sub2ind(sizeVred,j,n)];
        coeffsEig = [coeffsEig,1];
    end
    
    % side parts
    j = dim+1;
    for k = 1:dim-1
        for m = k+1:dim
            idEig = [idEig,sub2ind(sizeV,k,m,n)];
            idredEig = [idredEig,sub2ind(sizeVred,j,n)];
            idEig = [idEig,sub2ind(sizeV,m,k,n)];
            idredEig = [idredEig,sub2ind(sizeVred,j,n)];
            coeffsEig = [coeffsEig,1/sqrt(2),1/sqrt(2)];
            j = j+1;
        end
    end
end


    function [Sred, V, D] = def_covOrder4to2(S)
        Sred = zeros(newDim,newDim);
        Sred(idred) = bsxfun(@times,S(id),coeffs);
        [v,D] = eig(Sred);
        V = zeros(dim,dim,newDim);
        V(idEig) = bsxfun(@times,v(idredEig),coeffsEig);
    end
    function [S, V, D] = def_covOrder2to4(Sred)
        [v,D] = eig(Sred);
        V = zeros(dim,dim,newDim);
        V(idEig) = bsxfun(@times,v(idredEig),coeffsEig);
        
        S = zeros(dim,dim,dim,dim);
        for i = 1:size(V,3)
            S = S + D(i,i).*outerprod(V(:,:,i),V(:,:,i));
        end
    end

covOrder4to2 = @def_covOrder4to2;
covOrder2to4 = @def_covOrder2to4;

end