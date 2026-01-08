function newModel = spd_GMR(modelPD, xIn, in, out)
% GMR for SPD data on Riemannian manifold.
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

%disp('Regression...');
nbData = size(xIn,3);
nbVarOut = length(out);

uhat = zeros(nbVarOut,nbVarOut,nbData);
xhat = zeros(nbVarOut,nbVarOut,nbData);
uOutTmp = zeros(nbVarOut,nbVarOut,modelPD.nbStates,nbData);
SigmaTmp = zeros(modelPD.nbVarCovOut,modelPD.nbVarCovOut,modelPD.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,nbVarOut,nbVarOut,nbData);
H = [];

% Initialisation of the covariance transformation for GMM over man. ellips.
[covOrder4to2, covOrder2to4] = set_covOrder4to2(modelPD.nbVar);

% Initialisation of the covariance transformation for output covariance
[covOrder4to2_in, ~] = set_covOrder4to2(length(in));

for t=1:nbData
    % GMR for manipulability ellipsoids
    % Compute activation weight
    for i=1:modelPD.nbStates
        % Compute H using the reduced form
        xInVec = symMat2Vec(xIn(:,:,t));
        MuManInVec = symMat2Vec(modelPD.MuMan(in,in,i));
        MuInVec = symMat2Vec(modelPD.Mu(in,in,i));
        %SigmaInVec = covOrder4to2_out(modelPD.Sigma(in,in,in,in,i));
        SigmaInVec = covOrder4to2_in(modelPD.Sigma(in,in,in,in,i));
        H(i,t) = modelPD.Priors(i) * gaussPDF(xInVec-MuManInVec, MuInVec, SigmaInVec);
       %H(i,t) = modelPD.Priors(i) * gaussPDF(xIn(:,t)-modelPD.MuMan(in,in,i),...
        %modelPD.Mu(in,in,i), modelPD.Sigma(in,in,in,in,i));
    end
    H(:,t) = H(:,t) / sum(H(:,t)+realmin);
    
    % Compute conditional mean (with covariance transportation)
    if t==1
        [~,id] = max(H(:,t));
        xhat(:,:,t) = modelPD.MuMan(out,out,id); % Initial point
    else
        xhat(:,:,t) = xhat(:,:,t-1);
    end
    for n=1:modelPD.nbIter
        uhat(:,:,t) = zeros(nbVarOut,nbVarOut);
        for i=1:modelPD.nbStates
            % Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t)
            S1 = modelPD.MuMan(out,out,i);
            S2 = xhat(:,:,t);
            Ac = blkdiag(eye(length(in)),transp(S1,S2));
            
            % Parallel transport of eigenvectors
            for j = 1:size(modelPD.V,3)
                pV(:,:,j,i) = Ac * modelPD.D(j,j,i)^.5 * modelPD.V(:,:,j,i) * Ac';
            end
            
            % Parallel transported sigma (reconstruction from eigenvectors)
            pSigma(:,:,:,:,i) = zeros(size(modelPD.Sigma(:,:,:,:,i)));
            for j = 1:size(modelPD.V,3)
                pSigma(:,:,:,:,i) = pSigma(:,:,:,:,i) + outerprod(pV(:,:,j,i),pV(:,:,j,i));
            end
            
            [SigmaTmp(:,:,i), pV(:,:,:,i), pD(:,:,i)] = covOrder4to2(pSigma(:,:,:,:,i));
            
            % Gaussian conditioning on the tangent space
            %uOutTmp(:,:,i,t) = logmap(modelPD.MuMan(out,out,i), xhat(:,:,t)) + ...
            %	tensor4o_mult(tensor4o_div(pSigma(out,out,in,in,i),pSigma(in,in,in,in,i)), (xIn(:,t)-modelPD.MuMan(in,in,i)));
            % Gaussian conditioning on the tangent space
            uOutTmp(:,:,i,t) = logmap(modelPD.MuMan(out,out,i), xhat(:,:,t)) + ...
                tensor4o_mult(tensor4o_div(pSigma(out,out,in,in,i),pSigma(in,in,in,in,i),covOrder4to2_in), ...
                (xIn(:,:,t)-modelPD.MuMan(in,in,i)));
            
            uhat(:,:,t) = uhat(:,:,t) + uOutTmp(:,:,i,t) * H(i,t);
        end
        
        xhat(:,:,t) = expmap(uhat(:,:,t), xhat(:,:,t));
    end
    
    % Compute conditional covariances
    % 	for i=1:modelPD.nbStates
    % 		expSigma(:,:,:,:,t) = expSigma(:,:,:,:,t) + H(i,t) * (pSigma(out,out,out,out,i) ...
    % 			- tensor4o_mult(tensor4o_div(pSigma(out,out,in,in,i),pSigma(in,in,in,in,i)),pSigma(in,in,out,out,i)));
    %   end
    % Compute conditional covariances (note that since uhat=0, the final part in the GMR computation is dropped)
    for i=1:modelPD.nbStates
        SigmaOutTmp = pSigma(out,out,out,out,i) ...
            - tensor4o_mult(tensor4o_div(pSigma(out,out,in,in,i),pSigma(in,in,in,in,i),covOrder4to2_in),pSigma(in,in,out,out,i));
        expSigma(:,:,:,:,t) = expSigma(:,:,:,:,t) + H(i,t) * (SigmaOutTmp + outerprod(uOutTmp(:,:,i,t),uOutTmp(:,:,i,t)));
    end
    
    
end
newModel.Mu = xhat;
newModel.Sigma = expSigma;
newModel.H = H;
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

function Ac = transp(S1,S2)
% Parallel transport (SPD manifold)
t = 1;
U = logmap(S2,S1);
Ac = S1^.5 * expm(0.5 .* t .* S1^-.5 * U * S1^-.5) * S1^-.5;
%Ac2 = S1^-.5 * expm(0.5 .* t .* S1^-.5 * U * S1^-.5) * S1^.5;

%Transportation operator to move V from S1 to S2: VT = Ac*V*Ac2 = Ac*V*Ac'?
%Computationally economic way: Ac = (S2/S1)^.5
end

function v = symMat2Vec(S)
% Reduced vectorisation of a symmetric matrix
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


function T = tensor4o_div(A,B,covOrder4to2)
% Division of two 4th-order tensors A and B
if ndims(A) == 4 || ndims(B) == 4
    sizeA = size(A);
    sizeB = size(B);
    if ismatrix(A)
        sizeA(3:4) = [1,1];
    end
    if ismatrix(B)
        T = A/B;
    else
        if sizeA(3) ~= sizeB(1) || sizeA(4) ~= sizeB(2)
            error('Dimensions mismatch: two last dim of A should be the same than two first dim of B.');
        end
        
        [~, V, D] = covOrder4to2(B);
        invB = zeros(size(B));
        for j = 1:size(V,3)
            invB = invB + D(j,j)^-1 .* outerprod(V(:,:,j),V(:,:,j));
        end
        
        % multiplication with outer product, more expensive (loops)
        % 		T = zeros(sizeA(1),sizeA(2),sizeB(3),sizeB(4));
        %
        % 		for i = 1:sizeA(3)
        % 			for j = 1:sizeA(4)
        % 				T = T + outerprod(A(:,:,i,j),permute(invB(i,j,:,:),[3,4,1,2]));
        % 			end
        % 		end
        
        % multiplication using matricisation of tensors
        T = reshape(tensor2mat(A,[1,2]) * tensor2mat(invB,[1,2]), [sizeA(1),sizeA(2),sizeB(3),sizeB(4)]);
    end
else
    if ismatrix(A) && isscalar(B)
        T = A/B;
    else
        error('Dimensions mismatch.');
    end
end
end
function T = tensor4o_mult(A,B)
% Multiplication of two 4th-order tensors A and B
if ndims(A) == 4 || ndims(B) == 4;
    sizeA = size(A);
    sizeB = size(B);
    if ismatrix(A)
        sizeA(3:4) = [1,1];
    end
    if ismatrix(B)
        sizeB(3:4) = [1,1];
    end
    
    if sizeA(3) ~= sizeB(1) || sizeA(4) ~= sizeB(2)
        error('Dimensions mismatch: two last dim of A should be the same than two first dim of B.');
    end
    
    T = zeros(sizeA(1),sizeA(2),sizeB(3),sizeB(4));
    
    for i = 1:sizeA(3)
        for j = 1:sizeA(4)
            T = T + outerprod(A(:,:,i,j),permute(B(i,j,:,:),[3,4,1,2]));
        end
    end
else
    if ismatrix(A) && isscalar(B)
        T = A*B;
    else
        error('Dimensions mismatch.');
    end
end
end

function M = tensor2mat(T, rows, cols)
% Matricisation of a tensor
% The rows, respectively columns of the matrix are 'rows', respectively
% 'cols' of the tensor.
if nargin <=2
    cols = [];
end

sizeT = size(T);
N = ndims(T);

if isempty(rows)
    rows = 1:N;
    rows(cols) = [];
end
if isempty(cols)
    cols = 1:N;
    cols(rows) = [];
end

if any(rows(:)' ~= 1:length(rows)) || any(cols(:)' ~= length(rows)+(1:length(cols)))
    T = permute(T,[rows(:)' cols(:)']);
end

M = reshape(T,prod(sizeT(rows)), prod(sizeT(cols)));

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