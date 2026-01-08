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