function A = generateSPDmatrix(n)
% Generates a random n x n Symmetric, Positive Definite matrix

A = rand(n,n);

% construct a symmetric matrix using either
A = 0.5*(A+A'); 

% OR
% A = A*A';
% The first is significantly faster: O(n^2) compared to O(n^3)

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding nI

%random = @(a,b,m,n)  a + (b-a).*rand(m,n);
%A = A + random(n-2,n+2,1,1)*eye(n);
% OR
A = A + n*eye(n);

end