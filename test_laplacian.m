R = 'B'; % Other possible shapes include S,N,C,D,A,H,B
% Generate and display the grid.
n = 32;
G = numgrid(R,n);
spy(G)
title('A finite difference grid')
%pause()
D = delsq(G);
DF = full(D);
spy(D)
title('The 5-point Laplacian')

N = sum(G(:)>0);
rhs = ones(N,1);
if (R == 'N') % For nested dissection, turn off minimum degree ordering.
   spparms('autommd',0)
   u = D\rhs;
   spparms('autommd',1)
else
   u = D\rhs; % This is used for R=='L' as in this example
end

U = G;
U(G>0) = full(u(G(G>0)));
clabel(contour(U));
prism
axis square ij