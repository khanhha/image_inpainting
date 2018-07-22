G = numgrid('S',5);
A = -delsq(G);
% Don't need F here since its 0,
% but included for demonstration
h = 0.125;
[X,Y] = meshgrid(0:h:0.5,0.5:-h:0);
% careful here with ordering of Y in meshgrid!!
F = 0*X;
F(find(G==0)) = 0;
C = zeros(size(G));
C(1,:) = [0 25 50 75 100];
C(:,end) = [100 75 50 25 0]';
B = h^2*F - 4*del2(C);
b = B(G~=0);
w = A\b;