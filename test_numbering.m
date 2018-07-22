h = 1/20;
[x,y] = meshgrid(-1:h:1);
u = x.^2 + y.^2;
d = (4/h^2) * del2(u);
[x,y] = meshgrid(-1:h:1);

xv = [0  0  1 1 -1 -1 0];
yv = [0 -1 -1 1  1  0 0];
plot(xv, yv);
pause()
[in,on] = inregion(x,y,xv,yv);
