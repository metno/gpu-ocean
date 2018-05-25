%function [A]=kernal2D(D,U,V,F)
  % Kernal for two dimensional problems
  % D-diffusion coefficient
  % U-Advection velocity in x-direction
  % V-Advection velocity in y-direction
  % F-Coefficient for source term that depend on the variable
  % Note that the routine requires that D, U, and F are constants

  close all, clear all, pack
  
 % Global parameters
    global x dx Nx y dy Ny

  % Model setting
    % Grid
      Nx=800;  dx=50000;
      Ny=1000; dy=50000;
      x=((1:Nx)-0.5)*dx; y=((1:Ny)-0.5)*dy;
      
      x_center = dx*Nx/2
      y_center = dy*Ny/2
      
      % Init conditions
      [X,Y]=meshgrid(x-dx/2-x_center, y-dy/2-y_center);
      R=sqrt(X.*X+Y.*Y); figure; pcolor(R); shading flat; colorbar; title("Radius")
    % Physical parameters
      g=9.81;
      f=1.2e-4;
      H0=1000;
     
      % Derived parameters
      c=sqrt(g*H0)
      a=c/f
      aa=1/(a*a)
    
    % Finding the "matricss" kernnal for numerics
      D=1; U=0; V=0; F=aa;
      [A]=kernal2D(-D,U,V,F);
    
    % Values of the source function (i.e., -h0)
      s=zeros(Nx,Ny);
%       % Case, square
%         isize=40 /2
%         s((Nx-1)/2-isize:(Nx-1)/2+isize,(Ny-1)/2-isize:(Ny-1)/2+isize)=aa;       
      % Softer radial conditions (implemented in CTCS code)
        L = 15*dx
        D = 50*dx
        etaamp = 0.2
        s = 0.5*etaamp*(1.0+tanh((-R+D)/L))'; 
      % Plotting the initial condition
       figure; pcolor(s'); colorbar; shading flat; title("Init Eta")
       
    % Finding the solution
      % Organize the matrix for inversion
        B=aa*reshape(s,Nx*Ny,1);
        tmp=A\B;
        h(:,:)=reshape(tmp,Nx,Ny);
      % Plot the solution
        figure; pcolor(h(2:end-1,2:end-1)'); colorbar; shading flat; title("Original result (eta)")
        figure; plot(x/dx, h(:,Ny/2)/etaamp, x/dx, s(:,Ny/2)/aa/etaamp); title("Original Comparison init - steady state")
        xlim([-150 150])
        figure; plot(x/dx, h(:,Ny/2), x/dx, s(:,Ny/2)); title("Comparison init - steady state")
        xlim([-150 150])
        %Not sure why it does not work on the boundaries
      % Save data
       %save('KG.mat','h', 's', 'x', 'y')
         save -6 KG.mat, h, s, x, y;
       
       max(max( h(:,Ny/2)/etaamp))
       
