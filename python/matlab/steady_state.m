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
      Nx=801;  dx=50000;
      Ny=1001; dy=50000;
      x=((1:Nx)-(Nx-1)/2)*dx; y=((1:Ny)-(Ny-1)/2)*dy;
      
    % Init conditions
      [X,Y]=meshgrid(x,y);
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
        %s =aa* 0.5*etaamp*(1.0+tanh((-R+D)/L))'; 
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
        figure; plot(x/dx, h(:,(Ny-1)/2)/etaamp, x/dx, s(:,(Ny-1)/2)/aa/etaamp); title("Original Comparison init - steady state")
        xlim([-150 150])
        figure; plot(x/dx, h(:,(Ny-1)/2), x/dx, s(:,(Ny-1)/2)); title("Comparison init - steady state")
        xlim([-150 150])
        %Not sure why it does not work on the boundaries
      % Save data
       save('KG.mat','h') %, 'X', 'Y')
       
       max(max( h(:,(Ny-1)/2)/etaamp))
       
%% Loop for sensitivity
   close all, clear HH hmax A B s h
   figure(10); hold on; xlabel('x/dx','FontSize',[16]); ylabel('\eta_{max}/\eta_0','FontSize',[16])
   ic=0;
   for xH=100:100:5000; ic=ic+1;
      HH(ic)=xH;
      H0=xH;
      c=sqrt(g*H0);
      a=c/f;
      aa=1/(a*a)
    
      D=1; U=0; V=0; F=aa;
      [A]=kernal2D(-D,U,V,F);
    

        L = 5*dx;  D = 20*dx;  etaamp = 0.2; %D used at two places!!!!
        s=zeros(Nx,Ny);
        s =aa* 0.5*etaamp*(1.0+tanh((-R+D)/L))'; 
       
        B=reshape(s,Nx*Ny,1);
        tmp=A\B;
        h(:,:)=reshape(tmp,Nx,Ny);
      % Plot the solution
        figure(10); plot(x/dx, h(:,(Ny-1)/2)/etaamp, x/dx, s(:,(Ny-1)/2)/aa/etaamp)
        xlim([-150 150])
      % save('KG.mat','h')
     hmax(ic)=max(max(h)) / etaamp ;
     BB(ic)=sum(sum(h)) *dx / etaamp ;
 
   end
   figure; plot(HH,hmax,'LineWidth',[2]); title('\eta at steady state','FontSize',[18])
     xlabel('H_0 [m]','FontSize',[16]); ylabel('\eta_{max}/\eta_0','FontSize',[16])
     xlim([0 5000]); ylim([0 1])
   
    qq= BB/(sum(sum(s))/aa/etaamp *dx)./hmax; % Clumsy, but here is the normalization 
   figure; plot(HH,qq,'LineWidth',[2]); title('Width at steady state','FontSize',[18])
     xlabel('H_0 [m]','FontSize',[16]); ylabel('\int(eta)/\int(eta_0)','FontSize',[16])
     xlim([0 5000]); %ylim([0 1])
   
   