function [A]=kernal2D(D,U,V,F)
  % Kernal for two dimensional problems
  % D-diffusion coefficient
  % U-Advection velocity in x-direction
  % V-Advection velocity in y-direction
  % F-Coefficient for source term that depend on the variable
  % Note that the routine requires that D, U, and F are constants
  
  % This software is part of GPU Ocean. 
  % 
  % Copyright (C) 2017, 2018 SINTEF Digital
  % Copyright (C) 2017, 2018 Norwegian Meteorological Institute
  % 
  % This script generates the matrices involved in finding the reference
  % solution for the Rossby adjustment problem.
  % 
  % This program is free software: you can redistribute it and/or modify
  % it under the terms of the GNU General Public License as published by
  % the Free Software Foundation, either version 3 of the License, or
  % (at your option) any later version.
  % 
  % This program is distributed in the hope that it will be useful,
  % but WITHOUT ANY WARRANTY; without even the implied warranty of
  % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  % GNU General Public License for more details.
  % 
  % You should have received a copy of the GNU General Public License
  % along with this program.  If not, see <http://www.gnu.org/licenses/>.

  % Global parameters
    global x dx Nx y dy Ny

  % size of resulting matrix
    nn=Nx*Ny;

  % Create the matrix of the problem
    % iv=1: Using full matrix in the initial stage
    % iv=2: Using sparse matrix but using loops
    % iv=3: Using sparse matrix and vectorsization
    iv=3

    if iv==1; 
      % Creating the solution without vectorization: initially called a
        clear a; a(1:nn,1:nn)=0;
        % The interior
          for jj=2:Ny-1; for ii=2:Nx-1;
            % some helpful indexes
              xi=x(ii); yj=y(jj);
              i=ii+(jj-1)*Nx; ip1=i+1;  im1=i-1;
              j=i;            jp1=j+Nx; jm1=j-Nx;
            % x-direction
              a(im1,i) =a(im1,i)     +D/dx^2   -U/(2*dx);
              a(i,i)   =a(i,i)     -2*D/dx^2                +F;
              a(ip1,i) =a(ip1,i)     +D/dx^2   +U/(2*dx);
            % y-direction
              a(jm1,j) =a(jm1,j)     +D/dy^2   -V/(2*dy);
              a(j,j)   =a(j,j)     -2*D/dy^2;
              a(jp1,j) =a(jp1,j)     +D/dy^2   +V/(2*dy);
        end; end;
        % Boundary conditions
          % Left condition (i.e., at boundary x=0)
            for jj=1:Ny;
              i=(jj-1)*Nx+1;
              a(i,i)=1;
            end
          % Rigth condition (i.e., at boundary x=1)
            for jj=1:Ny;
              i=(jj-1)*Nx+Nx;
              a(i,i)=1;
            end
          % Lower condition (i.e., at boundary y=0)
            for ii=1:Nx;
              i=ii;
              a(i,i)=1;
            end
          % Upper condition (i.e., at boundary y=1)
            for ii=1:Nx;
              i=ii+(Ny-1)*Nx;
              a(i,i)=1;
             end
 
      % Creating the sparse matrix
          A=sparse(a);
          % spy(A)
          clear a
    end;




    if iv==2;
      % Create the matrix of the problem using sparse matrix but using loops
        clear A; A=sparse(1,1,0,nn,nn);
       % The interior
         for jj=2:Ny-1; jj,for ii=2:Nx-1;
           % some helpful indexes
             xi=x(ii); yj=y(jj);
             i=ii+(jj-1)*Nx; ip1=i+1;  im1=i-1;
             j=i;            jp1=j+Nx; jm1=j-Nx;

           % x-direction
             % Calculating the constants
               aiim1 =         D/dx^2   -U/(2*dx);
               aii   =      -2*D/dx^2                  +F;
               aiip1 =         D/dx^2   +U/(2*dx);
             % design the sparse matrix
               A=A+sparse(im1,i, aiim1, nn,nn);
               A=A+sparse(i,  i, aii  , nn,nn);
               A=A+sparse(ip1,i, aiip1, nn,nn);
             % Calculating the constants
               ajjm1 =         D/dy^2   -V/(2*dy);
               ajj   =      -2*D/dy^2;
               ajjp1 =         D/dy^2   +V/(2*dy);
             % design the sparse matrix
               A=A+sparse(jp1,j, ajjp1, nn,nn);
               A=A+sparse(j,  j, ajj  , nn,nn);
               A=A+sparse(jm1,j, ajjm1, nn,nn);
          end; end;
        % Boundary conditions
          % Left condition (i.e., at boundary x=0)
            for jj=1:Ny;
              i=(jj-1)*Nx+1;
              A=A+sparse(i,i,1,nn,nn);
            end
          % Rigth condition (i.e., at boundary x=1)
            for jj=1:Ny;
              i=(jj-1)*Nx+Nx;
              A=A+sparse(i,i,1,nn,nn);
            end
          % Lower condition (i.e., at boundary y=0)
            for ii=1:Nx;
              i=ii;
              A=A+sparse(i,i,1,nn,nn);
            end
          % Upper condition (i.e., at boundary y=1)
            for ii=1:Nx;
              i=ii+(Ny-1)*Nx;
              A=A+sparse(i,i,1,nn,nn);
            end
    end;


    if iv==3;
      % Create the matrix of the problem using sparse matrix and vectorsization
        % difining matrixes to be used
          ai=zeros(Nx,Ny); aip1=ai; aim1=ai;
          aj=zeros(Nx,Ny); ajp1=aj; ajm1=aj;
       % Some useful defintions     
        % Setting up left, middle and rigth matrixes
          i=2; im1=i-1; ip1=i+1;     n=Nx-1; nm1=n-1; np1=n+1;
          j=2; jm1=j-1; jp1=j+1;     m=Ny-1; nm1=n-i; im1=i-1;
          % The matrix for the interior of the domain
            aim1(i:n,j:m)=aim1(i:n,j:m)+     D/dx^2   -U/(2*dx);
              ai(i:n,j:m)=  ai(i:n,j:m)   -2*D/dx^2                 +F;
            aip1(i:n,j:m)=aip1(i:n,j:m)+     D/dx^2   +U/(2*dx);
        % Setting up upper, middle and lower matrixes
          % The matrix for the interior of the domain
            ajm1(i:n,j:m)=ajm1(i:n,j:m)+     D/dy^2   -V/(2*dy);
              aj(i:n,j:m)=  aj(i:n,j:m)   -2*D/dy^2;
            ajp1(i:n,j:m)=ajp1(i:n,j:m)+     D/dy^2   +V/(2*dy);

          % Left Boundary
            ai(1,1:Ny)= 1;

          % Rigth Boundary
            ai(Nx,1:Ny)= 1;

          % Upper Boundary
            aj(1:Nx,1)= 1;

          % Lower Boundary
            aj(1:Nx,Ny)= 1;
  
        % Creating the sparse matrix
          clear A;
          i=2; n=Nx-1;
          j=2; m=Ny-1; mp1=m+1; mm1=m-1; 
 
          A=sparse(        1:nn,              1:nn,         reshape(ai+aj            ,nn        ,1) ,nn,nn);
          A=A+sparse( Nx+1-1:Nx*(Ny-1)-1,  Nx+1:Nx*(Ny-1),  reshape(aim1(1:Nx,2:Ny-1), Nx*(Ny-2),1) ,nn,nn);
          A=A+sparse( Nx+1+1:Nx*(Ny-1)+1,  Nx+1:Nx*(Ny-1),  reshape(aip1(1:Nx,2:Ny-1), Nx*(Ny-2),1) ,nn,nn);
          A=A+sparse(Nx+1+Nx:Nx*(Ny-1)+Nx, Nx+1:Nx*(Ny-1),  reshape(ajp1(1:Nx,2:Ny-1), Nx*(Ny-2),1) ,nn,nn);
          A=A+sparse(Nx+1-Nx:Nx*(Ny-1)-Nx, Nx+1:Nx*(Ny-1),  reshape(ajm1(1:Nx,2:Ny-1), Nx*(Ny-2),1) ,nn,nn);

          % spy(A');

    end;

    