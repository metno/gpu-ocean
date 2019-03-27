# -*- coding: utf-8 -*-

"""
This python module implements Cuda context handling
Copyright (C) 2018  SINTEF ICT
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""



import numpy as np

from matplotlib.colors import Normalize



def genSchlieren(rho):
    #Compute length of z-component of normalized gradient vector 
    normal = np.gradient(rho) #[x, y, 1]
    length = 1.0 / np.sqrt(normal[0]**2 + normal[1]**2 + 1.0)
    schlieren = np.power(length, 128)
    return schlieren


def genVorticity(rho, rho_u, rho_v):
    u = rho_u / rho
    v = rho_v / rho
    u = np.sqrt(u**2 + v**2)
    u_max = u.max()
    
    du_dy, _ = np.gradient(u)
    _, dv_dx = np.gradient(v)
    
    #Length of curl
    curl = dv_dx - du_dy
    return curl


def genColors(rho, rho_u, rho_v, cmap, vmax, vmin):
    schlieren = genSchlieren(rho)
    curl = genVorticity(rho, rho_u, rho_v)

    colors = Normalize(vmin, vmax, clip=True)(curl)
    colors = cmap(colors)
    for k in range(3):
        colors[:,:,k] = colors[:,:,k]*schlieren

    return colors
