# -*- coding: utf-8 -*-

"""
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

from ctypes import *
import numpy as np
from abc import ABCMeta, abstractmethod

class WIND_STRESS_PARAMS(Structure):
    """Mapped to struct WindStressParams in common.cu
    DO NOT make changes here without changing common.cu accordingly!
    """
    _fields_ = [("wind_stress_type", c_int),
                ("tau0", c_float),
                ("rho", c_float),
                ("rho_air", c_float),
                ("alpha", c_float),
                ("xm", c_float),
                ("Rc", c_float),
                ("x0", c_float),
                ("y0", c_float),
                ("u0", c_float),
                ("v0", c_float),
                ("wind_speed", c_float),
                ("wind_direction", c_float)]

class BaseWindStress(object):
    """Superclass for wind stress params."""
    
    __metaclass__ = ABCMeta

    def __init__(self):
        pass
    
    @abstractmethod
    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        pass
    
    @abstractmethod
    def tostruct(self):
        """Return correct WindStressParams struct (defined above AND in common.cu)"""
        pass
    
    def csize(self):
        """Return size (in bytes) of WindStressParams struct (defined above AND in common.cu)"""
        return sizeof(WIND_STRESS_PARAMS)

class NoWindStress(BaseWindStress):
    """No wind stress."""

    def __init__(self):
        pass
    
    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 0
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type())
        return wind_stress

class GenericUniformWindStress(BaseWindStress):
    """Generic uniform wind stress.
    
    rho_air: Density of air (approx. 1.3 kg / m^3 at 0 deg. C and 1013.25 mb)
    speed: Wind speed in m/s
    direction: Wind direction in degrees (clockwise, 0 being wind blowing from north towards south)
    """

    def __init__(self, \
                 rho_air=0, \
                 wind_speed=0, wind_direction=0):
        self.rho_air = np.float32(rho_air)
        self.wind_speed = np.float32(wind_speed)
        self.wind_direction = np.float32(wind_direction)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 1
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  rho_air=self.rho_air,
                                  wind_speed=self.wind_speed,
                                  wind_direction=self.wind_direction)
        return wind_stress
        
class UniformAlongShoreWindStress(BaseWindStress):
    """Uniform along shore wind stress.
    
    tau0: Amplitude of wind stress (Pa)
    rho: Density of sea water (1025.0 kg / m^3)
    alpha: Offshore e-folding length (1/(10*dx) = 5e-6 m^-1)
    """

    def __init__(self, \
                 tau0=0, rho=0, alpha=0):
        self.tau0 = np.float32(tau0)
        self.rho = np.float32(rho)
        self.alpha = np.float32(alpha)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 2
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  tau0=self.tau0,
                                  rho=self.rho,
                                  alpha=self.alpha)
        return wind_stress
        
class BellShapedAlongShoreWindStress(BaseWindStress):
    """Bell shaped along shore wind stress.
    
    xm: Maximum wind stress for bell shaped wind stress
    tau0: Amplitude of wind stress (Pa)
    rho: Density of sea water (1025.0 kg / m^3)
    alpha: Offshore e-folding length (1/(10*dx) = 5e-6 m^-1)
    """

    def __init__(self, \
                 xm=0, tau0=0, rho=0, alpha=0):
        self.xm = np.float32(xm)
        self.tau0 = np.float32(tau0)
        self.rho = np.float32(rho)
        self.alpha = np.float32(alpha)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 3
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  xm=self.xm,
                                  tau0=self.tau0,
                                  rho=self.rho,
                                  alpha=self.alpha)
        return wind_stress
        
class MovingCycloneWindStress(BaseWindStress):
    """Moving cyclone wind stress.
    
    Rc: Distance to max wind stress from center of cyclone (10dx = 200 000 m)
    x0: Initial x position of moving cyclone (dx*(nx/2) - u0*3600.0*48.0)
    y0: Initial y position of moving cyclone (dy*(ny/2) - v0*3600.0*48.0)
    u0: Translation speed along x for moving cyclone (30.0/sqrt(5.0))
    v0: Translation speed along y for moving cyclone (-0.5*u0)
    """

    def __init__(self, \
                 Rc=0, \
                 x0=0, y0=0, \
                 u0=0, v0=0):
        self.Rc = np.float32(Rc)
        self.x0 = np.float32(x0)
        self.y0 = np.float32(y0)
        self.u0 = np.float32(u0)
        self.v0 = np.float32(v0)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 4
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  Rc=self.Rc,
                                  x0=self.x0,
                                  y0=self.y0,
                                  u0=self.u0,
                                  v0=self.v0)
        return wind_stress
