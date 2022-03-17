#!/bin/bash

####################################################################
# This software is part of GPU Ocean. 
#
# Copyright (C) 2019 SINTEF Digital
# Copyright (C) 2019 Norwegian Meteorological Institute
#
# This python program is used to set up and run a data-assimilation 
# and drift trajectory forecasting experiment.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
####################################################################

output_dir="tmp/forecasts_$(date +%Y_%m_%d-%H_%M_%S)"
echo $output_dir

gpuoceanpython=python

which $gpuoceanpython

# IWEPF2
$gpuoceanpython run_experiment.py -N 100 --method iewpf2 --media_dir $output_dir

# LETKF
$gpuoceanpython run_experiment.py -N 100 --method letkf --media_dir $output_dir

