# -*- coding: utf-8 -*-

"""
This python module implements helpers for IPython / Jupyter and CUDA

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

import logging

from IPython.core.magic import line_magic, Magics, magics_class
import pycuda.driver as cuda



@magics_class
class MyIPythonMagic(Magics): 
    @line_magic
    def cuda_context_handler(self, context_name):
        self.logger =  logging.getLogger(__name__)
        
        self.logger.debug("Registering %s as a global context", context_name)
        
        if context_name in self.shell.user_ns.keys():
            self.logger.debug("Context already registered! Ignoring")
            return
        else:
            self.logger.debug("Creating context")
            self.shell.ex(context_name + " = Common.CUDAContext(blocking=False)")
        
        # this function will be called on exceptions in any cell
        def custom_exc(shell, etype, evalue, tb, tb_offset=None):
            self.logger.exception("Exception caught: Resetting to CUDA context %s", context_name)
            while (cuda.Context.get_current() != None):
                context = cuda.Context.get_current()
                self.logger.info("Popping <%s>", str(context.handle))
                cuda.Context.pop()

            if context_name in self.shell.user_ns.keys():
                self.logger.info("Pushing <%s>", str(self.shell.user_ns[context_name].cuda_context.handle))
                self.shell.ex(context_name + ".cuda_context.push()")
            else:
                self.logger.error("No CUDA context called %s found (something is wrong)", context_name)
                self.logger.error("CUDA will not work now")

            self.logger.debug("==================================================================")
            
            # still show the error within the notebook, don't just swallow it
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

        # this registers a custom exception handler for the whole current notebook
        get_ipython().set_custom_exc((Exception,), custom_exc)
        
        
        # Handle CUDA context when exiting python
        import atexit
        def exitfunc():
            self.logger.info("Exitfunc: Resetting CUDA context stack")
            while (cuda.Context.get_current() != None):
                context = cuda.Context.get_current()
                self.logger.info("`-> Popping <%s>", str(context.handle))
                cuda.Context.pop()
            self.logger.debug("==================================================================")
        atexit.register(exitfunc)
        
    @line_magic
    def setup_logging(self, log_filename):
        import sys
        
        #Get root logger
        logger = logging.getLogger('')
        logger.setLevel(logging.DEBUG)

        #Add log to file
        fh = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        #Add log to screen
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        
        logger.info('Registering logging to %s', log_filename)
        logger.info("Python version %s", sys.version)


# Register 
ip = get_ipython()
ip.register_magics(MyIPythonMagic)
