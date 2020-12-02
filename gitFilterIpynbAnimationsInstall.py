"""
gitFilterIpynbAnimationsInstall.py

Tailormade for the GPU-Ocean repository

Inspired by https://github.com/kynan/nbstripout


Instructions:

1.  cloning the gpuocean repository
2.  follow the instructions to set the conda environment
3.  run this script by `ipython gitFilterIpynbAnimationsInstall.py' in the gpuocean folder!

Effect:
A git filter will be set in the config file which prevents to commit animations in notebooks 
and helps to keep the worktree cleaner

"""

import sys

with open('.git/config', 'a') as f:
	f.write('\n')
	f.write('[filter "ipynbAnimations"]\n')
	filepath = '"{}" gitFilterIpynbAnimations.py'.format(sys.executable.replace('\\', '/')).replace("exe","exe\\")
	if '.exe' in filepath:
		filepath = '\\' + filepath
	else:
		filepath = 'python gitFilterIpynbAnimations.py'
	f.write('   clean = ' + filepath + '\n')
	f.write('   smudge = cat')
   
with open('.gitattributes', 'a') as f:
    f.write('*.ipynb filter=ipynbAnimations\n')