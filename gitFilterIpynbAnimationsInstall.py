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

# Add filter to configs

with open('.git/config', 'r') as f:
	conf = f.read()
	filter_exists_conf = '[filter "ipynbAnimations"]' in conf

if not filter_exists_conf:
	with open('.git/config', 'a') as f:
		f.write('\n')
		f.write('[filter "ipynbAnimations"]\n')
		filepath = '"{}" gitFilterIpynbAnimations.py'.format(sys.executable.replace('\\', '/')).replace(".exe",".exe\\")
		if sys.platform == 'win32': #Windows: win32
			filepath = '\\' + filepath
		else: #others like Linux and Mac: linux/linux2/darwin
			filepath = 'python gitFilterIpynbAnimations.py'
		f.write('   clean = ' + filepath + '\n')
		f.write('   smudge = cat' + '\n')

# Add filter to configs for diff

with open('.git/config', 'r') as f:
	conf = f.read()
	filter_exists_conf = '[filter "ipynbAllOutputs"]' in conf

if not filter_exists_conf:
	with open('.git/config', 'a') as f:
		f.write('\n')
		f.write('[filter "ipynbAllOutputs"]\n')
		filepath = '"{}" gitFilterIpynbAll.py'.format(sys.executable.replace('\\', '/')).replace(".exe",".exe\\")
		if sys.platform == 'win32': #Windows: win32
			filepath = '\\' + filepath
		else: #others like Linux and Mac: linux/linux2/darwin
			filepath = 'python gitFilterIpynbAll.py'
		f.write('   clean = ' + filepath + '\n')
		f.write('   smudge = cat' + '\n')
   
# Add filter to attributes

with open('.gitattributes', 'r') as f:
	attrs = f.read()
	filter_exists_attr = '*.ipynb filter=ipynbAnimations' in attrs

if not filter_exists_attr:
	with open('.gitattributes', 'a') as f:
		f.write('*.ipynb filter=ipynbAnimations\n')

# Add filter driver to attributes

with open('.gitattributes', 'r') as f:
	attrs = f.read()
	filter_exists_attr = '*.ipynb diff=ipynbAllOutputs' in attrs

if not filter_exists_attr:
	with open('.gitattributes', 'a') as f:
		f.write('*.ipynb diff=ipynbAllOutputs\n')