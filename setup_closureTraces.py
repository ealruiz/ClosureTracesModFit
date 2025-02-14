### RUN ME AS:			python setup_closureTraces.py build_ext --inplace
### IF USING CASA:	/home/ezequiel/softwares/casa-6.5.4-9-pipeline-2023.1.0.125/bin/python3 setup_closureTraces.py build_ext --inplace

from distutils.core import setup, Extension
import numpy as np

printM  = '\n'
printM += '#####################################################################\n'
printM += '# Compiling with numpy version %s \n'%(np.__version__)
printM += '#                              %s\n'%(np.__file__)
printM += '#####################################################################\n'
printM += '\n'

print(printM)


sourcefiles = ['_closureTraces.cpp']


c_ext = Extension("_closureTraces", sources=sourcefiles,
						extra_compile_args=["-Wno-deprecated","-O3","-std=c++11"],
						libraries=[],
						include_dirs=[np.get_include()],
						extra_link_args=["-Xlinker", "-export-dynamic"])


setup(
	ext_modules=[c_ext],include_dirs=['./'],
)

