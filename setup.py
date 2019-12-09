from setuptools import setup
from setuptools import find_packages


setup(name='csgan',
      version='0.1',
      description="The code used for our recent works on Conditional Spatial GANS",
      author='Cyprien Ruffino',
      author_email='ruffino.cyprien@gmail.com',
      url='https://github.com/cyprienruffino/csgan',
      license='GPL-3',
      install_requires=['numpy',  'h5py', 'tensorflow', 'progressbar2'],
      extras_require={},
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Topic :: Software Development :: Libraries'
      ],
      packages=find_packages())
