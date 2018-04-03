from setuptools import setup
from setuptools import find_packages


setup(name='keras_contrib',
      version='2.0.8',
      description="An easy-to-use GAN library built on Keras",
      author='Cyprien Ruffino',
      author_email='ruffino.cyprien@gmail.com',
      url='https://github.com/cyprienruffino/keras-gan',
      license='GPL-3',
      install_requires=['keras', 'numpy',  'h5py'],
      extras_require={
          'visualize': ['pydot>=1.2.0'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GPL-3 License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())