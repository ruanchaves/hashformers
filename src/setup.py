from setuptools import find_packages, setup

setup(
   name='word_segmentation',
   version='0.1.0',
   author='Ruan Chaves Rodrigues',
   author_email='ruanchave93@gmail.com',
   description='Word segmentation with transformers',
   packages=find_packages('src'),
   package_dir={'': 'src'},
)