from setuptools import find_packages, setup

setup(
   name='hashformers',
   version='2.0.0',
   author='Ruan Chaves Rodrigues',
   author_email='ruanchave93@gmail.com',
   description='Word segmentation with transformers',
   packages=find_packages('src'),
   package_dir={'': 'src'},
   install_requires=[
   "minicons",
   "twitter-text-python",
   "pandas"
   ]
)
