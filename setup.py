from setuptools import find_packages, setup

setup(
   name='hashformers',
   version='1.0.1',
   author='Ruan Chaves Rodrigues',
   author_email='ruanchave93@gmail.com',
   description='Word segmentation with transformers',
   packages=find_packages('src'),
   package_dir={'': 'src'},
   install_requires=[
   "mlm-hashformers",
   "lm-scorer-hashformers"
   ]
)
