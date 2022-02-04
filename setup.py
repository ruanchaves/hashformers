from setuptools import find_packages, setup

setup(
   name='hashformers',
   version='1.0.0',
   author='Ruan Chaves Rodrigues',
   author_email='ruanchave93@gmail.com',
   description='Word segmentation with transformers',
   packages=find_packages('src'),
   package_dir={'': 'src'},
   install_requires=[
   "mlm @ git+git://github.com/ruanchaves/mlm-scoring.git@master#egg=mlm-0.1",
   "lm_scorer @ https://github.com/ruanchaves/hashformers/raw/master/deps/lm_scorer-0.4.2-py3-none-any.whl"
   ]
)