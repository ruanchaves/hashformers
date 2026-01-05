from setuptools import find_packages, setup

setup(
   name='hashformers',
   version='2.0.0',
   author='Ruan Chaves Rodrigues',
   author_email='ruanchave93@gmail.com',
   description='Probabilistic word segmentation for noisy text (source code, URLs, hashtags)',
   packages=find_packages('src'),
   package_dir={'': 'src'},
   python_requires='>=3.7',
   install_requires=[
       "twitter-text-python>=1.1.0",
       "pandas>=1.3.0",
       "torch>=1.9.0",
       "transformers>=4.10.0"
   ]
)
