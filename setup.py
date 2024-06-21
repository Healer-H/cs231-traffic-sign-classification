from setuptools import setup, find_packages

setup(
    name='traffic_sign_classification',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
