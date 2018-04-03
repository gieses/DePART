"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.


setup(
    name='DePART',
    version='1.1',
    author='Sven H. Giese',
    author_email='sven.giese88@gmail.com',
    packages=['DePART', 
              'DePART.tests',
              'DePART.learning',
              'DePART.reader',
              'DePART.preprocessing',
              'DePART.wrapper'],
              
    scripts=['bin/pyDePART.py'],
    url='http://pypi.python.org/pypi/DePART/',
    license='LICENSE.txt',
    description='DePART - Deep Learning for Predicting Anion Exchange Chromatography Retention Times.',  # Required
    long_description=long_description,
    install_requires=['pandas', 'biopython','pyteomics','keras',
                      'scikit-learn','numpy', 'joblib', 'tensorflow'],
        #"Django >= 1.1.1",
        #"caldav == 0.1.4",
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/gieses/DePART',
        'Source': 'https://github.com/gieses/DePART',
    }
)


