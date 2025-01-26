from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="My_Plugin", 
        version=VERSION,
        author="Yen-Hsing Lin",
        author_email="yel051@ucsd.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'

        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: Nah",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux",
        ]
)
