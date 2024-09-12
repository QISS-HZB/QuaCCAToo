from setuptools import setup, find_packages

setup(
    name='QuaCCAToo',
    version='0.1',
    description='Quantum Color Centers Analysis Toolbox',
    url='https://github.com/QISS-HZB/QuaCCAToo',
    author='Lucas Tsunaki, Anmol Singh, Sergei Trofimov',
    author_email='lucas.tsunaki@helmholt-berlin.de, anmol.singh@helmholtz-berlin.de, sergei.trofimov@helmholtz-berlin.de',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy', 'qutip'],
    python_requires='>=3.9',
    package_dir={'': '.'},
)
