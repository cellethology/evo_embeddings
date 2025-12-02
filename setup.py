from setuptools import setup, find_packages

with open('evo/version.py') as infile:
    exec(infile.read())

try:
    with open('README.md') as f:
        readme = f.read()
except FileNotFoundError:
    readme = 'DNA foundation modeling from molecular to genome scale.'

try:
    with open('requirements.txt') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = []

sources = {
    'evo': 'evo',
    'evo.scripts': 'scripts',
}

# Find all packages including retrieve_embeddings
all_packages = find_packages(exclude=['tests', 'tests.*', 'test_files', 'test_files.*'])

setup(
    name='evo-model',
    version=version,
    description='DNA foundation modeling from molecular to genome scale.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Team Evo',
    url='https://github.com/evo-design/evo',
    license='Apache-2.0',
    packages=all_packages,
    package_data={'evo': ['evo/configs/*.yml']},
    include_package_data=True,
    package_dir=sources,
    install_requires=requirements,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'retrieve-embeddings=retrieve_embeddings.retrieve_embeddings:main',
        ],
    },
)
