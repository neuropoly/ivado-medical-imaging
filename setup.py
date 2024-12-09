from setuptools import setup, find_packages
from codecs import open
from os import path

# Manually specified, more generic version of the software.
# See: https://stackoverflow.com/a/49684835
with open('requirements.txt') as f:
    requirements = f.readlines()

# Get README
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get Release version
path_version = path.join(this_directory, 'ivadomed', 'version.txt')
with open(path_version) as f:
    version = f.read().strip()

extra_requirements = {
    'docs': [
        # pin sphinx to match what RTD uses:
        # https://github.com/readthedocs/readthedocs.org/blob/ecac31de54bbb2c100f933e86eb22b0f4389ba84/requirements/pip.txt#L16
        'sphinx',
        'sphinx_rtd_theme',
        'sphinx-tabs',
        'sphinx-toolbox',
        'sphinx-jsonschema',
        'pypandoc',
    ],
    'tests': [
        'pytest~=6.2',
        'pytest-cases~=3.6.8',
        'pytest-cov',
        'pytest-ordering~=0.6',
        'pytest-console-scripts~=1.1',
        'coverage',
        'coveralls',
    ],
    'contrib': [
        'pre-commit>=2.10.1',
        'flake8',
    ]
}

extra_requirements['dev'] = [
    requirements,
    extra_requirements['docs'],
    extra_requirements['tests'],
    extra_requirements['contrib'],
    ]

setup(
    name='ivadomed',
    version=version,
    description='Feature conditioning for IVADO medical imaging project.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neuropoly/ivadomed',
    author='NeuroPoly and Mila',
    author_email='none@none.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.7,<3.11',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    extras_require=extra_requirements,
    entry_points={
        'console_scripts': [
            'ivadomed=ivadomed.main:run_main',
            'ivadomed_prepare_dataset_vertebral_labeling=ivadomed.scripts.prepare_dataset_vertebral_labeling:main',
            'ivadomed_automate_training=ivadomed.scripts.automate_training:main',
            'ivadomed_compare_models=ivadomed.scripts.compare_models:main',
            'ivadomed_visualize_transforms=ivadomed.scripts.visualize_transforms:main',
            'ivadomed_convert_to_onnx=ivadomed.scripts.convert_to_onnx:main',
            'ivadomed_extract_small_dataset=ivadomed.scripts.extract_small_dataset:main',
            'ivadomed_download_data=ivadomed.scripts.download_data:main',
            'ivadomed_segment_image=ivadomed.scripts.segment_image:main',
            'ivadomed_training_curve=ivadomed.scripts.training_curve:main',
            'ivadomed_visualize_and_compare_testing_models=ivadomed.scripts.visualize_and_compare_testing_models:main'
        ],
    },
)
