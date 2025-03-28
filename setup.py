import setuptools

from pathlib import Path

INSTALL_REQUIRES = ['torch>=1.7', 'timm==0.9.2']

def get_version():
    locals_ = dict()

    with open(Path(__file__).parent / 'ttm' / '__version__.py') as f:
        exec(f.read(), globals(), locals_)
        return locals_['__version__']

def get_short_description():
    return 'A Python library for volumetric data processing with PyTorch. Useful for machine learning.'

def get_long_description():
    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        return f.read()

def get_python_min_version():
    return (3, 7, 0)

def get_python_min_version_str():
    return '.'.join(map(str, get_python_min_version()))

setuptools.setup(
    name='ttm',
    version=get_version(),
    description=get_short_description(),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=INSTALL_REQUIRES,
    extras_require={'tests': ['pytest']},
    url='https://github.com/rostepifanov/ttm',
    download_url='https://github.com/rostepifanov/ttm/tags',
    author='Rostislav Epifanov',
    author_email='rostepifanov@gmail.com',
    license='MIT',
    python_requires=f'>={get_python_min_version_str()}',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ] + [
        f'Programming Language :: Python :: 3.{i}'
            for i in range(get_python_min_version()[1], 12)
    ],
    keywords='ttm, deep learning',
)
