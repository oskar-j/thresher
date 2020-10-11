DISTNAME = 'thresher-py'
DESCRIPTION = 'Find threshold for fine-tuning output from predict_proba'
AUTHOR = 'Oskar Jarczyk'
AUTHOR_EMAIL = 'oskar.jarczyk@gmail.com'
MAINTAINER = 'Oskar Jarczyk'
MAINTAINER_EMAIL = 'oskar.jarczyk@gmail.com'
LICENSE = 'MIT'
URL = 'https://github.com/oskar-j/thresher'
VERSION = '0.1.0'
DOWNLOAD_URL = 'https://github.com/oskar-j/thresher/archive/v_01_0.tar.gz'
KEYWORDS = ['Hyper-parameters', 'Finetuning', 'ML', 'Optimization', 'AutoML']
CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'License :: OSI Approved',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               ]


def setup_package():
    from setuptools import setup, find_packages

    with open("README.md", "r") as fh:
        long_description = fh.read()

    metadata = dict(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=VERSION,
        classifiers=CLASSIFIERS,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        packages=find_packages(exclude=['*tests*']),
        install_requires=['numpy', 'pandas', 'xlrd'])

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
