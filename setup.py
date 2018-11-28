from setuptools import setup

setup(
    name="freehead",
    version="0.1",
    description="Functions for head tracking experiments with Optotrak and Pupil Labs",
    author="Julius Krumbiegel",
    license="MIT",
    packages=["freehead"],
    install_requires=[
        'numpy',
        'scipy',
        'zmq',
        'pylsl',
        'msgpack',
        'pyyaml',
        'pygame',
    ],
    zip_safe=False)
