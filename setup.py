from setuptools import setup, find_packages
setup(name="TurbineTimeSeries",
      version="0.1",
      packages=['TurbineTimeSeries'],
      install_requires=[
            'sqlalchemy',
            'sklearn',
            'pandas'
      ],
      zip_safe=False)