from setuptools import setup, find_packages
import subprocess

def setup_package():
	setup(name="TurbineTimeSeries",
      	      version="0.1",
      	      packages=['TurbineTimeSeries'],
      	      install_requires=[
            	    'sqlalchemy',
            	    'sklearn',
            	    'pandas',
            	    'awscli'
              ],
              zip_safe=False)


if __name__ == '__main__':
    subprocess.call("setup.sh", shell =True)
    setup_package()
