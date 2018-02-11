# TurbineTimeSeries

### Install
```
git clone https://github.com/j-goldsmith/TurbineTimeSeries.git
cd TurbineTimeSeries
pip install .
```
#### Alternative Install
```
pip install git+ssh://git@github.com/j-goldsmith/TurbineTimeSeries.git
```
*note: always use 'git' user in url, not your actual username. auth is by ssh key.

### Upgrade
```
pip install . --upgrade

pip install git+ssh://git@github.com/j-goldsmith/TurbineTimeSeries.git --upgrade
```
Run an upgrade after any changes made to /TurbineTimeSeries. Notebook kernels must be restarted afterwards. 

### Usage
```python
from TurbineTimeSeries import tasks
tasks.sql_insert_anonymized_data('.config')
```
