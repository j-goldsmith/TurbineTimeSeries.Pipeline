# TurbineTimeSeries

### Install
```
git clone https://github.com/j-goldsmith/TurbineTimeSeries.git
cd TurbineTimeSeries
pip install .
```
#### Alternative Install
Not working...TODO
```
pip install git+ssh://<username>@github.com/j-goldsmith/TurbineTimeSeries.git
```

### Upgrade
```
pip install . --upgrade

pip install git+ssh://<username>@github.com/j-goldsmith/TurbineTimeSeries.git --upgrade
```
Run an upgrade after any changes made to /TurbineTimeSeries. Notebook kernels must be restarted afterwards. 

### Usage
**default config points to SDSC hosted data sources**
```python
from TurbineTimeSeries import tasks
tasks.sql_insert_anonymized_data('.config')
```
