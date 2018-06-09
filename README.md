# TurbineTimeSeries

## Install
```
git clone https://github.com/j-goldsmith/TurbineTimeSeries.git
cd TurbineTimeSeries
pip install .
```
### Alternative Install
```
pip install git+ssh://git@github.com/j-goldsmith/TurbineTimeSeries.git
```
*note: always use 'git' user in url, not your actual username. auth is by ssh key.

## Upgrade
```
pip install . --upgrade

pip install git+ssh://git@github.com/j-goldsmith/TurbineTimeSeries.git --upgrade
```
Run an upgrade after any changes made to /TurbineTimeSeries. Notebook kernels must be restarted afterwards.

## Running
#### Update Config File
sample.config is included in the repository.
 - The connection string must be updated to a working server before the pipeline can be run. (proprietary data).
- Optionally, set the export_dir to TurbineTimeSeries.UI/data/pipeline_files

Update the config path in /run_pipeline.py

#### Execute Pipeline

    python run_pipeline.py

#### User Interface
Depending on what your configured export_dir was set to, you may need to move files to /data/pipeline_files.

## Modules
*TurbineTimeSeries is split up into two main modules, Transforms and Exports. Transforms make up the pipeline and Exports pick out the necessary data for TurbineTimeSeries.UI. The pipeline is defined in the Train module.*

*In addition, there are package and environment config modules, a raw data storage module (SQL implemented), and profiling capabilities.*

### Package Models
 - ##### PackageModel
	*Configuration object for package model types*
	###### Construction Params
	- model_number (int, required) - identifying model number
	- indexes (list\<str>, required) - list of column names to use for indices
	- ignored (list\<str>, required) - list of columns from raw data to drop
	- tags (list\<Tag>, required) - list of Tags included in the package data
	###### Functions
	- independent_variables() - return all Tags marked as independent.
	- subsystem(str) - return all Tags in the specified subsystem
 - ##### Tag
	*Configuration object for package data fields*
	###### Construction Params
	- name (str, required)
	- subsystem  (str, required)
	- independent_variable:  (bool, required)
	- measurement_type:  (str, required)
	- description:  (str, required)

### Config
 - ##### load_config
	*Loads CSV config from specified path*
	###### Expected Keys
	- postgres_connection_url
	- cache_dir
	- cache_ttl
	- export_dir

### Storage

 - ##### MachineDataStore
 - ##### MachineDataQuery
 - ##### SqlBuilder
 - ##### QueryCache
 - ##### SqlImport

### Transformations
 - ##### Transformation
 - ##### TransformationUnion

 - ##### DropNA
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### DropCols
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### DropSparseCols
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### RoundTimestampIndex
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### StandardScaler
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### PCA
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### KMeansLabels
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### PartitionByTime
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### PartitionBy20min
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### FlattenPartitionedTime
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### KinkFinderLabels
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### HdbscanLabels
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### PowerStepSize
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### StepSize
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe

 - ##### ConsensusEnsemble
	 *Description*
	###### Construction Params
	- exporter (Exporter, optional)
	###### Expected Input Dataframe
	###### Output Dataframe


### Exports
 - ##### Exporter

 - ##### pkl_save
 - ##### csv_save
 - ##### csv_save_by_psn

 - ##### csv_psn
 - ##### csv_fields
 - ##### csv_packagemodel_tags

 - ##### csv_partition_stats

 - ##### csv_pca_by_psn
 - ##### csv_pca_eigenvalues
 - ##### png_pca_eigenvalues_as_tags
 - ##### png_pca_eigenvector_scatter_plot
 - ##### png_pca_variance_explained_curve

 - ##### csv_cluster_stats
 - ##### csv_cluster_distribution_by_psn
 - ##### csv_inertia
 - ##### csv_package_similarity
 - ##### png_cluster_grid
 - ##### png_cluster_distribution



### Train
 - ##### init

### Profiler


## Raw Data Import
```python
from TurbineTimeSeries.storage import SqlImporter
importer = SqlImporter(config_path)
importer.import_csvs(data_dir,"my_sql_table")
```