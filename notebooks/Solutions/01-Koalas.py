# Databricks notebook source
# DBTITLE 1,Let's use the Bike Sharing Data as an example
# MAGIC %fs ls
# MAGIC /databricks-datasets/bikeSharing/

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook will - for the most part - follow the pattern of:
# MAGIC - 'Here's how do to **X** in pandas'
# MAGIC - 'Here's how to do **X** in Koalas'

# COMMAND ----------

# DBTITLE 1,Reading Data in Pandas
import pandas as pd

pandas_bike = pd.read_csv('/dbfs/databricks-datasets/bikeSharing/data-001/day.csv')

pandas_bike.head(15)

# COMMAND ----------

# DBTITLE 1,To install Koalas
# Option 1: RECOMMENDED Add to the cluster libraries manually
# Option 2: If using Conda+MLR, run the below: 
# %conda install -c conda-forge koalas
# Option 3: If not using Machine Learning Runtime:
# dbutils.library.installPyPI("koalas")

# COMMAND ----------

# DBTITLE 1,Reading Data in Koalas
import databricks.koalas as ks

koalas_bike = ks.read_csv('/databricks-datasets/bikeSharing/data-001/day.csv')

koalas_bike.head(15)

# COMMAND ----------

# DBTITLE 1,Getting Summary Statistics in Pandas
pandas_bike.describe()

# COMMAND ----------

# DBTITLE 1,Getting Summary Statistics in Koalas 
koalas_bike.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC You'll notice that calling `describe()` in Koalas kicks off a couple spark jobs - again this is because the data frame is distributed. Because the aggregates required for the `describe()` output require passing data across nodes, this creates stage boundaries.

# COMMAND ----------

# DBTITLE 1,Sorting by Columns in Pandas
pandas_bike.sort_values(by='temp').head()

# COMMAND ----------

# DBTITLE 1,Sorting by Columns in Koalas 
koalas_bike.sort_values(by='temp').head()

# COMMAND ----------

# DBTITLE 1,Transposing Data in Pandas
pandas_bike.transpose().head()

# COMMAND ----------

# DBTITLE 1,Transposing Data in Koalas 
koalas_bike[['season','holiday']].transpose()

# COMMAND ----------

# MAGIC %md 
# MAGIC **Best Practice** DataFrame.transpose() will fail when the number of rows is more than the value of compute.max_rows, which is set to 1000 by default. This is to prevent users from unknowingly executing expensive operations. In Koalas, you can easily reset the default compute.max_rows.

# COMMAND ----------

ks.get_option('compute.max_rows')

# COMMAND ----------

ks.set_option('compute.max_rows',2000)

# COMMAND ----------

ks.get_option('compute.max_rows')

# COMMAND ----------

# DBTITLE 1,Selecting Data in Pandas
pandas_bike['season']

# COMMAND ----------

koalas_bike['season']

# COMMAND ----------

# MAGIC %md
# MAGIC Note that when selecting a single row using Koalas, it returns a series. 

# COMMAND ----------

type(koalas_bike['season'])

# COMMAND ----------

# MAGIC %md
# MAGIC However, when multiple columns are selected, it returns a dataframe

# COMMAND ----------

type(koalas_bike[['temp','season']])

# COMMAND ----------

# DBTITLE 1,Slicing Rows and Columns in Pandas
pandas_bike.iloc[:2,:4]

# COMMAND ----------

# DBTITLE 1,Slicing Rows and Columns in Koalas 
koalas_bike.iloc[:2, :4]

# COMMAND ----------

# MAGIC %md
# MAGIC **Best Practice:** By default, Koalas disallows adding columns coming from different DataFrames or Series to a Koalas DataFrame as adding columns requires join operations which are generally expensive. This operation can be enabled by setting compute.ops_on_diff_frames to True, but this could affect performance

# COMMAND ----------

ks.set_option("compute.ops_on_diff_frames", True)

# COMMAND ----------

ks.get_option('compute.ops_on_diff_frames')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 1:
# MAGIC Using Koalas please do the following
# MAGIC - Read in the data from the `databricks-datasets/definitive-guide/data/retail-data/all/online-retail-dataset.csv` path
# MAGIC - Filter the Koalas DF such that the values fall between the 1st and 3rd quantiles for the `UnitPrice` column
# MAGIC - Sort the resulting DF by `Quantity`

# COMMAND ----------

path = 'databricks-datasets/definitive-guide/data/retail-data/all/online-retail-dataset.csv'

# Read in the data
_df1 = ks.read_csv('dbfs:/'+path)

# Get the quantile values 
_df1.quantile([0.0, 0.25, 0.50, 0.75, 1.00])

#Filter so all observations are between 1st and 3rd quantile
_df1[(_df1.UnitPrice >= 1.25) & (_df1.UnitPrice <= 4.13)]

# COMMAND ----------

# DBTITLE 1,Applying a Function to a Dataframe in Pandas
import numpy as np
pandas_bike[['temp', 'season']].apply(np.cumsum).head()

# COMMAND ----------

# DBTITLE 1,Applying a Function to a Dataframe in Koalas 
koalas_bike[['temp', 'season']].apply(np.cumsum).head()

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the default index is zero - however, the `index` can be set. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Best Practices** It's always a good idea to specify the return type hint for for Spark's return type internally when applying a UDF to a Koalas Dataframe. If the return type hint is not specified, Koalas runs the function once for a small sample to infer the Spark return type which can be fairly expensive

# COMMAND ----------

# MAGIC %md
# MAGIC Note that global `apply` in Koalas doesn't support global aggs. This is by design. However, you can use the `computer_shortcut` limit to get around this limitation if data is small enough

# COMMAND ----------

# DBTITLE 1,Grouping Data in Pandas
pandas_bike.groupby('mnth').head(5)

# COMMAND ----------

koalas_bike.groupby('mnth').head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Beyond data manipulation, Koalas aslo has code coverage for visual functions as well. 

# COMMAND ----------

# DBTITLE 1,Building a Line Chart in Pandas
display(pandas_bike.plot.line(x='dteday',y='temp'))

# COMMAND ----------

# DBTITLE 1,Building a Line Chart in Koalas
display(koalas_bike.plot.line(x='dteday',y='temp'))

# COMMAND ----------

# DBTITLE 1,Building a Scatter Plot with Colormap in Pandas
display(pandas_bike.plot.scatter(x='temp', y='windspeed',c='weathersit', colormap='gist_heat'))

# COMMAND ----------

# DBTITLE 1,Building a Scatter Plot with Colormap in Koalas
display(koalas_bike.plot.scatter(x='temp', y='windspeed',c='weathersit'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 2:
# MAGIC Let's build a plot that uses the retail data from **Exercise 1** to look at the number of products sold per season. Please use Koalas to do the following:
# MAGIC - Group by the season
# MAGIC - Count the number of products
# MAGIC - Create a bar chart that displays the item description X quantity

# COMMAND ----------

# Group the dataframe, sum by group, sort by total items
_df2 = koalas_bike.groupby("season", as_index = False).sum()[["season", "cnt"]].sort_values(by="cnt", ascending = True).head(20)

# Make the bar chart
display(_df2.plot.bar(x="season", y = "cnt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## When and Why to convert across Pandas and Koalas 
# MAGIC Because Koalas only have about 70% converage of Pandas - not to mention some pandas operations fundamentally aren't able to be distributed - the workflow for implementing functions that exist in Pandas and *not* in Koalas and vice versa is to use the `to_Pandas()` and `to_Koalas` syntax

# COMMAND ----------

# DBTITLE 1,Converting a Pandas Dataframe to a Koalas Dataframe
convert = ks.from_pandas(pandas_bike)
type(convert)

# COMMAND ----------

# DBTITLE 1,Converting a Koalas Dataframe to a Pandas Dataframe
convert = koalas_bike.to_pandas()
type(convert)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: This will collect all your data on the driver. If the data is larger than the amount of memory on the driver, this will return an *Out of Memory* error

# COMMAND ----------

# DBTITLE 1,Converting to an Index/Column to a List in Pandas
pandas_bike.index.to_list()

# COMMAND ----------

koalas_bike.index.to_list()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Natively Supporting Pandas Objects 

# COMMAND ----------

# Adding timestamp column to pandas df
pandas_bike['timestamp'] = pd.Timestamp('19960524')
koalas_with_timestamp = ks.from_pandas(pandas_bike)

# Quickly view the data 
koalas_with_timestamp.head()

# COMMAND ----------

# Check that it's a Koalas DF
type(koalas_with_timestamp)

# COMMAND ----------

# MAGIC %md
# MAGIC Koalas has Koalas specific functions that support distributing a pandas function across a Koalas dataframe

# COMMAND ----------

# DBTITLE 1,Distributing a Pandas Function in Koalas 
date_range = pd.date_range('1996-05-24', periods=731, freq='1D1min')
kdf = ks.DataFrame({'Test': ["timestamp"]}, index = date_range)
kdf.dtypes

# COMMAND ----------

kdf.map_in_pandas(func=lambda pdf: pdf.between_time('0:15', '0:16'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using SQL with Koalas 

# COMMAND ----------

ks.sql('select * from {koalas_bike} where weekday = 6').head()

# COMMAND ----------

# DBTITLE 1,Using SQL to join a Koalas Dataframe to a Pandas Dataframe
ks.sql('SELECT ks.temp, pd.atemp FROM {koalas_bike} ks INNER JOIN {pandas_bike} pd ON ks.instant = pd.instant ORDER BY ks.temp, pd.atemp').head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3: 
# MAGIC Recreate the solution from **Exercise 2** using Spark SQL and Koalas

# COMMAND ----------

# Create the DF from the SQL query
_df3 = ks.sql("SELECT season, SUM(CNT) AS cnt FROM {koalas_bike} GROUP BY SEASON")
# Plot using Koalas 
display(_df3.plot.bar(x="season", y = "cnt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Working with Pyspark in Koalas 

# COMMAND ----------

# DBTITLE 1,Converting a Koalas Dataframe to a Pyspark Dataframe
spark_df = koalas_bike.to_spark()
type(spark_df)

# COMMAND ----------

# DBTITLE 1,Converting a Spark Dataframe to a Koalas Dataframe
koalas_bike_from_spark = spark_df.to_koalas()
type(koalas_bike_from_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the conversion from a Spark dataframe to a Koalas dataframe can cause an OOM error if the default index is of type `sequence`. You can change the index by using the `compute.default_index_type (default = sequence)`. However, if the index must be a sequence you should use a distributed sequence

# COMMAND ----------

# MAGIC %md
# MAGIC **Best Practice**: Best Practice: Converting from a PySpark DataFrame to Koalas DataFrame can have some overhead because it requires creating a new default index internally – PySpark DataFrames do not have indices. You can avoid this overhead by specifying the column that can be used as an index column. See the Default Index type for more detail.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Koalas to check the Spark Execution plan

# COMMAND ----------

koalas_bike.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC **Best Practice**: Using the `explain()` function can be really useful to optimize your spark code 

# COMMAND ----------

# DBTITLE 1,Caching a Dataframe in Koalas 
cache_df = koalas_bike.loc[koalas_bike['cnt']>850]
cache_df.cache()
cache_df.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC Note you can use `unpersist()` to remove your dataframe from cached memory
