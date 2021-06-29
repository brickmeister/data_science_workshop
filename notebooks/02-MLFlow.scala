// Databricks notebook source
// MAGIC %md
// MAGIC # Introduction

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Abstract
// MAGIC 
// MAGIC ### Demographic Segmentation
// MAGIC 
// MAGIC Population segmentation is a method widely used to group customers into distinct tiers and brackets. Grouping of these customers into tiers and brackets allow more targetted services to optimize customer spend vs. reward. The following notebook demos a method of segmenting a population using a combination of supervised and unsupervised learning.
// MAGIC 
// MAGIC <img src="https://i2.wp.com/www.iedunote.com/img/30731/4-types-of-market-segmentation.jpg" width="100%" /img>

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Methods
// MAGIC 
// MAGIC Data is first cleaned and featurized using Scala SparkML libraries. Resultant data is passed through two machine learning models (hierarchical bisecting-kmeans, decision tree classifier) in order to segment a dataset.
// MAGIC 
// MAGIC ### Libraries Used
// MAGIC * [Spark DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html)
// MAGIC * [Delta Lake](https://docs.delta.io/latest/delta-intro.html)
// MAGIC * [SparkML](http://spark.apache.org/docs/latest/ml-guide.html)
// MAGIC * [MLFlow](https://www.mlflow.org/docs/latest/index.html)
// MAGIC 
// MAGIC ### Models Used
// MAGIC * [Hierarchical Bisecting-K-Means](https://medium.com/@afrizalfir/bisecting-kmeans-clustering-5bc17603b8a2)
// MAGIC * [Decision Tree Classifier](https://medium.com/swlh/decision-tree-classification-de64fc4d5aac)
// MAGIC 
// MAGIC ### Architecture
// MAGIC <img src="files/shared_uploads/mark.lee@databricks.com/Population_Segmentation__1_.png" width = 100% /img>

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Results
// MAGIC 
// MAGIC SHOW EXAMPLE RESULTS

// COMMAND ----------

// MAGIC %md
// MAGIC # Setup Dataset

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Retrieve Gold Level Data

// COMMAND ----------

// %scala

// /*
//   Load the Databricks Adult income Dataset into a dataframe
// */

// val df_columns = Seq("age", "workclass", "fnlwgt", "education", "education-num", 
//                      "marital-status", "occupation", "relationship", "race", "sex", 
//                      "capital-gain", "capital-loss", "hours-per-week", "native-country",
//                      "label");

// val df = spark.read
//           .format("csv")
//           .load("dbfs:/databricks-datasets/adult/adult.data")
//           .toDF(df_columns : _*);

// display(df);

// COMMAND ----------

// MAGIC %fs
// MAGIC 
// MAGIC ls /databricks-datasets/bikeSharing/data-001

// COMMAND ----------

// DBTITLE 1,Retrieve Data from Delta Lake
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.sql import DataFrame
// MAGIC 
// MAGIC """
// MAGIC   Setup a dataframe to read in data from
// MAGIC   a gold level table  
// MAGIC """
// MAGIC 
// MAGIC df : DataFrame = (spark.read
// MAGIC                        .format("csv")
// MAGIC                        .option("header", True)
// MAGIC                        .option("inferSchema", True)
// MAGIC                        .load("dbfs:/databricks-datasets/bikeSharing/data-001/day.csv"))
// MAGIC 
// MAGIC df.createOrReplaceTempView("bronze_table");

// COMMAND ----------

// DBTITLE 1,Preview the data
// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC   Show the contents of the dataframe
// MAGIC """
// MAGIC 
// MAGIC display(df.limit(100));

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Get the Schema of the Data
// MAGIC 
// MAGIC Retrieve the schema of the data to get an idea as to what data types we are working with. This is used for featurizing the dataset.

// COMMAND ----------

// DBTITLE 1,Get the Schema of the Bronze Table
// MAGIC %sql
// MAGIC 
// MAGIC --
// MAGIC -- Get the schema of the columns
// MAGIC --
// MAGIC 
// MAGIC DESCRIBE BRONZE_TABLE;

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Clean the data

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC from pyspark.sql import DataFrame
// MAGIC from pyspark.sql.functions import col
// MAGIC 
// MAGIC """
// MAGIC   Remove censored data and cast data to proper data types
// MAGIC """
// MAGIC 
// MAGIC # remove censored rows
// MAGIC cleaned_df : DataFrame = (df.na.drop()
// MAGIC                             .withColumn("dteday", col("dteday").cast("timestamp")))
// MAGIC   
// MAGIC cleaned_df.createOrReplaceTempView("silver_table")
// MAGIC 
// MAGIC display(cleaned_df)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Setup Dataframes for ML
// MAGIC 
// MAGIC Dataframes need to be featurized and split into train and test partitions for machine learning.

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Featurize the Dataset

// COMMAND ----------

// DBTITLE 1,Create a ML Dataset
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml.feature import VectorAssembler
// MAGIC from typing import List
// MAGIC 
// MAGIC """
// MAGIC   Setup the dataframes needed for k-means clustering
// MAGIC """
// MAGIC 
// MAGIC feature_columns : List[str] = ["season", "temp", "mnth"]
// MAGIC   
// MAGIC # Generate the feature vectors for ML
// MAGIC assembler = VectorAssembler(inputCols = feature_columns,
// MAGIC                             outputCol = "features");
// MAGIC 
// MAGIC df_dataset : DataFrame = assembler.transform(cleaned_df);
// MAGIC 
// MAGIC # display the dataset for training
// MAGIC display(df_dataset)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Setup the 70:30 train:test split

// COMMAND ----------

// DBTITLE 1,Create the train and test datasets
// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC   Separate the training and testing dataset into two dataframes
// MAGIC """
// MAGIC 
// MAGIC (trainingDF, testingDF) = df_dataset.randomSplit([0.7, 0.3])

// COMMAND ----------

// MAGIC %md
// MAGIC # Model Training
// MAGIC 
// MAGIC For the purposes of this experiment, we will use MLFLOW to persist results and save models

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## K-Means Model Training

// COMMAND ----------

// MAGIC %md
// MAGIC ### Training Function

// COMMAND ----------

// DBTITLE 1,K-Means Training Function
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml.clustering import BisectingKMeans, BisectingKMeansModel
// MAGIC from pyspark.ml.evaluation import ClusteringEvaluator
// MAGIC from pyspark.sql import DataFrame
// MAGIC import mlflow
// MAGIC from typing import Tuple
// MAGIC 
// MAGIC """
// MAGIC   Setup K-Means modeling
// MAGIC """
// MAGIC 
// MAGIC def kMeansTrain(nCentroids : int,
// MAGIC                 seed : int,
// MAGIC                 dataset : DataFrame,
// MAGIC                 featuresCol : str = "features") -> Tuple[BisectingKMeansModel,float]:
// MAGIC   """
// MAGIC     Setup K-Means modeling
// MAGIC     
// MAGIC     @return Trained model
// MAGIC     @return Silhouete with squared euclidean distance 
// MAGIC     
// MAGIC     @param nCentroids   | number of centroids to cluster around
// MAGIC     @param seed          | random number seed
// MAGIC     @param dataset       | Spark DataFrame containing features
// MAGIC     @param featuresCol   | Name of the vectorized column
// MAGIC   """
// MAGIC   
// MAGIC   with mlflow.start_run() as run:
// MAGIC   
// MAGIC     mlflow.log_param("Number_Centroids", str(nCentroids))
// MAGIC     mlflow.log_metric("Training Data Rows", dataset.count())
// MAGIC     mlflow.log_param("seed", str(seed))
// MAGIC 
// MAGIC     ## Start up the bisecting k-means model
// MAGIC     bkm = BisectingKMeans()\
// MAGIC                       .setFeaturesCol(featuresCol)\
// MAGIC                       .setK(nCentroids)\
// MAGIC                       .setSeed(seed)\
// MAGIC                       .setPredictionCol("predictions")
// MAGIC 
// MAGIC     ## Start up the evaluator
// MAGIC     evaluator = ClusteringEvaluator()\
// MAGIC                       .setPredictionCol("predictions")
// MAGIC 
// MAGIC     ## Train a model
// MAGIC     model = bkm.fit(dataset)
// MAGIC 
// MAGIC     ## Make some predictions
// MAGIC     predictions = model.transform(dataset)
// MAGIC 
// MAGIC     ## Evaluate the clusters
// MAGIC     silhouette = evaluator.evaluate(predictions)
// MAGIC 
// MAGIC     ## Log some modeling metrics
// MAGIC     mlflow.log_metric("Silhouette", silhouette)
// MAGIC     mlflow.spark.log_model(model, f"K-Means_{nCentroids}")
// MAGIC 
// MAGIC   
// MAGIC     ## Return the class and silhouette
// MAGIC     return (model, silhouette)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Train the K-Means Model

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Hyperparameter Tuning for Number of Centroids

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC   Tune a K-Means Model
// MAGIC """
// MAGIC 
// MAGIC ## Tune the K Means model by optimizing the number of centroids (hyperparameter tuning)
// MAGIC kMeansTuning = [(i, kMeansTrain(nCentroids = i, dataset = df_dataset, featuresCol = "features", seed = 1)) for i in range(2 ,15, 1)]
// MAGIC 
// MAGIC ## Return the results into a series of arrays
// MAGIC kMeansCosts = [(a[0], a[1][1]) for a in kMeansTuning]

// COMMAND ----------

// MAGIC %md
// MAGIC #### Elbow Plot

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC   Show the efffect of increasing the number of centroids
// MAGIC   for a K Means cluster
// MAGIC """
// MAGIC 
// MAGIC kMeansCostsDF = sc.parallelize(kMeansCosts)\
// MAGIC                   .toDF(["Number of Centroids", "Loss"])
// MAGIC 
// MAGIC display(kMeansCostsDF)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Label Dataset with Clusters

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC   Label the testing and training dataframes with the optimal clustering model
// MAGIC """
// MAGIC 
// MAGIC optimalClusterModel = kMeansTuning[6][1][0]
// MAGIC 
// MAGIC ## label the training and testing dataframes
// MAGIC clusteredtrainingDF = optimalClusterModel.transform(trainingDF)\
// MAGIC                             .withColumnRenamed("predictions", "cluster")
// MAGIC clusteredTestingDF = optimalClusterModel.transform(testingDF)\
// MAGIC                             .withColumnRenamed("predictions", "cluster")
// MAGIC 
// MAGIC clusteredtrainingDF.createOrReplaceTempView("clustered_training_df")
// MAGIC clusteredTestingDF.createOrReplaceTempView("clustered_testing_df")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Visualize Class Balance Between Training and Test Datasets

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC --
// MAGIC -- Check for class imbalance
// MAGIC --
// MAGIC 
// MAGIC SELECT "TRAINING" AS LABEL,
// MAGIC        CLUSTER,
// MAGIC        LOG(COUNT(*)) AS LOG_COUNT
// MAGIC FROM clustered_training_df
// MAGIC GROUP BY CLUSTER
// MAGIC UNION ALL
// MAGIC SELECT "TESTING" AS LABEL,
// MAGIC        CLUSTER,
// MAGIC        LOG(COUNT(*)) AS LOG_COUNT
// MAGIC FROM clustered_testing_df
// MAGIC GROUP BY CLUSTER
// MAGIC ORDER BY CLUSTER ASC;

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Decision Tree Model Training

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC Decision Tree Function
// MAGIC """
// MAGIC 
// MAGIC import mlflow
// MAGIC from pyspark.ml.classification import DecisionTreeClassifier
// MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
// MAGIC from pyspark.sql import DataFrame
// MAGIC from typing import Tuple
// MAGIC 
// MAGIC def dtcTrain(p_max_depth : int,
// MAGIC              training_data : DataFrame,
// MAGIC              test_data : DataFrame,
// MAGIC              seed : int,
// MAGIC              featuresCol : str,
// MAGIC              labelCol : str) -> Tuple[int, float]:
// MAGIC   with mlflow.start_run() as run:
// MAGIC     # log some parameters
// MAGIC     mlflow.log_param("Maximum_depth", p_max_depth)
// MAGIC     mlflow.log_metric("Training Data Rows", training_data.count())
// MAGIC     mlflow.log_metric("Test Data Rows", test_data.count())
// MAGIC     
// MAGIC     # start the decision tree classifier
// MAGIC     dtc = DecisionTreeClassifier()\
// MAGIC                           .setFeaturesCol(featuresCol)\
// MAGIC                           .setLabelCol(labelCol)\
// MAGIC                           .setMaxDepth(p_max_depth)\
// MAGIC                           .setSeed(seed)\
// MAGIC                           .setPredictionCol("predictions")\
// MAGIC                           .setMaxBins(4000)
// MAGIC     
// MAGIC     # Start up the evaluator
// MAGIC     evaluator = MulticlassClassificationEvaluator()\
// MAGIC                       .setLabelCol("cluster")\
// MAGIC                       .setPredictionCol("predictions")
// MAGIC 
// MAGIC     # Train a model
// MAGIC     model = dtc.fit(training_data)
// MAGIC 
// MAGIC     # Make some predictions
// MAGIC     predictions = model.transform(test_data)
// MAGIC 
// MAGIC     # Evaluate the tree
// MAGIC     silhouette = evaluator.evaluate(predictions)
// MAGIC     
// MAGIC     # Log the accuracy
// MAGIC     mlflow.log_metric("F1", silhouette)
// MAGIC     
// MAGIC     # Log the feature importances
// MAGIC     mlflow.log_param("Feature Importances", model.featureImportances)
// MAGIC     
// MAGIC     # Log the model
// MAGIC     mlflow.spark.log_model(model, f"Decision_tree_{p_max_depth}")
// MAGIC     
// MAGIC     ## Return the class and silhouette
// MAGIC     return (model, silhouette)

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC Tune the max depth of the Decision tree
// MAGIC """
// MAGIC 
// MAGIC dtcTuning = [(i, dtcTrain(p_max_depth = i,
// MAGIC                           training_data = clusteredtrainingDF,
// MAGIC                           test_data = clusteredTestingDF,
// MAGIC                           seed = 1,
// MAGIC                           featuresCol = "features",
// MAGIC                           labelCol = "cluster"))
// MAGIC               for i in range(2, 15, 1)]
// MAGIC 
// MAGIC ## Return the results into a series of arrays
// MAGIC dtcF1 = [(a[0], a[1][1]) for a in dtcTuning]

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Elbow Plot

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC   Show the efffect of increasing the max depth
// MAGIC   for a Decision Tree
// MAGIC """
// MAGIC 
// MAGIC dtcF1DF = sc.parallelize(dtcF1)\
// MAGIC                       .toDF()\
// MAGIC                       .withColumnRenamed("_1", "Max Depth")\
// MAGIC                       .withColumnRenamed("_2", "F1")
// MAGIC 
// MAGIC display(dtcF1DF)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Training with a Grid Search

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC Turn on MLFlow autlogging
// MAGIC """
// MAGIC 
// MAGIC mlflow.autolog(log_models=True, exclusive=True)

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC --
// MAGIC -- Setup MLFlow Tracking
// MAGIC --
// MAGIC 
// MAGIC set spark.databricks.mlflow.trackMLlib.enabled=true

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC """
// MAGIC Visualize the optimal decision tree
// MAGIC """
// MAGIC 
// MAGIC display(dtcTuning[4][1][0])

// COMMAND ----------

// DBTITLE 1,Decision Tree Training With Grid Search
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.ml import Pipeline
// MAGIC from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
// MAGIC from pyspark.ml.classification import DecisionTreeClassifier
// MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
// MAGIC 
// MAGIC """
// MAGIC Do a cross validation of the decision tree model
// MAGIC """
// MAGIC 
// MAGIC # Set the decision tree that will be optimized
// MAGIC dt = DecisionTreeClassifier()\
// MAGIC             .setFeaturesCol("features")\
// MAGIC             .setLabelCol("cluster")\
// MAGIC             .setSeed(1)\
// MAGIC             .setPredictionCol("predictions")\
// MAGIC             .setMaxBins(4000)
// MAGIC 
// MAGIC # Build the grid of different parameters
// MAGIC paramGrid = ParamGridBuilder() \
// MAGIC     .addGrid(dt.maxDepth, range(1,9,1)) \
// MAGIC     .addGrid(dt.maxBins, [4000, 5000, 6000]) \
// MAGIC     .build()
// MAGIC 
// MAGIC # Generate an average F1 score for each prediction
// MAGIC evaluator = MulticlassClassificationEvaluator()\
// MAGIC                   .setLabelCol("cluster")\
// MAGIC                   .setPredictionCol("predictions")
// MAGIC 
// MAGIC # Build out the cross validation
// MAGIC crossval = CrossValidator(estimator = dt,
// MAGIC                           estimatorParamMaps = paramGrid,
// MAGIC                           evaluator = evaluator,
// MAGIC                           numFolds = 3)  
// MAGIC pipelineCV = Pipeline(stages=[crossval])
// MAGIC 
// MAGIC # Train the model using the pipeline, parameter grid, and preceding BinaryClassificationEvaluator
// MAGIC cvModel_u = pipelineCV.fit(clusteredtrainingDF)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC # Deploy Model

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Register the model with MLFlow registry

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC import mlflow
// MAGIC from mlflow.models import ModelSignature
// MAGIC 
// MAGIC """
// MAGIC Log the mlflow model to the registry
// MAGIC """
// MAGIC 
// MAGIC model_version_major = 1
// MAGIC model_version_minor = 1
// MAGIC 
// MAGIC with mlflow.start_run() as run:
// MAGIC   # get the dataframe signature for the model
// MAGIC   _signature = mlflow.models.infer_signature(cleaned_df,
// MAGIC                                              clusteredtrainingDF.drop("features"))
// MAGIC   # Log the model
// MAGIC   mlflow.spark.log_model(spark_model = dtcTuning[4][1][0],
// MAGIC                          signature = _signature,
// MAGIC                          registered_model_name = "bike_segmentation",
// MAGIC                          artifact_path = f"pipeline_model_v{model_version_major}.{model_version_minor}"
// MAGIC                         )
