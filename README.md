# Introduction : Databricks Data Science Workshop

Welcome to the repository for the Databricks Data Science Workshop

This repository contains the notebooks that are used in the workshop to demonstrate the use of different Databricks tools in a Data Science environment.

![ML](https://databricks.com/wp-content/uploads/2021/05/data-foundation-for-the-full-ml-lifecycle-2.png)

- [Introduction : Databricks Data Science Workshop](#introduction--databricks-data-science-workshop)
- [Reading Resources](#reading-resources)
- [Workshop Flow](#workshop-flow)
- [Setup / Requirements](#setup--requirements)
  - [DBR Version](#dbr-version)
  - [Repos](#repos)
  - [DBC Archive](#dbc-archive)

# Reading Resources

* [Lakehouse Whitepaper](https://databricks.com/wp-content/uploads/2020/12/cidr_lakehouse.pdf)
* [Datascience Jumpstart](https://pages.databricks.com/Making-Machine-Learning-Simple.html)

# Workshop Flow

The workshop consists of 4 interactive sections that are separated by 4 notebooks located in the notebooks folder in this repository. Each is run sequentially as we explore the abilities of the Databricks Data Science platform from Pandas integration to distributed modeling, and CI/CD workflows.

|Notebook|Summary|
|--------|-------|
|`01-Koalas.py`|Leveraging Koalas for Pandas workloads on Databricks|
|`02-MLFlow.scala`|Designing a Segmentation Forecaster on Databricks|
|`03-Cluster Optimization.py`|Debugging and understanding cluster performance|
|`04-Git Integration.py`|CI/CD Workflow for Databricks Repos|

# Setup / Requirements

This workshop requires a running Databricks workspace. If you are an existing Databricks customer, you can use your existing Databricks workspace. Otherwise, the notebooks in this workshop have been tested to run on [Databricks Community Edition](https://databricks.com/product/faq/community-edition) as well.

## DBR Version

The features used in this workshop require `DBR 8.3 ML`.

## Repos

If you have repos enabled on your Databricks workspace. You can directly import this repo and run the notebooks as is and avoid the DBC archive step.

## DBC Archive

Download the DBC archive from releases and import the archive into your Databricks workspace.