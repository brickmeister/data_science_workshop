# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Repos Git Integration
# MAGIC 
# MAGIC The Databricks [Repos](https://docs.databricks.com/repos.html) product is an integral part of the Databricks vision for CI/CD moving forwards. By allowing Databricks users to import whole Git repos into their workspaces; users can leverage common CI/CD git workflows like [Gitlab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html).
# MAGIC 
# MAGIC ## Workflow
# MAGIC <img src='https://docs.databricks.com/_images/repos-best-practices.png' /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Acessing Repos
# MAGIC 
# MAGIC To add a repo, go to your sidebar and click the Repos icon. There should be a button named "Add Repo"
# MAGIC 
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/add-repo.png' /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Clone a Repo
# MAGIC 
# MAGIC Upon clicking "Add Repo", a floating dialog should appear asking you how you want to access the repo. If the repo is behind a private proxy or SSO, one will need to use git token integration to access the repo.
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/clone-from-repo.png' /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Checkout / Create a Branch
# MAGIC 
# MAGIC Checking out or creating a new branch involes clicking the branch icon and specifying which branch you want to pull.
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/git-dialog-new-branch.png' /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Submitting Code to a branch
# MAGIC 
# MAGIC Once notebooks are updated with the the proper code, developers are able to push to branches in a git repo they have access to.
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/git-commit-push.png' /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Create a Pull Request on your Git Provider
# MAGIC 
# MAGIC After submitting to a feature branch, developers can use their git providers to create pull requests for merging their branch into the `dev`,`qa`, `prod` branches. See below for an example on Github.
# MAGIC 
# MAGIC <img src='https://docs.github.com/assets/images/help/pull_requests/pull-request-review-edit-branch.png' /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Pulling in new changes to a branch
# MAGIC 
# MAGIC After merging in changes, developers can pull in changes for certain repos by using pressing the Pull button.
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/git-dialog-settings.png' /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Job Automation
# MAGIC 
# MAGIC The last step in the CI/CD process involves scheduling jobs off of protected git branches. These are set with our top level repos features, which allows admins to set read-only git repositories in repos.
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/top-level-repo-folders.png' /img>
