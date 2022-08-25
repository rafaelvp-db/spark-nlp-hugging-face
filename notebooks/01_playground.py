# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Spark NLP Playground
# MAGIC 
# MAGIC In this notebook, we showcase installation and examples for NLP use cases using John Snow Labs' Spark NLP.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Pre-trained model from HF

# COMMAND ----------

from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(f"/tmp/{MODEL_NAME}/tokenizer")
model.save_pretrained(f"/tmp/{MODEL_NAME}/model", saved_model=True)

# COMMAND ----------

ASSET_PATH = f"/tmp/{MODEL_NAME}/model"

# get label2id dictionary
labels = model.config.label2id
# sort the dictionary based on the id
labels = sorted(labels, key=labels.get)

with open(ASSET_PATH + '/labels.txt', 'w') as f:
    f.write('\n'.join(labels))

# COMMAND ----------

!ls {ASSET_PATH}

# COMMAND ----------

# DBTITLE 1,Initializing Spark NLP
import sparknlp
# let's start Spark with Spark NLP
spark = sparknlp.start()

# COMMAND ----------



# COMMAND ----------

from sparknlp.annotator import *
from sparknlp.base import *

sequenceClassifier = BertForSequenceClassification.loadSavedModel(
     '/FileStore/tmp/{}/model/saved_model/1'.format(MODEL_NAME),
     spark
 )\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")\
  .setCaseSensitive(True)\
  .setMaxSentenceLength(128)

# COMMAND ----------

from sparknlp.annotator.classifier_dl.bert_for_sequence_classification import BertForSequenceClassification
from sparknlp import DocumentAssembler
from sparknlp.annotator.token.tokenizer import Tokenizer

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

tokenizer = Tokenizer() \
   .load("/dbfs/tmp/distilbert/tokenizer") \
   .setInputCols("document") \
   .setOutputCol("token")

classifier = BertForSequenceClassification.loadSavedModel("/dbfs/tmp/distilbert/model") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, classifier])

data = spark.createDataFrame([["Aliens have invaded earth"]]).toDF("text")

result = pipeline.fit(data).transform(data)

# COMMAND ----------



# COMMAND ----------


