from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.window import Window
import math
import numpy
import sys
from subprocess import call
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, SQLContext


def goodDensityVarSelection(data, attributes, varSelMethod, densityLow, densityHigh):
    describe = data.describe()
    filter = 'summary ="' + varSelMethod + '"'
    result = describe.where(filter).select(*attributes).rdd.collect()
    resultset = [float(x) for x in list(result[0])]
    goodDensityVar = [i for i, j in zip(attributes, resultset) if j > densityLow and j < densityHigh]
    return goodDensityVar


def stepwiseKmeansVarSelection(data, numVarToSelect, goodDensityVar, seedVar, numOfClusters, kmeansMaxIter, kmeansSeed):
    # Start KMEANS with seed variables to get a list of variables which forms best-separated clusters
    attr_best = "none"
    for i in range(numVarToSelect):
        try_ls = [item for item in goodDensityVar if item not in seedVar]
        wssse_best = float('inf')
        for attribute in try_ls:
            test_var = list(seedVar)
            test_var.append(attribute)
            vectorAssembler = VectorAssembler(inputCols=test_var, outputCol="features")
            vdf_exp = vectorAssembler.transform(data)
            kmeans = KMeans(k=numOfClusters, maxIter=kmeansMaxIter, seed=kmeansSeed)
            model = kmeans.fit(vdf_exp)
            wssse = model.computeCost(vdf_exp)
            if wssse < wssse_best:
                wssse_best = wssse
                attr_best = attribute
        seedVar.append(attr_best)


if __name__ == "__main__":
    '''
    spark = SparkSession \
        .builder \
        .appName("IdentifyingTribeSegClusterVariables") \
        .getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "50")
    '''
    conf = SparkConf().setAppName("defVar").setMaster("yarn").set("spark.sql.shuffle.partitions", "50")
    sc = SparkContext(conf=conf)
    spark = HiveContext(sc)

    ## Reading all the process variables for TribeSegmentation from HIVE table processController
    processController = spark.sql("select * from dev_db.processdriver where processName='TribeSegmentation'")
    processController.persist()
    distillTable = processController.where("VariableName='distillTable'").select("Value").collect()[0][0]
    densityLow = float(processController.where("VariableName='densityLow'").select("Value").collect()[0][0])
    densityHigh = float(processController.where("VariableName='densityHigh'").select("Value").collect()[0][0])
    seedVar = processController.where("VariableName='seedVar'").select("Value").collect()[0][0].split("~")
    numVarToSelect = int(processController.where("VariableName='numVarToSelect'").select("Value").collect()[0][0])
    numOfClusters = int(processController.where("VariableName='numOfClusters'").select("Value").collect()[0][0])
    kmeansMaxIter = int(processController.where("VariableName='kmeansMaxIter'").select("Value").collect()[0][0])
    kmeansSeed = int(processController.where("VariableName='kmeansSeed'").select("Value").collect()[0][0])

    ## Reading the cleansed data for Tribe calculation from HIVE table
    #customerData = spark.sql("select * from dev_db.TribeSegData")
    customerData = spark.sql("select * from %s " % (distillTable))
    customerData.persist()

    ## Finding the list of variable which meets the density threshold
    attributes = customerData.columns
    try:
        attributes.remove("sequence")
    except:
        pass
    goodDensityVar = goodDensityVarSelection(customerData, attributes, "mean", densityLow, densityHigh)

    ## Check quality of density variables and stop the job if necessary
    if len(goodDensityVar) >= 20:
        print("You got a list of " + str(len(goodDensityVar)) + " variables : " + ",".join(goodDensityVar))
    elif len(goodDensityVar) < 20:
        print("You have only got a list of " + str(
            len(goodDensityVar)) + " variables, something might be wrong, exiting process: " + ",".join(goodDensityVar))
        appId = spark.sparkContext.applicationId
        call(["yarn", "application", "-kill", appId])

    ## Finding the list of final set of seed variables which forms best clusters
    stepwiseKmeansVarSelection(customerData, numVarToSelect, goodDensityVar, seedVar, numOfClusters, kmeansMaxIter,
                               kmeansSeed)

    ## Saving the final KMEANS clustering variable list to HIVE
    df = spark.createDataFrame(Row([seedVar]), StructType([StructField("columns", ArrayType(StringType()))]))
    df.createOrReplaceTempView("tmp")
    spark.sql("drop table if exists dev_db.dev_kmeans_variables")
    spark.sql("create table dev_db.dev_kmeans_variables as select * from tmp")

    ## Unpersisting data from memory
    spark.catalog.dropTempView("tmp")
    processController.unpersist()
    customerData.unpersist()

    spark.stop()
