from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import math
import numpy
from subprocess import call
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("TribeSegementaion_Mapping") \
        .getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "50")
    ## Reading all the required process variables for TribeSegmentation from HIVE table processController
    processController = spark.sql("select * from dev_db.processdriver where processName='TribeSegmentation'")
    processController.persist()
    distillTable = processController.where("VariableName='distillTable'").select("Value").collect()[0][0]
    numOfClusters = int(processController.where("VariableName='numOfClusters'").select("Value").collect()[0][0])
    kmeansMaxIter = int(processController.where("VariableName='kmeansMaxIter'").select("Value").collect()[0][0])
    kmeansSeed = int(processController.where("VariableName='kmeansSeed'").select("Value").collect()[0][0])

    processController.unpersist()
    ## Reading the KMEANS clustering variable list from HIVE table
    kmeansVar = spark.sql("select * from dev_db.dev_kmeans_variables").collect()[0][0]

    ## Reading the cleansed data for Tribe calculation from HIVE table
    #customerData = spark.sql("select * from dev_db.TribeSegData")
    customerData = spark.sql("select * from %s " % (distillTable))
    customerData.persist()

    ## Run Kmeans with selected variables.
    vectorAssembler = VectorAssembler(inputCols=kmeansVar, outputCol="features")
    vdf_exp = vectorAssembler.transform(customerData)
    kmeans = KMeans(k=numOfClusters, maxIter=kmeansMaxIter, seed=kmeansSeed)
    model = kmeans.fit(vdf_exp)

    ## Check how our records are distributed across the five clusters, if not as expected process exits
    predictionValues = list(model.transform(vdf_exp).select("prediction").rdd.countByValue().values())
    dataCount = customerData.count()
    max_percent = numpy.amax(predictionValues) / dataCount
    min_percent = numpy.amin(predictionValues) / dataCount
    if max_percent > 0.85 or min_percent < 0.05:
        print("The distribution of clusters are not healthy, exiting the process")
        appId = spark.sparkContext.applicationId
        call(["yarn", "application", "-kill", appId])

    ## Mapping the clusters to four main groups. We will get None_index, Exp_index, Ba_index, Re_index and Lux_index for mapping clusters to tribes.

    centers = model.clusterCenters()
    dist_to_none = [numpy.linalg.norm([0, 0] - centers[x][0:2]) for x in range(numOfClusters)]
    None_index = numpy.argmin(dist_to_none)
    mean_in_outdoors = [centers[x][0] for x in range(numOfClusters)]
    Exp_index = numpy.argmax(mean_in_outdoors)
    Ba_index = mean_in_outdoors.index(
        numpy.amin(numpy.array(mean_in_outdoors)[mean_in_outdoors != mean_in_outdoors[None_index]]))
    mean_in_luxurist = [centers[x][1] for x in range(numOfClusters)]
    left_index = [x for x in range(numOfClusters) if x not in [None_index, Explorer_index, Bask_index]]
    Lux_index = mean_in_luxurist.index(numpy.amax([mean_in_luxurist[i] for i in left_index]))
    Re_index = [x for x in left_index if x != Luxurist_index][0]

    ## Replace prediction with Tribe Names
    df = model.transform(vdf_exp)
    df = df.withColumn("Group", F.when(df["prediction"] == int(Exp_index), "Exp") \
                       .when(df["prediction"] == int(Ba_index), "Ba").when(df["prediction"] == int(Lux_index),
                                                                               "Lux") \
                       .when(df["prediction"] == int(Re_index), "Re").otherwise("None")).drop("prediction").drop(
        "features")

    ## Save the results in hive as: ds_db.dev_MainClusters

    df.createOrReplaceTempView("tmp")
    spark.sql("drop table IF EXISTS ds_db.dev_MainClusters")
    spark.sql("create table ds_db.dev_MainClusters as select * from tmp")

    spark.catalog.dropTempView("tmp")
    customerData.unpersist()
    spark.stop()
