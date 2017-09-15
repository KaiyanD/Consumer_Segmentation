from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("TribeSegmentation_DataPrep") \
        .getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "50")
    ## reading the Tribe Segmentation process variables from HIVE table
    processController = spark.sql("select * from dev_db.processdriver where processName='TribeSegmentation'")
    processController.persist()
    ## preparing the select expression
    selectedVar = processController.where("VariableName='selectedVar'").select("Value").collect()[0][0].split("~")
    selectExpr = list(
        map(lambda x: "cast(case when " + x + " is NULL then 0 else " + x + " end as float) as " + x, selectedVar))
    ## reading the filter conditions from processController
    queryTable = processController.where("VariableName='qryTable'").select("Value").collect()[0][0]
    distillTable = processController.where("VariableName='distillTable'").select("Value").collect()[0][0]
    gIncome = processController.where("VariableName='gIncome'").select("Value").collect()[0][0]
    ageLow = int(processController.where("VariableName='ageLow'").select("Value").collect()[0][0])
    ageHigh = int(processController.where("VariableName='ageHigh'").select("Value").collect()[0][0])

    ## reading the data from HIVE with filters, cast statements and fill NULL with zeroes
    #customerData = spark.sql("select %s from dev_db.customer_data where gincome not in ('1','2','A','B','') AND age_txt BETWEEN %s And %s" % (        ','.join(selectExpr), ageLow, ageHigh))
    customerData = spark.sql(
        "select %s from %s where gincome not in %s AND age_txt BETWEEN %s And %s" % (
            ','.join(selectExpr), queryTable, gIncome, ageLow, ageHigh))
    customerData.createOrReplaceTempView("customerData")
    #spark.sql("drop table if exists dev_db.TribeSegData")
    #spark.sql("create table dev_db.TribeSegData stored as parquet as select * from customerData")
    spark.sql("drop table if exists %s " % (distillTable))
    spark.sql("create table %s stored as parquet as select * from customerData" % (distillTable))
    processController.unpersist()
    spark.stop()
