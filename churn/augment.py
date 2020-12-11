#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# notebook parameters

import os

spark_master = "local[*]"
app_name = "augment"
input_file = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn-.csv")
output_prefix = ""
output_mode = "overwrite"
output_kind = "parquet"
driver_memory = "8g"
executor_memory = "8g"

dup_times = 100


# # Sanity-checking
#
# We're going to make sure we're running with a compatible JVM first — if we run on macOS, we might get one that doesn't work with Scala.

# In[ ]:


from os import getenv


# In[ ]:


getenv("JAVA_HOME")


# # Spark setup

# In[ ]:


import pyspark


# In[ ]:


session = (
    pyspark.sql.SparkSession.builder.master(spark_master)
    .appName(app_name)
    .config("spark.driver.memory", driver_memory)
    .config("spark.executor.memory", executor_memory)
    .getOrCreate()
)
session


# # Schema definition
#
# Most of the fields are strings representing booleans or categoricals, but a few (`tenure`, `MonthlyCharges`, and `TotalCharges`) are numeric.

# In[ ]:


from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pyspark.sql.functions as F

fields = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
double_fields = set(["tenure", "MonthlyCharges", "TotalCharges"])

schema = pyspark.sql.types.StructType(
    [
        pyspark.sql.types.StructField(
            f, DoubleType() if f in double_fields else StringType()
        )
        for f in fields
    ]
)


# In[ ]:


original_df = session.read.csv(input_file, header=True, schema=schema)

original_df = original_df.dropna()
pristine_df = original_df
df = original_df


# In[ ]:


def str_part(seed=0x5CA1AB1E):
    "generate the string part of a unique ID"
    import random

    r = random.Random(seed)

    while True:
        yield "%X" % r.getrandbits(24)


sp = str_part()

if dup_times > 1:
    uniques = session.createDataFrame(
        schema=StructType([StructField("u_value", StringType())]),
        data=[dict(u_value=next(sp)) for _ in range(dup_times)],
    )

    original_df = (
        original_df.crossJoin(uniques.distinct())
        .withColumn("customerID", F.format_string("%s-%s", "customerID", "u_value"))
        .drop("u_value")
    )

    df = original_df


# In[ ]:


original_df.orderBy("customerID").show()


# # Categorical and boolean features

# In[ ]:


columns = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

if False:
    for c in columns:
        print(c, [row[0] for row in original_df.select(c).distinct().rdd.collect()])


# # Splitting the data frame
#
# The training data schema looks like this:
#
# - customerID
# - gender
# - SeniorCitizen
# - Partner
# - Dependents
# - tenure
# - PhoneService
# - MultipleLines
# - InternetService
# - OnlineSecurity
# - OnlineBackup
# - DeviceProtection
# - TechSupport
# - StreamingTV
# - StreamingMovies
# - Contract
# - PaperlessBilling
# - PaymentMethod
# - MonthlyCharges
# - TotalCharges
# - Churn
#
# We want to divide the data frame into several frames that we can join together in an ETL job.
#
# Those frames will look like this:
#
# - **Customer metadata**
#   - customerID
#   - gender
#   - date of birth (we'll derive age and senior citizen status from this)
#   - Partner
#   - Dependents
#   - (nominal) MonthlyCharges
# - **Billing events**
#   - customerID
#   - date (we'll derive tenure from the number/duration of billing events)
#   - kind (one of "AccountCreation", "Charge", or "AccountTermination")
#   - value (either a positive nonzero amount or 0.00; we'll derive TotalCharges from the sum of amounts and Churn from the existence of an AccountTermination event)
# - **Customer phone features**
#   - customerID
#   - feature (one of "PhoneService" or "MultipleLines")
# - **Customer internet features**
#   - customerID
#   - feature (one of "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
#   - value (one of "Fiber", "DSL", "Yes", "No")
# - **Customer account features**
#   - customerID
#   - feature (one of "Contract", "PaperlessBilling", "PaymentMethod")
#   - value (one of "Month-to-month", "One year", "Two year", "No", "Yes", "Credit card (automatic)", "Mailed check", "Bank transfer (automatic)", "Electronic check")

# In[ ]:


original_df.printSchema()


# We'll start by generating a series of monthly charges (in the `charges` data frame), then a series of account creation events (`serviceStarts`) and churn events (`serviceTerminations`).

# In[ ]:


import datetime

now = datetime.datetime.now(datetime.timezone.utc)

w = pyspark.sql.Window.orderBy(F.lit("")).partitionBy(df.customerID)

charges = (
    df.select(
        df.customerID,
        F.lit("Charge").alias("kind"),
        F.explode(
            F.array_repeat(df.TotalCharges / df.tenure, df.tenure.cast("int"))
        ).alias("value"),
    )
    .withColumn("now", F.lit(now))
    .withColumn("month_number", -F.row_number().over(w))
    .withColumn("date", F.expr("add_months(now, month_number)"))
    .drop("now", "month_number")
)

serviceStarts = (
    df.select(
        df.customerID,
        F.lit("AccountCreation").alias("kind"),
        F.lit(0.0).alias("value"),
        F.lit(now).alias("now"),
        (-df.tenure - 1).alias("month_number"),
    )
    .withColumn("date", F.expr("add_months(now, month_number)"))
    .drop("now", "month_number")
)

serviceTerminations = df.where(df.Churn == "Yes").select(
    df.customerID,
    F.lit("AccountTermination").alias("kind"),
    F.lit(0.0).alias("value"),
    F.add_months(F.lit(now), 0).alias("date"),
)


# `billingEvents` is the data frame containing all of these events:  account activation, account termination, and individual payment events.

# In[ ]:


billingEvents = charges.union(serviceStarts).union(serviceTerminations).orderBy("date")


# We'll define a little helper function to use the parameters we defined earlier while writing data frames to Parquet.

# In[ ]:


def write_df(df, name, partition_by=None):
    write = df.write
    if partition_by is not None:
        if type(partition_by) == str:
            partition_by = [partition_by]
        write = write.partitionBy(*partition_by)
    name = "%s.%s" % (name, output_kind)
    if output_prefix != "":
        name = "%s-%s" % (output_prefix, name)
    kwargs = {}
    if output_kind == "csv":
        kwargs["header"] = True
    getattr(write.mode(output_mode), output_kind)(name, **kwargs)


# In[ ]:


write_df(billingEvents, "billing_events", "date")


# Our next step is to generate customer metadata, which includes the following fields:
#
#   - gender
#   - date of birth (we'll derive age and senior citizen status from this)
#   - Partner
#   - Dependents
#
# We'll calculate date of birth by using the hash of the customer ID as a pseudorandom number and then assuming that ages are uniformly distributed between 18-65 and exponentially distributed over 65.

# In[ ]:


SENIOR_CUTOFF = 65
ADULT_CUTOFF = 18
DAYS_IN_YEAR = 365.25
EXPONENTIAL_DIST_SCALE = 6.3

customerMetaRaw = original_df.select(
    "customerID",
    F.lit(now).alias("now"),
    (F.abs(F.hash(original_df.customerID)) % 4096 / 4096).alias("choice"),
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "MonthlyCharges",
)

customerMetaRaw = customerMetaRaw.withColumn(
    "ageInDays",
    F.floor(
        F.when(
            customerMetaRaw.SeniorCitizen == 0,
            (
                customerMetaRaw.choice
                * ((SENIOR_CUTOFF - ADULT_CUTOFF - 1) * DAYS_IN_YEAR)
            )
            + (ADULT_CUTOFF * DAYS_IN_YEAR),
        ).otherwise(
            (SENIOR_CUTOFF * DAYS_IN_YEAR)
            + (
                DAYS_IN_YEAR
                * (-F.log1p(-customerMetaRaw.choice) * EXPONENTIAL_DIST_SCALE)
            )
        )
    ).cast("int"),
)

customerMetaRaw = customerMetaRaw.withColumn(
    "dateOfBirth", F.expr("date_sub(now, ageInDays)")
)

customerMeta = customerMetaRaw.select(
    "customerID",
    "dateOfBirth",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "MonthlyCharges",
    "now",
).orderBy("customerID")


# In[ ]:


write_df(customerMeta, "customer_meta")


# Now we can generate customer phone features, which include:
#
#   - customerID
#   - feature (one of "PhoneService" or "MultipleLines")
#   - value (always "Yes"; there are no records for "No" or "No Phone Service")

# In[ ]:


phoneService = original_df.select(
    "customerID", F.lit("PhoneService").alias("feature"), F.lit("Yes").alias("value")
).where(original_df.PhoneService == "Yes")

multipleLines = original_df.select(
    "customerID", F.lit("MultipleLines").alias("feature"), F.lit("Yes").alias("value")
).where(original_df.MultipleLines == "Yes")

customerPhoneFeatures = phoneService.union(multipleLines).orderBy("customerID")


# In[ ]:


write_df(customerPhoneFeatures, "customer_phone_features")


# Customer internet features include:
#   - customerID
#   - feature (one of "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
#   - value (one of "Fiber", "DSL", "Yes" -- no records for "No" or "No internet service")

# In[ ]:


internet_service = original_df.select(
    "customerID",
    F.lit("InternetService").alias("feature"),
    original_df.InternetService.alias("value"),
).where(original_df.InternetService != "No")

customerInternetFeatures = internet_service

for feature in [
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]:
    tmpdf = original_df.select(
        "customerID",
        F.lit(feature).alias("feature"),
        original_df[feature].alias("value"),
    ).where(original_df[feature] == "Yes")

    customerInternetFeatures = customerInternetFeatures.union(tmpdf)

write_df(customerInternetFeatures.orderBy("customerID"), "customer_internet_features")


# Customer account features include:
#
#   - customerID
#   - feature (one of "Contract", "PaperlessBilling", "PaymentMethod")
#   - value (one of "Month-to-month", "One year", "Two year", "Yes", "Credit card (automatic)", "Mailed check", "Bank transfer (automatic)", "Electronic check")

# In[ ]:


accountSchema = pyspark.sql.types.StructType(
    [
        pyspark.sql.types.StructField(f, StringType())
        for f in ["customerID", "feature", "value"]
    ]
)

customerAccountFeatures = session.createDataFrame(schema=accountSchema, data=[])

for feature in ["Contract", "PaperlessBilling", "PaymentMethod"]:
    tmpdf = original_df.select(
        "customerID",
        F.lit(feature).alias("feature"),
        original_df[feature].alias("value"),
    ).where(original_df[feature] != "No")

    customerAccountFeatures = customerAccountFeatures.union(tmpdf)

write_df(customerAccountFeatures, "customer_account_features")


# In[ ]:
