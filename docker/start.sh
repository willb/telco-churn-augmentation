#!/bin/bash

set -x

export JARPATH=$(realpath $(dirname $0))



if [[ -v LIGHTRUN ]]; then
    DUP_TIMES="--dup-times 50"
else
    DUP_TIMES="--dup-times 500"
fi

mkdir tmp
export SPARK_TEMP=$(pwd)/tmp

pipenv run spark-submit --driver-memory 32G --master="spark://localhost:7077" --conf spark.eventLog.enabled=True --conf spark.eventLog.dir=${SPARK_TEMP} --conf spark.default.parallelism=512 --conf spark.sql.analyzer.failAmbiguousSelfJoin=false --conf spark.local.dir=${SPARK_TEMP} ./generate.py ${DUP_TIMES}
pipenv run spark-submit --driver-memory 32G --master="spark://localhost:7077" --conf spark.eventLog.enabled=True --conf spark.eventLog.dir=${SPARK_TEMP} --conf spark.sql.adaptive.enabled=true --conf spark.rapids.sql.batchSizeBytes=1073741824 --conf spark.rapids.sql.concurrentGpuTasks=2 --conf spark.default.parallelism=256 --conf spark.rapids.sql.explain=NOT_ON_GPU --conf spark.rapids.sql.decimalType.enabled=true --conf spark.rapids.sql.variableFloatAgg.enabled=true --conf spark.local.dir=${SPARK_TEMP} ./do-analytics.py --log-level WARN
