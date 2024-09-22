#!/bin/bash

function F_USAGE
{
   echo "USAGE: ${1##*/} -M '<Mode>'"
   echo "USAGE: ${1##*/} -A '<Application>'"
   echo "USAGE: ${1##*/} -P '<SP>'"
   echo "USAGE: ${1##*/} -H '<HELP>' -h '<HELP>'"
   exit 10
}
#
# Main Section
#
if [[ "${1}" = "-h" || "${1}" = "-H" ]]; then
   F_USAGE $0
fi
## MAP INPUT TO VARIABLES
while getopts M:A:P: opt
do
   case $opt in
   (M) MODE="$OPTARG" ;;
   (A) APPLICATION="$OPTARG" ;;
   (P) SP="$OPTARG" ;;
   (*) F_USAGE $0 ;;
   esac
done

[[ -z ${MODE} ]] && echo "You must specify a run mode: Local, Standalone, k8s or Yarn or gcp" && F_USAGE $0
MODE=`echo ${MODE}|tr "[:upper:]" "[:lower:]"`
if [[ "${MODE}" != "local" ]] && [[ "${MODE}" != "standalone" ]] && [[ "${MODE}" != "yarn" ]] && [[ "${MODE}" != "k8s" ]] && [[ "${MODE}" != "docker" ]] && [[ "${MODE}" != "gcp" ]]
then
        echo "Incorrect value for build mode. The run mode can only be local, standalone, docker, k8s or yarn or gcp"  && F_USAGE $0
fi
[[ -z ${APPLICATION} ]] && echo "You must specify an application value " && F_USAGE $0
#
if [[ -z ${SP} ]]
then
        export SP=55555
fi

ENVFILE=/home/hduser/dba/bin/environment.ksh

#set -e
pyspark_venv="pyspark_venv"
source_code="genai"
HDFS_HOST="50.140.197.220"
HDFS_PORT="9000"
property_file="/home/hduser/dba/bin/build/properties"
IMAGE="pytest-repo/spark-py:3.1.1"
IMAGEGCP="eu.gcr.io/axial-glow-224522/spark-py:java8_3.1.1"
CURRENT_DIRECTORY=`pwd`
CODE_DIRECTORY="/home/hduser/dba/bin/python/"
CODE_DIRECTORY_CLOUD=" gs://axial-glow-224522-spark-on-k8s/codes/"
cd $CODE_DIRECTORY
[ -f ${source_code}.zip ] && rm -r -f ${source_code}.zip
echo `date` ", ===> creating source zip directory from  ${source_code}"
# zip needs to be done at root directory of code
zip -rq ${source_code}.zip ${source_code}
hdfs dfs -rm -r hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${source_code}.zip
hdfs dfs -put ${source_code}.zip hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${source_code}.zip
hdfs dfs -rm -r hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${APPLICATION}
hdfs dfs -put /home/hduser/dba/bin/python/genai/src/${APPLICATION} hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${APPLICATION}
#gsutil cp ${source_code}.zip $CODE_DIRECTORY_CLOUD
#gsutil cp /home/hduser/dba/bin/python/genai/src/${APPLICATION} $CODE_DIRECTORY_CLOUD
cd $CURRENT_DIRECTORY

#export PYSPARK_DRIVER_PYTHON=/home/hduser/dba/bin/build/${pyspark_venv}/bin/python
#export PYSPARK_PYTHON=/home/hduser/dba/bin/build/${pyspark_venv}/bin/python

echo `date` ", ===> Submitting spark job"

# example ./generic.ksh -M k8s -A testme.py

#unset PYTHONPATH

if [[ "${MODE}" = "local" ]]
then
        spark-submit --verbose \
           --master local[4] \
           --conf "spark.yarn.appMasterEnv.SPARK_HOME=$SPARK_HOME" \
           --conf "spark.yarn.appMasterEnv.PYTHONPATH=${PYTHONPATH}" \
           --conf "spark.executorEnv.PYTHONPATH=${PYTHONPATH}" \
           --py-files hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/genai.zip \
           --archives hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${pyspark_venv}.tar.gz#${pyspark_venv} \
           hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${APPLICATION}

elif [[ "${MODE}" = "yarn" ]]
then
        spark-submit --verbose \
           --master yarn \
           --deploy-mode cluster \
           --conf "spark.yarn.appMasterEnv.SPARK_HOME=$SPARK_HOME" \
           --py-files hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/genai.zip \
           --conf "spark.driver.memory"=16G \
           --conf "spark.executor.memory"=4G \
           --conf "spark.num.executors"=3 \
           --conf "spark.executor.cores"=2 \
           hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${APPLICATION}

elif [[ "${MODE}" = "k8s" ]]
then
# Get MASTER-POD_IP
# docker run {{image_name}} .....
# docker run pytest-repo/spark-py:3.1.1 driver hdfs://50.140.197.220:9000/genai/codes/testme.py
#  docker run -u 0 -it c3bebe947f7b bash
#           --conf "spark.yarn.dist.archives"=hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${pyspark_venv}.tar.gz#${pyspark_venv} \
        unset PYSPARK_DRIVER_PYTHON
        export PYSPARK_PYTHON=/usr/bin/python3
export VOLUME_TYPE=hostPath
export VOLUME_NAME=genai-mount
export SOURCE_DIR=/d4T/hduser/genai
export MOUNT_PATH=$SOURCE_DIR/mnt
#--conf "spark.yarn.dist.archives"=hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${pyspark_venv}.tar.gz#${pyspark_venv} \
##--archives hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${pyspark_venv}.tar.gz#${pyspark_venv} \
           ##--conf spark.kubernetes.file.upload.path=$SOURCE_DIR \
           ##--conf spark.kubernetes.driver.volumes.$VOLUME_TYPE.$VOLUME_NAME.mount.path=$MOUNT_PATH \
           ##--conf spark.kubernetes.driver.volumes.$VOLUME_TYPE.$VOLUME_NAME.options.path=$MOUNT_PATH \
           ##--conf spark.kubernetes.executor.volumes.$VOLUME_TYPE.$VOLUME_NAME.mount.path=$MOUNT_PATH \
           ##--conf spark.kubernetes.executor.volumes.$VOLUME_TYPE.$VOLUME_NAME.options.path=$MOUNT_PATH \

        spark-submit --verbose \
           --master k8s://$K8S_SERVER \
           --deploy-mode cluster \
           --name pytest \
           --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./pyspark_venv/bin/python \
           --py-files hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/genai.zip \
           --conf spark.kubernetes.namespace=spark \
           --conf spark.network.timeout=300 \
           --conf spark.executor.instances=2 \
           --conf spark.kubernetes.driver.limit.cores=1 \
           --conf spark.executor.cores=1 \
           --conf spark.executor.memory=5000m \
           --conf spark.kubernetes.driver.docker.image=${IMAGEGCP} \
           --conf spark.kubernetes.executor.docker.image=${IMAGEGCP} \
           --conf spark.kubernetes.container.image=${IMAGEGCP} \
           --conf spark.driver.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true" \
           --conf spark.executor.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true" \
           --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark-serviceaccount \
           hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${APPLICATION}
        
        DRIVER_POD_NAME=`kubectl get pods -n spark |grep driver|awk '{print $1}'`
        kubectl logs $DRIVER_POD_NAME -n spark
        #kubectl describe pod $DRIVER_POD_NAME -n spark
        kubectl delete pod $DRIVER_POD_NAME -n spark
elif [[ "${MODE}" = "docker" ]]
then
        # docker run -it 64147345051 bash
        docker run pytest-repo/spark-py:3.1.1 driver \
           --py-files hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/genai.zip,hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/pyspark_venv.zip \
           hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${APPLICATION}

elif [[ "${MODE}" = "standalone" ]]
then
# Get MASTER-POD_IP
MASTER_POD_NAME=`kubectl get pods -o wide |grep master|awk '{print $1}'`
MASTER_POD_IP=`kubectl get pods -o wide |grep master|awk '{print $6}'`
kubectl exec $MASTER_POD_NAME -it  \
           -- spark-submit --verbose \
           --deploy-mode client \
           --conf spark.driver.bindAddress=$MASTER_POD_IP \
           --conf spark.driver.host=$MASTER_POD_IP \
           --py-files hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/genai.zip \
           --archives hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${pyspark_venv}.tar.gz#${pyspark_venv} \
           hdfs://$HDFS_HOST:$HDFS_PORT/genai/codes/${APPLICATION}

elif [[ "${MODE}" = "gcp" ]]
then
        echo "Run this code from cloud, quitting!"
        exit 1

        gsutil rm gs://$PROJECT-spark-on-k8s/codes/${source_code}.zip
        gsutil cp ${CODE_DIRECTORY}/${source_code}.zip gs://$PROJECT-spark-on-k8s/codes/
        gsutil rm gs://$PROJECT-spark-on-k8s/codes/${APPLICATION}
        gsutil cp  /home/hduser/dba/bin/python/genai/src/${APPLICATION} gs://$PROJECT-spark-on-k8s/codes/
        gcloud config set compute/zone europe-west2-c
        export PROJECT=$(gcloud info --format='value(config.project)')
        gcloud container clusters get-credentials spark-on-gke --zone europe-west2-c
        export KUBERNETES_MASTER_IP=$(gcloud container clusters list --filter name=spark-on-gke --format='value(MASTER_IP)')
        
        spark-submit --verbose \
           --properties-file ${property_file} \
           --master k8s://https://$KUBERNETES_MASTER_IP:443 \
           --deploy-mode cluster \
           --name pytest \
           --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./pyspark_venv/bin/python \
           --py-files gs://$PROJECT-spark-on-k8s/codes/genai.zip \
           --conf spark.kubernetes.namespace=spark \
           --conf spark.network.timeout=300 \
           --conf spark.executor.instances=2 \
           --conf spark.kubernetes.driver.limit.cores=1 \
           --conf spark.driver.cores=1 \
           --conf spark.executor.cores=1 \
           --conf spark.executor.memory=2000m \
           --conf spark.kubernetes.container.image=${IMAGEGCP} \
           gs://$PROJECT-spark-on-k8s/codes/${APPLICATION}
fi
