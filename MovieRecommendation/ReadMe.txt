ReadMe File 

Table of Contents :
 ----> main 
        -->code
          1) ALS
          2) SVL
          3) Item_Item_CF
          4) hybrid
          

  -->
        
 ---->login to perseus.vsnet.gmu.edu from the Windows Powershell 
 ---->scp main folder files and texts data into the perseus and run on the persues
 ----> put the data into the hdfs
  ---->hdfs dfs -put ml-20m /user/sbyrapu/input/
  ----> hdfs dfs -ls -R /user/sbyrapu/input/

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 Item_Item_CF.py

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 ALS.py

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 hybrid.py

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 SVL.py
