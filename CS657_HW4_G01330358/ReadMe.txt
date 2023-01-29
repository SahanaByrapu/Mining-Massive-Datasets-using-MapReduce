ReadMe File 

Table of Contents :
 ----> main 
        -->code
          1) CS657_HW4_ALS
          2) CS567_hw4_SVL
          3) CS657_HW4_Item_Item_CF
          4) CS657_hybrid
          

  -->
        
 ---->login to perseus.vsnet.gmu.edu from the Windows Powershell 
 ---->scp main folder files and texts data into the perseus and run on the persues
 ----> put the data into the hdfs
  ---->hdfs dfs -put ml-20m /user/sbyrapu/input/
  ----> hdfs dfs -ls -R /user/sbyrapu/input/

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 CS657_HW4_Item_Item_CF.py

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 CS657_HW4_ALS.py

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 CS657_hybrid.py

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 CS657_Hw4_SVL.py