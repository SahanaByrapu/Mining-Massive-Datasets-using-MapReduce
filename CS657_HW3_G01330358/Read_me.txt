ReadMe File 

Table of Contents :
 ----> main 
        -->code
          1) HW3_CS657_final.py
        -->output_CS657.text
        
 ---->login to perseus.vsnet.gmu.edu from the Windows Powershell 
 ---->scp main folder files and texts data into the perseus and run on the persues
 ----> put the data into the hdfs
  ---->hdfs dfs -put hashtag_joebiden.csv /user/sbyrapu/input/
 ----> run HW2_CS657.py to get the required output
 ----> spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 512m --executor-memory 512m --executor-cores 1 HW3_CS657_final.py