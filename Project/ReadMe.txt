
ReadMe File 

Table of Contents :


 ----> main 
        -->code
          1) CS567_preprocess.py
          2) CS657_regression.py
          3) CS657_lightgbm.py

 ---->login to perseus.vsnet.gmu.edu from the Windows Powershell 
 ---->scp main folder files and texts data into the perseus and run on the persues
 ----> put the data into the hdfs
 ----->hdfs dfs -put optiver-realized-volatility-prediction /user/sbyrapu/input/
 ----->hdfs dfs -ls -R /user/sbyrapu/input/

Run the files CS657_preprocessed.py, CS657_lightgbm.py in google collaboratory.

spark-submit --class org.apache.spark.examples.SparkPi --master yarn --deploy-mode client --num-executors 1 --driver-memory 8192m --executor-memory 8192m --executor-cores 1 CS657_regression.py


----->data 
       preprocessed data files train_pre_cs657 and test_pre_cs657  obtained after running 
      CS657_preprocessed.py