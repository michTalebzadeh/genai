DROP TABLE IF EXISTS ds.imputes_external;

CREATE EXTERNAL TABLE ds.imputes_external (
  col1 STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\n'
LOCATION 'hdfs://rhes75:9000/data/misc/';
DESC ds.imputes_external;
