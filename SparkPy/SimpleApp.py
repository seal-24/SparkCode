"""SimpleApp.py"""
from pyspark import SparkContext

logFile = "/home/xunw/spark-1.0.0-bin-hadoop1/README.md"  # Should be some file on your system
sc = SparkContext("local", "MyApp")
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print "Lines with a: %i, lines with b: %i" % (numAs, numBs)
