# Python script for reading 'RFP Questions.csv' and saving into a Redis database

import redis
import csv
import fileinput

FILENAME = 'RFP_Questions.csv'

# Initialize connection to Redis
r = redis.StrictRedis(host='localhost',port=6379,password=None)
r.flushdb()

with open(FILENAME, 'rb') as datafile:
	redisReader = csv.reader(datafile, delimiter=',')
	for row in redisReader:
		index = row[0]
		question = row[1]
		answer = row[2]
		value = {}
		value[question] = answer
		print value
		r.set(index, value)

r.save()