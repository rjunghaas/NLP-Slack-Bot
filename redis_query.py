# Sample Python service that queries a set of questions and answers in a Redis Database
import redis
import random
import json
import zerorpc

# Total questions in Redis Database
NUM_QUESTIONS = 30
# Total results to obtain
NUM_RESULTS = 3

# Create class to handle RPC queries from Slack client
class RedisQuery(object):
	
	# Single Function that queries the Redis database, randomly selects indexes and 
	# pulls out number of results that we have specified.
	# Results (Questions and Answers) are loaded into an object and returned.
	def queryRedis(self, query):
		returnObj = {}
		queryList = []
		
		# Connect to Redis Database
		r = redis.StrictRedis(host='localhost',port=6379,password=None)

		# Construct list of indexes that we randomly want to query
		i = 0
		while (i < NUM_RESULTS):
			key = random.randint(1, NUM_QUESTIONS)
			queryList.append(key)
			i += 1

		# Get the questions and answers at selected index and append to an object
		for each in queryList:
			value = r.get(each)
			json_acceptable_string = value.replace("'", "\"")
			d = json.loads(json_acceptable_string)
	
			for key in d:
				k = key

			returnObj[key] = d[key]
		
		print returnObj
		return returnObj

# Set up Listener for RPC
s = zerorpc.Server(RedisQuery())
s.bind("tcp://0.0.0.0:4242")
s.run()