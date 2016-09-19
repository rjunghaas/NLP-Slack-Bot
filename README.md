** Update 2016 September 18 **

This is the ReadMe for my RFP_Bot for Slack.

The purpose of the Bot is to have an easy to use interface for sales engineers to quickly
query questions for Requests for Proposals (RFPs) from sales prospects.  Typically, this
can take several hours of research to get answers to questions.  Using Slack as an
interface for querying the database, we can speed up this process significantly for
the sales engineer completing the RFP.

I built this Bot for 3 main reasons:
a) I wanted to test Howdy.ai and build a Slack Bot.
b) I wanted to leverage Unix domain sockets to communicate between Node.js and Python 
services.
c) I wanted to gain some experience using Redis as a datastore.
d) I wanted to gain some experience with Natural Language Processing with the Python
library nltk.
e) I wanted to gain experience using TensorFlow with Seq2Seq models to train a generative
algorithm for responding to questions.


Set up:
I have included 30 hypothetical RFP questions and answers in CSV format inside the
rfp_bot directory.  To start, you will need to run the db_init.py script which will
create a Redis database and populate with the data from the CSV file.

Technical Details:
1. Slack Bot is built on top of Howdy.ai's Botkit.  This is a handy framework that wraps
Slack's Real-Time APIs for enabling chats with bots.

2. Within the Slack Bot, users can send direct messages of key topics or questions to the
Bot.  The Bot will then use a Unix domain socket to communicate with a Python Service.
This is handled by a Node.js and Python framework called ZeroRPC which wraps ZeroMQ.

3. The Python service will then query the Redis Database and pull out 3 random question
and answers which will then be sent back to the Slack Bot via the Unix socket.

4. The Slack Bot will parse the returned data from the query and display the results.

5. seq2seq_nlp_model.py borrows heavily from the TensorFlow Seq2Seq tutorial at: 
https://www.tensorflow.org/versions/r0.9/tutorials/seq2seq/index.html.  It contains 
code for accessing the questions and answers in Redis, constructing tokens, then training
a Seq2Seq model per the tutorial.  Finally, the model can be used to generate responses
to user-entered queries that Slack Bot passes to Python service.

Future Work:
Further integration of Seq2Seq code with Python middleware service to be completed