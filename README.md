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
c) I wanted to gain some experience with Natural Language Processing with the Python
library nltk.
d) I wanted to gain some experience using Redis as a datastore.

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

Future Work:
As I build an understanding of NLP and nltk, I will be modifying the query service to
incorporate NLP to better match the user's query to questions in the database.