// RFP Slack Bot Code using Howdy.ai

// Check to make sure a Slack token has been provided
// Starting bot should be on command line: token=xoxb-43305999795-fafMx5aV0Amo8hSM2v9jyftQ node rfp_bot.js
if (!process.env.token) {
    console.log('Error: Specify token in environment');
    process.exit(1);
}

// Libraries
var Botkit = require('./lib/Botkit.js');
var os = require('os');
var zerorpc = require('zerorpc');

// Initialize ZeroRPC client which will be used to communicate with Python Service
var client = new zerorpc.Client();
client.connect("tcp://127.0.0.1:4242");

// Initialize Howdy.ai Slackbot
var controller = Botkit.slackbot({
    debug: true,
});

var bot = controller.spawn({
    token: process.env.token
}).startRTM();

// Set up listener for any messages that are sent privately to our Slack Bot.
// Upon receiving a direct message, Bot will send query to Python Service via ZeroRPC.
// Results of asynchronous query are then parsed and "spoken" by Bot in Slack.
controller.hears('', 'direct_message', function(bot, message) {
	client.invoke("queryRedis", "RPC", function(err, res) {
		keys = Object.keys(res)
		for (var i = 0; i < keys.length; i++) {
			key = keys[i]
			bot.reply(message, "Response ".concat((i+1).toString()));
			bot.reply(message, key);
			bot.reply(message, res[key]);
		}
	});
});