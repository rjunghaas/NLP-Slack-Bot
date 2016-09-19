# This is a Seq2Seq model using TensorFlow which accesses our questions and answers
# in Redis and uses these to train a Seq2Seq model for generating responses to user
# queries.  The code borrows very heavily from the TensorFlow Seq2Seq tutorial.
 
import redis
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import csv
import tensorflow as tf
import re
import random
import numpy as np
import time
import math
import sys

# Uncomment these for initial downloads of language corpuses
#nltk.download('punkt')
#nltk.download('brown')

# Constants
HOST = 'localhost'
PORT = 6379
PASSWORD = None
TRAINING_PCT = 0.8

# Special Vocab Symbols
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

# Integer codes for Vocab Symbols
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Constants for Seq2Seq Initialization
MAX_VOCAB_SIZE = 10000
LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_FACTOR = 0.99
MAX_GRADIENT_NORM = 5.0
BATCH_SIZE = 20
SIZE = 512
NUM_LAYERS = 2
STEPS_PER_CHECKPOINT = 200
FINISH_TRAINING_STEP_LOSS = 7.00
_buckets = [(5,10), (10,15), (20,25), (40,50)]

# Seq2Seq Model Class
class Seq2SeqModel(object):
    # Function to set up Seq2Seq object model
    def __init__(self, 
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
    
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        output_projection = None
        softmax_loss_function = None
       
        # For output projects, need to set up w and b variables
        with tf.variable_scope("projection", reuse=None):
            if num_samples > 0 and num_samples < self.target_vocab_size:
                w = tf.get_variable("proj_w", [size, self.target_vocab_size], dtype=dtype)
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
                output_projection = (w, b)
        
                # Use output projection as inputs for softmax loss function
                def sampled_loss(inputs, labels):
                    labels = tf.reshape(labels, [-1, 1])
                    local_w_t = tf.cast(w_t, tf.float32)
                    local_b = tf.cast(b, tf.float32)
                    local_inputs = tf.cast(inputs, tf.float32)
                    return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels, num_samples, self.target_vocab_size), dtype)
            
                softmax_loss_function = sampled_loss
        
        # Create cells for Seq2Seq model
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        
        # Wrapper function for Seq2Seq with attention in Tensorflow library
        with tf.variable_scope("projection"):
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols=source_vocab_size,num_decoder_symbols=target_vocab_size,embedding_size=size,output_projection=output_projection,feed_previous=do_decode,dtype=dtype)
        
        # Initialize Encoder Inputs, Decoder Inputs, and Target Weights
        with tf.variable_scope("projection", reuse=True):
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(buckets[-1][0]): 
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size], name="encoder{0}".format(i)))
    
            for i in xrange(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(dtype, shape=[batch_size], name="weight{0}".format(i)))
    
            targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]
        
        # Forward Only is for decoding
        with tf.variable_scope("projection"):
            if forward_only:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, buckets, lambda x,y: seq2seq_f(x,y,True), softmax_loss_function=softmax_loss_function)
                if output_project is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projections[1] for output in self.outputs[b]]
            else:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs,self.decoder_inputs,targets,self.target_weights,buckets,lambda x,y: seq2seq_f(x,y,False),softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()
        # Use Gradient Descent to train
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
    
        self.save = tf.train.Saver(tf.all_variables())
    
    # Function for getting a batch of data, setting up inputs, and returning them
    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        
        # Take a random sample of data from correct bucket, calculate amount of padding, then create encoder and
        # decoder inputs with proper padding.  Note Encoder Inputs are reversed per algorithm
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
        
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
        
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input + [PAD_ID] * decoder_pad_size)
    
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        
        # Expand to create 2x2 arrays for Encoder Inputs, Decoder Inputs, and Target Weights
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size -1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
    
    # Function for a step of the Seq2Seq algorithm
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]
        # Check to ensure inputs and target weights are proper size for bucket we are using
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket," " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket," " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket," " %d != %d." % (len(target_weights), decoder_size))
    
        # Set up inputs and outputs before passing to run()
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        
        if not forward_only:
            output_feed = [self.updates[bucket_id],
                      self.gradient_norms[bucket_id],
                      self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])
        
        outputs = session.run(output_feed, input_feed)
        
        # For training - Gradient norm, loss, no output
        if not forward_only:
            return outputs[1], outputs[2], None
        # For decoding - No gradient norm, loss, outputs
        else:
            return None, outputs[0], outputs[1:]

# Function for accessing Redis database to get our Q&A data
# Then convert to list of [[question_tokens],[answer_tokens],...]
def export_tokens_from_database(host, port, password):
	r = redis.StrictRedis(host=host,port=port,password=password)
	text = []
	value = ""
	i = 0

	while (value != None):
		# index starts at 1
		value = r.get(i + 1) 
		if value == None:
			break
	
		# remove leading and trailing "{" "}" characters
		value = value[1:-1] 
		spl = value.split(":")
		text.append(spl)
		i += 1

	tokens = []
	answers = []
	for each in range(len(text)):
		q = text[each][0]
		a = text[each][1]
		q_tokens = word_tokenize(q)
		tokens.append(q_tokens[1:-1])
		answers.append(word_tokenize(a))

	fd = nltk.FreqDist(brown.words(categories='news'))
	cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
	most_freq_words = fd.keys()[:200]

	scrubbed_tokens = []
	for each in tokens:
		tok = []
		for word in each:
			if (word not in most_freq_words):
				tok.append(word)
		scrubbed_tokens.append(tok)
	
	final_tokens = zip(scrubbed_tokens, answers)
	return final_tokens

# Tokenizer function for taking sentence and splitting into list of strings
def tokenize_sentence(sentence):
    tokens = []
    for word in sentence.split():
        word = word.lower()
        word = re.sub("[^a-zA-Z]+", "", word)
        tokens.append(word)
    return tokens

# Create a dictionary with unique words as keys and counts as values
def vocab_builder(data, max_vocabulary_size):
    vocab = {}
    counter = 0
    for line in data:
        counter += 1
        tokens = tokenize_sentence(line)
        for word in tokens:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    
    vocab.pop('')
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if (len(vocab_list) > max_vocabulary_size):
        vocab_list = vocab_list[:max_vocabulary_size]
    return vocab_list

# Function to write vocab to csv file
def write_vocab_file(vocabulary_path, filename, vocab):
    full_filename = vocabulary_path + filename + ".csv"
    print ("Writing Vocab File: %s" % full_filename)
    with open(full_filename, "wb") as vocab_file:
        writer = csv.writer(vocab_file, "wb")
        for each in vocab:
            writer.write(each + b"\n")
    vocab_file.close()

# Reads in CSV file of tokens and separates into question and answer vocabs.
# Returns a list of vocabularies: [questions_vocab, answers_vocab]
def create_vocabulary(vocabulary_path, data_, max_vocabulary_size):
        
    # Read in raw data
    raw_data = []
    for row in data:
        raw_data.append(row)
        
    # split into questions and answers
    questions = []
    answers = []
    for each in raw_data:
        split_index = each.index(']')
        questions.append(each[0:split_index])
        answers.append(each[split_index + 1:])
 
    questions_vocab = vocab_builder(questions, max_vocabulary_size)
    answers_vocab = vocab_builder(answers, max_vocabulary_size)
    vocabs = [questions_vocab, answers_vocab]
    
    # write vocab to file if file path is specified
    if vocabulary_path:
        i = 0
        for vocab in vocabs:
            filename = VOCAB_FILES[i]
            write_vocab_file(vocabulary_path, filename, vocab) 
            i += 1
    
    return vocabs

# Reverse a vocabulary so that it can be sorted by counts
def unpack_vocab(vocab):
    return dict([(x,y) for (y,x) in enumerate(vocab)])

# Open vocab file and create dictionary for vocab
def initialize_vocabulary(vocabulary_path, vocab):
    if vocab:
        v = unpack_vocab(vocab)
    elif vocabulary_path:
        v_list = []
        with open(vocabulary_path, "r") as v_file:
            reader = csv.reader(v_file)
            for line in v_file:
                v_list.append(line)
        v = unpack_vocab(v_list)
        v_file.close()
    else:
        raise ValueError("Vocabulary File not found")
    
    return v

# Take a sentence of strings, and convert to list of token_ids
def sentence_to_token_ids(sentence, vocab):
    ids = []
    for word in sentence.split():
        word = word.lower()
        word = re.sub("[^a-zA-Z]+", "", word)
        token_id = vocab.get(word, UNK_ID)
        ids.append(token_id)
    return ids

# Opens csv file of text, splits into question / answer as well as train/test data sets
# Once data is split, data is then turned into token_ids
# All four sets of token_ids are returned as list:
# [train_questions_data, test_questions_data, train_answers_data, test_answers_data]
def data_to_token_ids(data, vocab_sets):
    text = []
    for line in data:
        text.append(line)
    
    # Separate into training and testing data
    questions = []
    answers = [] 
    for each in text:
        split_index = each.index(']')
        questions.append(each[0:split_index])
        answers.append(each[split_index + 1:])
    train_questions_data, test_questions_data = questions[:int(len(questions)*TRAINING_PCT)], questions[int(len(questions)*TRAINING_PCT):]
    train_answers_data, test_answers_data = answers[:int(len(answers)*TRAINING_PCT)], answers[int(len(answers)*TRAINING_PCT):]
    split_data = [train_questions_data, test_questions_data, train_answers_data, test_answers_data]
    
    # For each line, tokenize sentence
    tokenized_sets = []
    set_counter = 0
    for set in split_data:
        if set_counter < 2:
            vocab = vocab_sets[0]
        else:
            vocab = vocab_sets[1]
        tokenized_set = []
        for line in set:
            token_ids = sentence_to_token_ids(line, vocab)
            tokenized_set.append(token_ids)
        
        tokenized_sets.append(tokenized_set)
        set_counter += 1
    
    return tokenized_sets

# Main function for starting process of taking text data file
# and creating vocabs and token_ids
# Args: vocabulary_path = where to write vocab file
# data = data represented as tokens in form [[question_tokens],[answer_tokens],...]
# vocabulary_size = max size of vocabulary
def prep_data(vocabulary_path, data, vocabulary_size):
    data = []
    vocabs = create_vocabulary(vocabulary_path, data, vocabulary_size)
    vocab_sets = [initialize_vocabulary(None, v) for v in vocabs]
    token_ids = data_to_token_ids(data, vocab_sets)
    return [vocab_sets, token_ids]

# Wrapper function to call initializer for Seq2Seq model
def create_model(session):
    dtype = tf.float32
    model = Seq2SeqModel(
        MAX_VOCAB_SIZE,
        MAX_VOCAB_SIZE,
        _buckets,
        SIZE,
        NUM_LAYERS,
        MAX_GRADIENT_NORM,
        BATCH_SIZE,
        LEARNING_RATE,
        LEARNING_RATE_DECAY_FACTOR,
        dtype=dtype)
    session.run(tf.initialize_all_variables())
    return model

# Aggregate dev and test question/answer data together
# Place token_ids representing Q&A pairs into proper buckets
def read_data(source_data, target_data, max_size=None):
    data_set = [[] for _ in _buckets]
    i = 0
    while i < len(source_data) and (not max_size or i < max_size):
        source_ids = [int(x) for x in source_data[i]]
        target_ids = [int(x) for x in target_data[i]]
        target_ids.append(EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids])
                break
        i += 1

    return data_set

# Function to reverse a dictionary, sorted by keys
def reverse_dict(dict):
    return list(reversed(sorted(dict.keys())))

# Main function for training Seq2Seq model
def train(final_tokens):
    # Unpack and prep needed vocabs and token_ids from data file
    data = prep_data(None, final_tokens, MAX_VOCAB_SIZE)  
    vocabs, token_ids = data[0], data[1]

    # Initialize model
    model = create_model(sess)
    
    # Separate and prep data sets for training
    train_questions_data, test_questions_data, train_answers_data, test_answers_data = token_ids[0], token_ids[1], token_ids[2], token_ids[3]
    dev_set = read_data(test_questions_data, test_answers_data)
    train_set = read_data(train_questions_data, train_answers_data, MAX_VOCAB_SIZE)
    
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i+1])/train_total_size for i in xrange(len(train_bucket_sizes))]
    
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    
    # For continuous training of the model
    while True:
        # Select the relevant bucket, pass to get_batch which will output encoder/decoder inputs and weights
        # Then call step to get step loss
        # Run training loop until step loss is below a specified minimum
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / STEPS_PER_CHECKPOINT
        loss += step_loss / STEPS_PER_CHECKPOINT
        current_step += 1
        if step_loss < FINISH_TRAINING_STEP_LOSS:
            sys.stdout.flush()
            return model, vocabs
        
        sys.stdout.flush()

# Function for passing in a sentence and asking model to generate a response
def decode(model, sess, sentence, vocabs):
    # Get vocab for answers and reverse it
    answers_vocab = vocabs[1]
    rev_answers_vocab = reverse_dict(answers_vocab)
    
    # Convert sentence to token_ids and get relevant bucket_id
    token_ids = sentence_to_token_ids(sentence, vocabs[0])
    bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
    
    # Pass inputs to get_batch() to get encoder/decoder inputs & target weights
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
    # Pass inputs and weights to run a step in the model in decode mode (forward only)
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
    # Take most likely predicted value of output logits
    outputs = [np.int32(np.argmax(logit, axis=1)) for logit in output_logits]
    
    # Pass output logits to reversed answers vocab to get most likely words in our generated answer
    # Build a sentence response from output logits and return
    answer = ""
    for output in outputs:
        if EOS_ID in output:
            output = output[:output.index(EOS_ID)]
        if int(output[0]) < len(rev_answers_vocab):
            str = tf.compat.as_str(rev_answers_vocab[output[0]])
            answer += str
            answer += " "
        else:
            answer += "UNK "
    
    sys.stdout.flush()
    return answer

# Main Function
# Initialize a Tensorflow session, train model, generate a sample question, and compute response
final_tokens = export_tokens_from_database(HOST, PORT, PASSWORD)
with tf.Session() as sess:
    trained_model, vocabs = train(final_tokens)
    sample_question = INPUT_TEXT[random.randint(0, len(INPUT_TEXT) - 1)]
    answer = decode(trained_model, sess, sample_question, vocabs)
    print sample_question
    print answer
