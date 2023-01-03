

import os
from os.path import exists

import tensorflow as tf
import tensorflow_io as tfio

import matplotlib.pyplot as plt

from datetime import datetime


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
file_1 = "F:\\datasets\\turbofan\\CMaps\\RUL_FD001.txt"
file_2 = "F:\\datasets\\turbofan\\CMaps\\RUL_FD002.txt"
file_3 = "F:\\datasets\\turbofan\\CMaps\\RUL_FD003.txt"
file_4 = "F:\\datasets\\turbofan\\CMaps\\RUL_FD004.txt"
textfile_array = [ file_1, file_2, file_3, file_4 ]

input_encoding = "utf8"

list_words = [ ]
for i in range( 256 ):
	list_words.append( str(i) )
layer = tf.keras.layers.StringLookup(vocabulary=list_words)

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyLSTMLayer( tf.keras.layers.LSTM ):
	def __init__(self, units, return_sequences, return_state):
		super(MyLSTMLayer, self).__init__( units, return_sequences=True, return_state=False )
		self.num_units = units
		self.w = []
		self.b = []

	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
		shape=[int(input_shape[-1]),
		self.num_units])
		
		w_init = tf.constant_initializer(10.0)
		self.w = tf.Variable( initial_value=w_init(shape=(input_shape[-1], self.num_units), dtype='float32'), trainable=True)
		b_init = tf.keras.initializers.Identity( gain=5.0 )
		self.b = tf.Variable( initial_value=b_init(shape=(input_shape[-1], self.num_units), dtype='float32'), trainable=True)

	def call(self, inputs):

		return tf.matmul(inputs, self.w) + self.b

def standardize(input_data):
	input_data = tf.strings.lower(input_data)
	input_data = tf.strings.regex_replace(input_data,  "<[^>]+>", " ")
	
	return input_data

def read_input_textfile( filename ) :
	dataset = tf.data.TextLineDataset([filename])
	
	output = []
	for d in dataset:
		output.append( layer( tf.constant([[ str(int(tf.strings.to_number(standardize(d.numpy())).numpy())) ]]) ).numpy()[0][0] )
		
	for i in range( 300 - len( output ) ) :
		output.append( 0 )
		
	return output
		
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
series_1 = read_input_textfile( textfile_array[0] )
series_2 = read_input_textfile( textfile_array[1] )
series_3 = read_input_textfile( textfile_array[2] )
series_4 = read_input_textfile( textfile_array[3] )	
series_values = [ series_1, series_2, series_3, series_4 ]

start = 0
limit = 300
delta = 1
x_scales = tf.range(start=start, limit=limit, delta=delta, dtype=tf.int32, name='range')

plt.plot( x_scales, series_1, linewidth=2.0, color='red' )
plt.plot( x_scales, series_2, linewidth=2.0, color='green' )
plt.plot( x_scales, series_3, linewidth=2.0, color='blue' )
plt.plot( x_scales, series_4, linewidth=2.0, color='yellow' )
plt.title("Series 1 to 4")
# plt.xlabel("Series 1 to 4")
# plt.ylabel("Distance as height")
plt.legend(['series_1', 'series_2', 'series_3', 'series_4'])

plt.ylim([1, 200])
plt.show()



DATA = tf.constant(tf.cast([ series_1, series_2, series_3, series_4 ], dtype=tf.float32), shape=(4, 1, 1, 300))
LABEL = tf.constant([ 1, 2, 3, 4 ], shape=(4, 1, 1, 1), dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (1, 300)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=input_shape),

	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(192, activation='relu'),
	tf.keras.layers.Dense(4),
])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		# if logs['loss'] <= 0.2 and self.wait > self.patience :
		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    name='Nadam'
)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.LogCosh(
    reduction=tf.keras.losses.Reduction.AUTO, name='log_cosh'
)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit( dataset, batch_size=1000, epochs=5000, callbacks=[custom_callback] )
model.save_weights(checkpoint_path)

predictions = model.predict( tf.expand_dims(tf.expand_dims(series_4, 0), 0) )
score = tf.nn.softmax(predictions[0])

start = 0
limit = 300
delta = 1
x_scales = tf.range(start=start, limit=limit, delta=delta, dtype=tf.float32, name='range')

plt.plot( x_scales, series_1, linewidth=2.0, color='red' )
plt.plot( x_scales, series_2, linewidth=2.0, color='green' )
plt.plot( x_scales, series_3, linewidth=2.0, color='blue' )
plt.plot( x_scales, series_4, linewidth=2.0, color='yellow' )
plt.plot( x_scales + 1, series_values[tf.math.argmax(score)], linewidth=2.0, color='orange' )
plt.title("Series " + str(tf.math.argmax(score).numpy()) + " scores: " + str(score[tf.math.argmax(score)].numpy()))
plt.xlabel("Series 1 to 4")
# plt.ylabel("Series " + str(tf.math.argmax(score)) + " scores: " + str(score[tf.math.argmax(score)]))
plt.legend(['series_1', 'series_2', 'series_3', 'series_4', 'prediction'])

plt.ylim([1, 200])
plt.show()
