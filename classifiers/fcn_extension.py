# CNN_TSFRESH model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
import pandas as pd
from utils.model_utils import save_logs
from utils.model_utils import calculate_metrics

class Classifier_FCN_TSFRESH:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, input_agg=None, dense=None):
		self.output_directory = output_directory
		self.dense = dense
		if dense == 'class':
			self.dense = 2 * nb_classes

		if build == True:
			self.model = self.build_model(input_shape, nb_classes, input_agg=input_agg)
			if verbose > 0:
				self.model.summary()
			self.verbose = verbose
			# self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_raw, nb_classes, input_agg):

		input_layer_raw = keras.layers.Input(input_raw)
		input_layer_agg = keras.layers.Input(shape=input_agg)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer_raw)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

		z = keras.layers.Concatenate()([gap_layer, input_layer_agg])

		if self.dense is not None:
			z = keras.layers.Dense(self.dense, activation='relu')(z)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(z)

		model = keras.models.Model(inputs=[input_layer_raw, input_layer_agg], outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
			min_lr=0.0001)
		e_s = keras.callbacks.EarlyStopping(monitor='loss', patience=100)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint, e_s]

		return model


