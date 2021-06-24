import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from preproc_utils import preprocParamObjs, preprocRawData 
from cnn_utils import setConvLayer, getLoss
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow.keras.backend as K
from hyperopt import STATUS_OK 
import time
import pickle

##### GLOBAL VARIABLES #####
FITVERBOSITY 	 = 2				# 1 for progress bar, 2 for no progress bar

# "driver" objective function
# preprocesses data set, builds the network, trains the network, and returns training metrics
def objective_func(space, dataList_train, dataList_val, data_format, modelDir, networkSuffix, preprocSpace, pseudo1d=False):

	tStart = time.time()

	# pre-process data
	# includes centering, normalization, and train/validation split
	dataTrain, dataVal = preprocRawData(dataList_train, dataList_val, 
										space['centering_scheme'], space['normal_scheme'], 
										space['val_perc'], modelDir, networkSuffix)

	numDims = dataTrain.ndim - 2 # assumed spatially-oriented data, so subtract samples and channels dimensions

	# up until now, data has been in NCHW, tranpose if requesting NHWC
	if (data_format == "channels_last"):
		if (numDims == 1):
			transAxes = (0,2,1)
		elif (numDims == 2):
			transAxes = (0,2,3,1)
		elif (numDims == 3):
			transAxes = (0,2,3,4,1)
		dataTrain = np.transpose(dataTrain, transAxes)
		dataVal   = np.transpose(dataVal, transAxes)

	# build network 
	featShape = dataTrain.shape[1:]  									# shape of each sample (including channels)
	model     = build_CAE(space, featShape, numDims, data_format, 0) 	# must be implicit batch for training

	# train network 
	# returns trained model and loss metrics
	encoder, decoder, loss_train, loss_val = train_CAE(space, model, dataTrain, dataVal)

	# check if best validation loss
	# save to disk if best, update best validation loss so far
	check_CAE(encoder, decoder, loss_val, space, modelDir, networkSuffix)

	# return optimization info dictionary
	return {
		'loss': loss_train,					# training loss at end of training
		'true_loss': loss_val, 				# validation loss at end of training
		'status': STATUS_OK,      			# check for correct exit
		'eval_time': time.time() - tStart,	# time (in seconds) to train model
	}

# construct convolutional autoencoder
def build_CAE(space, featShape, numDims, data_format, explicitBatch):

	####### NETWORK DEFINITION #######
		
	# alter format of some inputs
	space = preprocParamObjs(space, numDims, featShape, data_format)

	K.set_floatx('float'+str(int(space["layer_precision"]))) # set network numerical precision

	featShapeList = list(featShape)
	# implicit batch for tf.keras training
	if (explicitBatch == 0):
		featShapeList.insert(0,None)
	# batch size one for single inference
	elif (explicitBatch == 1):
		featShapeList.insert(0,1) 
	# explicit batch size networks for Jacobian inference
	else: 
		featShapeList.insert(0,explicitBatch)			# just for CAE compatibility for decoder Jacobian

	featShape = tuple(featShapeList)
	if (data_format == "channels_first"):
		numChannels = featShape[1]
	else:
		numChannels = featShape[-1]

	# handle some issues with HyperOpt making floats instead of ints
	numConvLayers  = int(space['num_conv_layers'])
	numFiltStart   = int(space['num_filt_start']) 
	filtGrowthMult = int(space['filt_growth_mult']) 
	kernSizeFixed  = space['kern_size_fixed_tuple'] 			# this is already handled in preprocUtils

	inputEncoder = Input(batch_shape=featShape, name='inputEncode')

	if space['all_conv']: numConvLayers += 1 	# extra layer for all-convolutional network

	# construct encoder portion of autoencoder
	def build_encoder():
	
		x = inputEncoder

		# define sequential convolution layers
		nFilter = numFiltStart
		for convNum in range(0, numConvLayers):

			nKernel = kernSizeFixed

			if (space['all_conv'] and (convNum == (numConvLayers - 1))):
				nStride = 1
			else:
				nStride = space['stride_list_tuple'][convNum]

			x = setConvLayer(inputVals = x, convNum = convNum, dims = numDims,
							nFilter     = nFilter, 
							nKernel     = nKernel,  		
							nStride     = nStride,
							data_format = data_format,
							padding     = 'same', 
							kern_reg    = space['kernel_reg'],
							act_reg     = space['act_reg'],
							bias_reg    = space['bias_reg'], 
							activation  = space['activation_func'],
							kernelInit  = space['kernel_init_dist'], 
							biasInit    = space['bias_init_dist'],
							trans = False)
			
			if (space['all_conv'] and (convNum == (numConvLayers - 2))):
				nFilter = space['filt_final']
			else:
				nFilter = nFilter * filtGrowthMult

		# flatten before dense layer
		shape_before_flatten = x.shape.as_list()[1:]
		x = Flatten(name='flatten')(x)

		# without dense layer
		if (not space["all_conv"]):
			
			# set dense layer, if specified
			x = Dense(int(space['latent_dim']),
						activation           = space['activation_func'],
						kernel_regularizer   = space['kernel_reg'],
						activity_regularizer = space['act_reg'],
						bias_regularizer     = space['bias_reg'],
						kernel_initializer   = space['kernel_init_dist'], 
						bias_initializer     = space['bias_init_dist'],
						name = 'fcnConv')(x) 


		# NOTE: this reshape is for conformity with TensorRT
		# TODO: add a flag if making for TensorRT, otherwise this is pointless
		if (numDims == 2):
			x = Reshape((1, 1, int(space['latent_dim'])))(x) 
		if (numDims == 3):
			x = Reshape((1, 1, 1, int(space['latent_dim'])))(x)

		return Model(inputEncoder,x), shape_before_flatten

	# get info necessary to define first few layers of decoder
	encoder, shape_before_flatten = build_encoder() 
	dim_before_flatten = np.prod(shape_before_flatten)

	# construct decoder portion of autoencoder
	def build_decoder(): 
		
		# hard to a priori know the final convolutional layer output shape, so just copy from encoder output shape
		decoderInputShape = encoder.layers[-1].output_shape[1:] 
		decodeInputShapeList = list(decoderInputShape)
		
		# implicit batch size for tf.keras training
		if (explicitBatch == 0):
			decodeInputShapeList.insert(0,None) 
		else: 
			decodeInputShapeList.insert(0,explicitBatch) 			# for explicit-size decoder batch Jacobian
			
		decoderInputShape = tuple(decodeInputShapeList)
		inputDecoder = Input(batch_shape = decoderInputShape, name = 'inputDecode')

		x = inputDecoder

		if (not space['all_conv']):
			# dense layer
			x = Dense(dim_before_flatten, 
						activation           = space['activation_func'],
						kernel_regularizer   = space['kernel_reg'],
						activity_regularizer = space['act_reg'],
						bias_regularizer     = space['bias_reg'],
						kernel_initializer   = space['kernel_init_dist'], 
						bias_initializer     = space['bias_init_dist'],
						name='fcnDeconv')(x)
		
		# reverse flattening for input to convolutional layer
		x = Reshape(target_shape = shape_before_flatten, name = 'reshapeConv')(x)

		# define sequential transpose convolutional layers
		nFilter = numFiltStart * filtGrowthMult**(numConvLayers - 2)
		for deconvNum in range(0, numConvLayers):

			# make sure last transpose convolution has a linear activation and as many filters as original channels
			if (deconvNum == (numConvLayers - 1)):
				deconvAct  = space['final_activation_func']
				nFilter = numChannels
			else:
				deconvAct  = space['activation_func']

			nKernel = kernSizeFixed
			if space['all_conv']: 
				if (deconvNum == 0):
					nStride = 1
				else:
					nStride = space['stride_list_tuple'][deconvNum-1]
			else:
				nStride = space['stride_list_tuple'][deconvNum]

			x = setConvLayer(inputVals = x, convNum = deconvNum, dims = numDims,
							nFilter     = nFilter, 
							nKernel     = nKernel, 
							nStride     = nStride,
							data_format = data_format,
							padding     = 'same', 
							kern_reg    = space['kernel_reg'],
							act_reg     = space['act_reg'],
							bias_reg    = space['bias_reg'], 
							activation  = deconvAct,
							kernelInit  = space['kernel_init_dist'], 
							biasInit    = space['bias_init_dist'],
							trans=True)

			nFilter = int(nFilter / filtGrowthMult)

		return Model(inputDecoder,x)

	decoder = build_decoder() 
	return Model(inputEncoder,decoder(encoder(inputEncoder)))



# train the model
# accepts untrained model and optimization params
# returns trained model and loss metrics
def train_CAE(space, model, dataTrain, dataVal):

	encoder = model.layers[-2]
	encoder.summary()
	decoder = model.layers[-1]
	decoder.summary()

	# breakpoint()

	loss = getLoss(space['loss_func'])
	model.compile(optimizer = Adam(learning_rate = float(space['learn_rate'])), loss = loss) 

	# define callbacks
	callbackList = []
	earlyStop = EarlyStopping(patience = int(space['es_patience']), restore_best_weights = True)
	callbackList.append(earlyStop)

	# train model
	model.fit(x = dataTrain, y = dataTrain, 
					batch_size = int(space['batch_size']), 
					epochs = int(space['max_epochs']),
					validation_data = (dataVal, dataVal),
					verbose = FITVERBOSITY,
					callbacks = callbackList)

	
	loss_train = model.evaluate(x = dataTrain, y = dataTrain, verbose = 0)
	loss_val   = model.evaluate(x = dataVal, y = dataVal, verbose = 0)

	# repeat this just in case encoder/decoder memory reference has changed
	encoder = model.layers[-2]
	decoder = model.layers[-1]

	return encoder, decoder, loss_train, loss_val


# check if current trained model is best model produced so far
# if best, save to disk and update best validation loss
def check_CAE(encoder, decoder, loss_val, space, modelDir, networkSuffix):

	lossLoc = os.path.join(modelDir, 'valLoss' + networkSuffix + '.dat')

	# open minimum loss file
	try:
		with open(lossLoc) as f:
			minLoss = float(f.read().strip())
	except FileNotFoundError:
		print('First iteration, writing best model')
		minLoss = loss_val + 1.0 # first time, guarantee that loss is written to file

	# if current model validation loss beats previous best, overwrite
	if loss_val < minLoss:
		print("New best found! Overwriting...")

		encoder.save(os.path.join(modelDir, 'encoder'+networkSuffix+'.h5'))
		decoder.save(os.path.join(modelDir, 'decoder'+networkSuffix+'.h5')) 

		spaceLoc = os.path.join(modelDir, "paramSpace"+networkSuffix+".pickle")
		with open(spaceLoc, "wb") as f:
			pickle.dump(space, f)

		with open(lossLoc,'w') as f:
			f.write(str(loss_val))

		os.rename(os.path.join(modelDir, 'cent_prof_temp'+networkSuffix+'.npy'), os.path.join(modelDir, 'cent_prof'+networkSuffix+'.npy'))
		os.rename(os.path.join(modelDir, 'norm_sub_prof_temp'+networkSuffix+'.npy'), os.path.join(modelDir, 'norm_sub_prof'+networkSuffix+'.npy'))
		os.rename(os.path.join(modelDir, 'norm_fac_prof_temp'+networkSuffix+'.npy'), os.path.join(modelDir, 'norm_fac_prof'+networkSuffix+'.npy'))

	else:
		os.remove(os.path.join(modelDir, 'cent_prof_temp'+networkSuffix+'.npy'))
		os.remove(os.path.join(modelDir, 'norm_sub_prof_temp'+networkSuffix+'.npy'))
		os.remove(os.path.join(modelDir, 'norm_fac_prof_temp'+networkSuffix+'.npy'))
