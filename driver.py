import numpy as np
import os
from paramDefs import defineParamSpace
from trainAutoencoder import objective_func, build_CAE
from preprocUtils import aggDataSets
from cnnUtils import transferWeights
from miscUtils import getVarsFromData, readInputFile, catchInput
from hyperopt import fmin, tpe, rand, Trials, space_eval
from functools import partial 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed
import argparse
import pickle

np.random.seed(24) # seed NumPy RNG
set_seed(24)

##################### START USER INPUTS #####################

# read working directory input
parser = argparse.ArgumentParser(description = "Read input file")
parser.add_argument('inputFile', type = str, help="input file")
inputFile = os.path.expanduser(parser.parse_args().inputFile)
assert (os.path.isfile(inputFile)), "Given inputFile does not exist"
inputDict = readInputFile(inputFile)

runCPU 			= catchInput(inputDict, "runCPU", False)

dataDir         = inputDict["dataDir"]
modelDir        = inputDict["modelDir"]
modelLabel      = inputDict["modelLabel"]
dataFiles_train = inputDict["dataFiles_train"]
dataFiles_val   = catchInput(inputDict, "dataFiles_val", [None])
varNetworkIdxs  = list(inputDict["varNetworkIdxs"])

numNetworks     = len(varNetworkIdxs)
idxStartList    = catchInput(inputDict, "idxStartList", [0]*numNetworks)
idxEndList      = catchInput(inputDict, "idxEndList", [None]*numNetworks)
idxSkipList     = catchInput(inputDict, "idxSkipList", [None]*numNetworks)
dataOrder       = inputDict["dataOrder"]
networkOrder    = inputDict["networkOrder"]

# HyperOpt parameters
useHyperOpt      = catchInput(inputDict, "useHyperOpt", False) 
hyperOptAlgo     = catchInput(inputDict, "hyperOptAlgo", "tpe") 
hyperOptMaxEvals = catchInput(inputDict, "hyperOptMaxEvals", 100)

# for TensorRT compliance
outputTRT = catchInput(inputDict, "outputTRT", False)

##################### END USER INPUTS ##################### 

# run on CPU vs GPU
if runCPU:
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
else:
	# make sure TF doesn't gobble up device memory
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)

# TODO: batch norm back into network
# TODO: add option for all-convolution network

modelDir = os.path.join(modelDir, modelLabel)
if not os.path.exists(modelDir): os.makedirs(modelDir) 

# setting preprocessing
space = defineParamSpace(inputDict, useHyperOpt)

networkSuffixes = ['']*numNetworks
for i, netIdxs in enumerate(varNetworkIdxs):
	for j, idx in enumerate(netIdxs):
		networkSuffixes[i] += "_" + str(idx)

# delete loss value file, so run overwrites anything already in folder
for networkSuffix in networkSuffixes:
	lossLoc = os.path.join(modelDir, "valLoss" + networkSuffix + ".dat")
	try:
		os.remove(lossLoc)
	except FileNotFoundError:
		pass

# hyperOpt search algorithm
# 'rand' for random search, 'tpe' for tree-structured Parzen estimator
if useHyperOpt:
	if (hyperOptAlgo == "rand"):
		hyperOptAlgo = rand.suggest
	elif (hyperOptAlgo == "tpe"):
		hyperOptAlgo = tpe.suggest
	else:
		raise ValueError("Invalid input for hyperOptAlgo: " + str(hyperOptAlgo))

####### LOAD RAW DATA #######

numDatasetsTrain = len(dataFiles_train)
if (len(idxStartList) == 1): idxStartList = idxStartList * numDatasetsTrain
if (len(idxEndList) == 1): idxEndList = idxEndList * numDatasetsTrain
if (len(idxSkipList) == 1): idxSkipList = idxSkipList * numDatasetsTrain

dataRaw_train = aggDataSets(dataDir, dataFiles_train, idxStartList, idxEndList, idxSkipList, dataOrder)
if (dataFiles_val[0] is not None):
	dataRaw_val = aggDataSets(dataDir, dataFiles_val, idxStartList, idxEndList, idxSkipList, dataOrder)
else:
	dataRaw_val = None

if (networkOrder == "NCHW"):
	data_format = "channels_first"
elif (networkOrder == "NHWC"):
	data_format = "channels_last"
else:
	raise ValueError("Invalid networkOrder: "+str(networkOrder))

####### MODEL OPTIMIZATION #######
# optimize as many models as requested, according to variable split
for netIdx in range(numNetworks):

	netSuff = networkSuffixes[netIdx]
	if (varNetworkIdxs is None):
		dataInput_train = dataRaw_train         
		dataInput_val   = dataRaw_val
	else:
		dataInput_train = getVarsFromData(dataRaw_train, netIdx, varNetworkIdxs)
		if (dataFiles_val[0] is not None):
			dataInput_val = getVarsFromData(dataRaw_val, netIdx, varNetworkIdxs)
		else:
			dataInput_val = dataRaw_val
		
	if useHyperOpt:
		print("Performing hyper-parameter optimization!")
		trials = Trials()
		# wrap objective function to pass additional arguments
		objective_func_wrapped = partial(objective_func, dataList_train = dataInput_train, dataList_val = dataInput_val, 
														data_format = data_format, modelDir = modelDir, 
														networkSuffix = netSuff, preprocSpace = True)

		# find "best" model according to specified hyperparameter optimization algorithm
		best = fmin(fn = objective_func_wrapped,
					space = space,
					algo = hyperOptAlgo,
					max_evals = hyperOptMaxEvals,
					show_progressbar = False,
					rstate = np.random.RandomState(24),
					trials = trials)

		# TODO: train the model again on the full dataset with the best hyper-parameters

		# save HyperOpt metadata to disk
		bestSpace = space_eval(space, best)
		print("Best parameters:")
		print(bestSpace)
		f = open(os.path.join(modelDir, "hyperOptTrials" + netSuff + ".pickle"), "wb")
		pickle.dump(trials, f)
		f.close()

	else: 
		print("Optimizing single architecture!")
		if (netIdx == 0):
			preprocSpace = True
		else:
			preprocSpace = False
		best = objective_func(space, dataInput_train, dataInput_val, data_format, modelDir, netSuff, preprocSpace)
		bestSpace = space

	# write parameter space to file
	f = open(os.path.join(modelDir, "bestSpace" + netSuff + ".pickle"), "wb")
	pickle.dump(bestSpace, f)
	f.close()

	# generate explicit batch networks for TensorRT
	if outputTRT:

		# load best model
		encoder = load_model(os.path.join(modelDir, 'encoder' + netSuff + '.h5'), compile=False)
		decoder = load_model(os.path.join(modelDir, 'decoder' + netSuff + '.h5'), compile=False)
		inputShape  = encoder.layers[0].input_shape[0][1:]
		spatialDims = dataInput_train[0].ndim - 2

		# save batch size one network 
		model_batchSizeOne = build_CAE(bestSpace, inputShape, spatialDims, data_format, 1)
		decoder_batchSizeOne = model_batchSizeOne.layers[-1]
		encoder_batchSizeOne = model_batchSizeOne.layers[-2]
		decoder_batchSizeOne = transferWeights(decoder, decoder_batchSizeOne)
		encoder_batchSizeOne = transferWeights(encoder, encoder_batchSizeOne)
		decoder_batchSizeOne.save(os.path.join(modelDir, 'decoder_batchOne' + netSuff + '.h5'))
		encoder_batchSizeOne.save(os.path.join(modelDir, 'encoder_batchOne' + netSuff + '.h5'))

		# save decoder with batch size equal to latent dimension
		model_batchJacob_decode = build_CAE(bestSpace, inputShape, spatialDims, data_format, bestSpace['latent_dim'])
		decoder_batchJacob_decode = model_batchJacob_decode.layers[-1]
		decoder_batchJacob_decode = transferWeights(decoder, decoder_batchJacob_decode)
		decoder_batchJacob_decode.save(os.path.join(modelDir, 'decoder_batchJacob' + netSuff + '.h5'))
