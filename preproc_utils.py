import numpy as np 
import os
from tensorflow.keras.regularizers import l1, l2 
from sklearn.model_selection import train_test_split

# given list of data locations, aggregate data sets
# puts all data in NCHW format
def aggDataSets(dataDir, dataLocList, idxStartList, idxEndList, idxSkipList, dataOrder, pseudo1d=False):
	dataRaw = []
	for fileCount, dataFile in enumerate(dataLocList):
		dataLoc = os.path.join(dataDir, dataFile)
		dataLoad = np.load(dataLoc)  

		numDims = dataLoad.ndim - 2 # excludes N and C dimensions

		# NOTE: Keras models only accept NCHW or NHWC. 
		# 	Additionally, when running on CPUs, convolutional layers only accept NHWC
		# For now, everything goes to NCHW, will get transposed to NHWC right before training if requested
			
		if (dataOrder != "NCHW"):

			if (dataOrder == "NHWC"):

				if (numDims == 1):
					transAxes = (0,2,1)
				elif (numDims == 2):
					transAxes = (0,3,1,2)
				elif (numDims == 3):
					transAxes = (0,4,1,2,3)

			elif (dataOrder == "HWCN"):

				if (numDims == 1):
					transAxes = (2,1,0)
				elif (numDims == 2):
					transAxes = (3,2,0,1)
				elif (numDims == 3):
					transAxes = (4,3,0,1,2)

			elif (dataOrder == "HWNC"):

				if (numDims == 1):
					transAxes = (1,2,0)
				elif (numDims == 2):
					transAxes = (2,3,0,1)
				elif (numDims == 3):
					transAxes = (3,4,0,1,2)

			elif (dataOrder == "CHWN"):

				if (numDims == 1):
					transAxes = (2,0,1)
				elif (numDims == 2):
					transAxes = (3,0,1,2)
				elif (numDims == 3):
					transAxes = (4,0,1,2,3)

			elif (dataOrder == "CNHW"):

				if (numDims == 1):
					transAxes = (1,0,2)
				elif (numDims == 2):
					transAxes = (1,0,2,3)
				elif (numDims == 3):
					transAxes = (1,0,2,3,4)

			else:
				raise ValueError("Invalid dataOrder: "+str(dataOrder))

			dataLoad = np.transpose(dataLoad, axes=transAxes)

		# extract a range of iterations
		if (numDims == 1):
			dataLoad = dataLoad[idxStartList[fileCount]:idxEndList[fileCount]:idxSkipList[fileCount],:,:]
		elif (numDims == 2):
			dataLoad = dataLoad[idxStartList[fileCount]:idxEndList[fileCount]:idxSkipList[fileCount],:,:,:]
		elif (numDims == 3):
			dataLoad = dataLoad[idxStartList[fileCount]:idxEndList[fileCount]:idxSkipList[fileCount],:,:,:,:]

		# if pseudo-1D for GEMS, ignore y-velocity and z-velocity data   
		if pseudo1d:
			raise ValueError("PSEUDO-1D HAS NOT BEEN FIXED")

			assert (numDims > 1), "If making pseudo-1D network, data must be 2D or 3D"
			if (networkOrder == "NCHW"):
				channelIdx = 1
			else:
				channelIdx = -1
			noVIdxs = ((np.arange(dataLoad.shape[channelIdx]) != 2) == (np.arange(dataLoad.shape[channelIdx]) != 3))

			# TODO: THIS IS NOT VALID FOR ALL DIMENSIONS, DATA FORMATS
			# 	ideally, should also assert that this be in NCHW for TensorRT
			# 	Also, should just default to 2D data, easier on TensorRT side I think
			dataLoad = dataLoad[:,noVIdxs,[2],:] 

		# aggregate all data sets
		dataRaw.append(dataLoad.copy())

	return dataRaw


def preprocParamObjs(space, numDims, featShape, data_format):
	# weight regularization switch

	if (space['kernel_reg_type'] is not None):
		space['kernel_reg'] = regularizationSwitch(space['kernel_reg_type'],space['kernel_reg_val'])
	else:
		space['kernel_reg'] = None

	# activation regularization switch
	if (space['act_reg_type'] is not None):
		space['act_reg'] = regularizationSwitch(space['act_reg_type'],space['act_reg_val'])
	else:
		space['act_reg'] = None

	# bias regularization switch
	if (space['bias_reg_type'] is not None):
		space['bias_reg'] = regularizationSwitch(space['bias_reg_type'],space['bias_reg_val'])
	else:
		space['bias_reg'] = None

	# strides and kernel sizes need to be tuples
	space['stride_list_tuple']    = space['stride_list']
	space['kern_size_fixed_tuple'] = (int(space['kern_size_fixed']),) * numDims

	# check whether all-convolutional network can be built from given inputs
	if space["all_conv"]:

		if (data_format == "channels_last"):
			spatialDims = list(reversed(featShape[:-1]))
		else:
			spatialDims = list(reversed(featShape[1:]))

		numConvLayers = space['num_conv_layers']
		xStrides = [x[0] for x in space['stride_list_tuple']]
		xDimFinal = spatialDims[0] / np.prod(xStrides[:numConvLayers])
		dimFinal = xDimFinal
		if (numDims == 2):
			yStrides = [x[1] for x in space['stride_list_tuple']]
			yDimFinal = spatialDims[1] / np.prod(yStrides[:numConvLayers])
			dimFinal *= yDimFinal
		if (numDims == 3):
			zStrides = [x[2] for x in space['stride_list_tuple']]
			zDimFinal = spatialDims[2] / np.prod(zStrides[:numConvLayers])
			dimFinal *= zDimFinal

		filtFinal = (space['latent_dim'] / dimFinal)
		assert(filtFinal.is_integer()), "Cannot make final layer all-convolutional"
		space['filt_final'] = int(filtFinal)

	return space

# return regularizer objects
def regularizationSwitch(regType,regMult):
	
	if (regType == 'l2'):
		return l2(regMult)
	elif (regType == 'l1'):
		return l1(regMult)
	else:
		raise ValueError('Invalid regularization type:' + str(regType))

# TODO: actually implement switches properly 
# TODO: handle multiple datasets correctly
def preprocRawData(dataList_train, dataList_val, centering_scheme, normal_scheme, val_perc, modelDir, networkSuffix):

	# make train/val split from given training data
	if (dataList_val is None):

		# concatenate samples after centering
		for datasetNum, dataArr in enumerate(dataList_train):
			dataIn = centerDataSet(dataArr, centering_scheme, modelDir, networkSuffix, saveCent=True)
			dataInTrain, dataInVal = train_test_split(dataIn, test_size = val_perc, random_state = 24)
			if (datasetNum == 0):
				dataTrain = dataInTrain.copy()
				dataVal   = dataInVal.copy()
			else:
				dataTrain = np.append(dataTrain, dataInTrain, axis=0)
				dataVal   = np.append(dataVal, dataInVal, axis = 0)  

	else:
		# aggregate training samples after centering
		for datasetNum, dataArr in enumerate(dataList_train):
			dataTrainIn = centerDataSet(dataArr, centering_scheme, modelDir, networkSuffix, saveCent=True)
			if (datasetNum == 0):
				dataTrain = dataTrainIn.copy()
			else:
				dataTrain = np.append(dataTrain, dataTrainIn, axis = 0) 
		# shuffle training data to avoid temporal/dataset bias
		np.random.shuffle(dataTrain)        

		# aggregate validation samples after sampling
		# don't need to shuffle validation data
		for datasetNum, dataArr in enumerate(dataList_val):
			dataValIn = centerDataSet(dataArr, centering_scheme, modelDir, networkSuffix, saveCent=False)
			if (datasetNum == 0):
				dataVal = dataValIn.copy()
			else:
				dataVal = np.append(dataVal, dataValIn, axis = 0) 

	# normalize training and validation sets separately
	dataTrain, normSubTrain, normFacTrain = normalizeDataSet(dataTrain, normal_scheme, modelDir, networkSuffix, saveNorm=True)
	dataVal, _, _ = normalizeDataSet(dataVal, normal_scheme, modelDir, networkSuffix, norms=[normSubTrain, normFacTrain], saveNorm=False)

	return dataTrain, dataVal

# assumed to be in NCHW format
def centerDataSet(data, centType, modelDir, networkSuffix, saveCent=False):

	numDims = data.ndim - 2

	if (centType == "init_cond"):
		if (numDims == 1):
			centProf = data[[0],:,:]
		elif (numDims == 2):
			centProf = data[[0],:,:,:]
		elif (numDims == 3):
			centProf = data[[0],:,:,:,:]
		else: 
			raise ValueError('Something went wrong with centering (data dimensions)')

	elif (centType == "none"):
		centProf = np.zeros((1,)+data.shape[1:], dtype=np.float64)

	else:
		raise ValueError("Invalid choice of centType: " + centType)

	data = data - centProf

	if saveCent:

		centProf = np.squeeze(centProf, axis=0)
		np.save(os.path.join(modelDir, 'cent_prof_temp' + networkSuffix+'.npy'), centProf)

	return data

# normalize data set according to normType
def normSwitch(data, normType, axes):

	onesProf = np.ones((1,)+data.shape[1:], dtype = np.float64)
	zeroProf = np.zeros((1,)+data.shape[1:], dtype = np.float64)

	if (normType == 'minmax'):
		dataMin = np.amin(data, axis=axes, keepdims=True)
		dataMax = np.amax(data, axis=axes, keepdims=True)
		normSub = dataMin * onesProf 
		normFac = (dataMax - dataMin) * onesProf 

	elif (normType == "l2"):
		normFac = np.square(data)
		for dimIdx in range(len(axes)):
			normFac = np.sum(normFac, axis=axes[dimIdx], keepdims=True) 
		for dimIdx in range(len(axes)):
			normFac[:] /= data.shape[axes[dimIdx]]
		normFac = normFac * onesProf
		normSub = zeroProf

	else:
		raise ValueError("Invalid choice of normType: "+normType)

	return normSub, normFac

# determine how to normalized data given shape, normalize
# assumed to be in NCHW format
def normalizeDataSet(data, normType, modelDir, networkSuffix, norms=None, saveNorm=False):

	# calculate norms
	if (norms is None):
		numDims = data.ndim - 2   # ignore samples and channels dimensions
		if (numDims == 1):
			normAxes = (0, 2)

		elif (numDims == 2):
			normAxes = (0, 2, 3)

		elif (numDims == 3):
			normAxes = (0, 2, 3, 4)

		else: 
			raise ValueError('Something went wrong with normalizing (data dimensions)')

		normSub, normFac = normSwitch(data, normType, axes=normAxes)

	# norms are provided
	else:
		normSub = norms[0][None,:,:]
		normFac = norms[1][None,:,:]

	data = (data - normSub) / normFac
	
	if ((norms is None) and saveNorm):

		normSub = np.squeeze(normSub, axis=0)
		normFac = np.squeeze(normFac, axis=0)

		np.save(os.path.join(modelDir, 'norm_sub_prof_temp' + networkSuffix + '.npy'), normSub)
		np.save(os.path.join(modelDir, 'norm_fac_prof_temp' + networkSuffix + '.npy'), normFac)

	return data, normSub, normFac
