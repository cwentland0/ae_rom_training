from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K


def transferWeights(model1,model2):
	"""
	Transfer weights from one network to another 
	Useful for building explicit-batch network from trained implicit-batch network
	"""

	model2.set_weights(model1.get_weights())
	return model2 


def getLoss(lossName):
	"""
	Switch function for network loss function
	"""

	if (lossName == "pure_l2"):
		return pureL2
	elif (lossName == "pure_mse"):
		return pureMSE
	else:
		return lossName # assumed to be a built-in loss string


def pureL2(yTrue, yPred): 
	"""
	Strict l2 error (opposed to mean-squared)
	"""

	return K.sum(K.square(yTrue - yPred))


def pureMSE(yTrue, yPred):
	"""
	Strict mean-squared error, not including regularization contribution
	"""

	mse = MeanSquaredError()
	return mse(yTrue, yPred)


# defines 1D, 2D, and 3D convolutions, as well as their corresponding transpose convolutions
# N.B.: ASSUMED "channels_first" FORMAT!
def setConvLayer(inputVals,
				convNum,
				dims,
				nFilter,
				nKernel,
				nStride,
				data_format,
				padding ='same',
				kern_reg = None,
				act_reg = None,
				bias_reg = None, 
				activation = 'None',
				kernelInit = 'glorot_uniform',
				biasInit = 'zeros',
				trans = False,
				):

	if trans:
		name = 'tconv'+str(convNum)
		if (dims == 1):
			x = Conv1DTranspose(filters = nFilter, kernel_size = nKernel, strides = nStride, padding = padding, activation = activation,
					   kernel_initializer = kernelInit, bias_initializer = biasInit, 
					   kernel_regularizer = kern_reg, activity_regularizer = act_reg, bias_regularizer = bias_reg,
					   data_format = data_format, name = name)(inputVals)
		elif (dims == 2):
			x = Conv2DTranspose(filters = nFilter, kernel_size = nKernel, strides = nStride, padding = padding, activation = activation,
					   kernel_initializer = kernelInit, bias_initializer = biasInit,
					   kernel_regularizer = kern_reg, activity_regularizer = act_reg, bias_regularizer = bias_reg,
					   data_format = data_format, name = name)(inputVals)
		elif (dims == 3):
			x = Conv3DTranspose(filters = nFilter, kernel_size = nKernel, strides = nStride, padding = padding, activation = activation,
					   kernel_initializer = kernelInit, bias_initializer = biasInit, 
					   kernel_regularizer = kern_reg, activity_regularizer = act_reg, bias_regularizer = bias_reg,
					   data_format = data_format, name = name)(inputVals) 
		else:
			raise ValueError("Invalid dimensions for transpose convolutional layer "+str(convNum))
	else:
		name = 'conv'+str(convNum)
		if (dims == 1):
			x = Conv1D(filters = nFilter, kernel_size = nKernel, strides = nStride, padding = padding, activation = activation,
					   kernel_initializer = kernelInit, bias_initializer = biasInit,
					   kernel_regularizer = kern_reg, activity_regularizer = act_reg, bias_regularizer = bias_reg,
					   data_format = data_format, name = name)(inputVals)
		elif (dims == 2):
			x = Conv2D(filters = nFilter, kernel_size = nKernel, strides = nStride, padding = padding, activation = activation,
					   kernel_initializer = kernelInit, bias_initializer = biasInit,
					   kernel_regularizer = kern_reg, activity_regularizer = act_reg, bias_regularizer = bias_reg, 
					   data_format = data_format, name = name)(inputVals)
		elif (dims == 3):
			x = Conv3D(filters = nFilter, kernel_size = nKernel, strides = nStride, padding = padding, activation = activation,
					   kernel_initializer = kernelInit, bias_initializer = biasInit,
					   kernel_regularizer = kern_reg, activity_regularizer = act_reg, bias_regularizer = bias_reg,
					   data_format = data_format, name = name)(inputVals) 
		else:
			raise ValueError("Invalid dimensions for convolutional layer "+str(convNum)) 

	return x