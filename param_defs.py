from hyperopt import hp
from hyperopt.pyll import scope
from miscUtils import catchInput
from numpy import nan
import numpy as np

# some reference logarithms
# log(-9.21034037) ~= 0.0001
# log(-6.90775527) ~= 0.001,  
# log(-4.60517018) ~= 0.01
# log(-2.30258509) ~= 0.1,

# TODO: add optimizer options (just Adam right now)

# list of viable parameters, given as keyword-dtype-default triplets
# if default is nan, then it is a required parameter and has no default
params = [
	["all_conv", bool, False], 						# whether network is all-convolutional
	["latent_dim", int, nan],						# latent dimension
	["centering_scheme", str, nan],					# data centering method
	["normal_scheme", str, nan],					# data normalization scheme
	["activation_func", str, nan],					# layer activation
	["final_activation_func", str, nan],			# final layer activation

	["stride_list", [tuple], nan],					# stride length at each layer
	["num_conv_layers", int, nan],					# number of convolutional layers
	["kern_size_fixed", int, nan], 					# uniform kernel size

	# specify filter growth behavior, fixed kernel
	["num_filt_start", int, nan],					# number of filters at initial convolutional layer
	["filt_growth_mult", int, nan],					# growth rate of number of filters from layer to layer

	["kernel_reg_type", str, None],					# weight regularization type
	["kernel_reg_val", float, 0.0],					# weight regularization value
	["kernel_init_dist", str, "glorot_uniform"],	# initial distribution of kernel values
	["act_reg_type", str, None],					# activity regularization type
	["act_reg_val", float, 0.0],					# activity regularization value
	["bias_reg_type", str, None],					# bias regularization type
	["bias_reg_val", float, 0.0],					# bias regularization value
	["bias_init_dist", str, "zeros"],				# initial distribution of biases

	["learn_rate", float, 1e-4],					# optimization algorithm learning rate
	["max_epochs", int, nan],						# maximum number of training epochs
	["val_perc", float, nan],						# percentage of dataset to partition as validation set
	["loss_func", str, "mse"],						# string for loss function reference
	["es_patience", int, nan],						# number of iterations before early-stopping kicks in
	["batch_size", int, nan],						# batch size

	["layer_precision", int, 32],					# either 64 (for double-precision) or 32 (for single-precision)
]

def setExpression(parameterName, expressionType, inputList: list):
	"""
	Generate HyperOpt expression
	inputList has different interpretation depending on expressionType
	"""

	if (expressionType == "choice"):
		expression = hp.choice(parameterName, inputList)
	elif (expressionType == "uniform"):
		assert(len(inputList) == 2), "uniform expression only accepts 2 inputs ("+parameterName+")"
		expression = hp.uniform(parameterName, inputList[0], inputList[1])		
	elif (expressionType == "uniformint"):
		assert(len(inputList) == 2), "uniformint expression only accepts 2 inputs ("+parameterName+")"
		expression = hp.uniformint(parameterName, inputList[0], inputList[1])
	elif (expressionType == "quniform"):
		assert(len(inputList) == 3), "quniform expression only accepts 3 inputs ("+parameterName+")"
		expression = hp.quniform(parameterName, inputList[0], inputList[1], inputList[2])
	elif (expressionType == "quniformint"):
		assert(len(inputList) == 3), "quniformint expression only accepts 3 inputs ("+parameterName+")"
		expression = scope.int(hp.quniform(parameterName, inputList[0], inputList[1], inputList[2]))
	elif (expressionType == "loguniform"):
		assert(len(inputList) == 2), "loguniform expression only accepts 2 inputs ("+parameterName+")"
		expression = hp.loguniform(parameterName, inputList[0], inputList[1])
	elif (expressionType == "qloguniform"):
		assert(len(inputList) == 3), "qloguniform expression only accepts 3 inputs ("+parameterName+")"
		expression = hp.qloguniform(parameterName, inputList[0], inputList[1], inputList[2])
	else:
		raise ValueError("Invalid or un-implemented HyperOpt expressionType: " + str(expressionType))

	return expression


def defineParamSpace(inputDict, useHyperOpt):
	"""
	Define architecture and optimization parameters
	"""

	space = {}

	for paramPair in params:

		# no input and no default
		if ((paramPair[2] is nan) and (paramPair[0] not in inputDict)):
			raise ValueError(paramPair[0] + " is a required input")

		# has input value
		if (paramPair[0] in inputDict):
			inputVal = inputDict[paramPair[0]]

			# using hyperOpt
			if useHyperOpt:

				# determine expression type, default to "choice"
				expressionTypeName = paramPair[0] + "_expType"
				if (expressionTypeName in inputDict):
					expressionType = inputDict[expressionTypeName]
				else:
					expressionType = "choice"

				if (type(inputVal) is not list):
					assert(expressionType == "choice"), "If providing non-list expression parameters, must use \"choice\""
					inputVal = [inputVal]

				# TODO: make this more general to other list inputs
				if (paramPair[0] == "stride_list"):
					if (type(inputVal[0]) is not list):
						inputVal = [inputVal]

				space[paramPair[0]] = setExpression(paramPair[0], expressionType, inputVal)

			# not using hyperOpt
			else:
				
				if (type(paramPair[1]) is type):
					if ((inputVal is None) and (paramPair[2] is None)):
						pass
					else:
						assert (type(inputVal) is paramPair[1]), "Data type for "+paramPair[0]+" must be "+str(paramPair[1])
				elif (type(paramPair[1]) is list):
					assert (type(inputVal) is list)
					# check that all list elements in inputVal match expected type
					assert (all(isinstance(x, paramPair[1][0]) for x in inputVal))

				space[paramPair[0]] = inputVal


		# does not have input value (but has default)
		else:
			inputVal = paramPair[2] # default value

			if useHyperOpt:
				# HyperOpt requires list arguments
				inputVal = [inputVal]
				space[paramPair[0]] = setExpression(paramPair[0], "choice", inputVal)

			else:
				space[paramPair[0]] = inputVal

		# used on following iteration
		if (paramPair[0] == "stride_list"):
			strideList = inputVal

		# check that all num_conv_layers is LTE to all stride list lengths
		if (paramPair[0] == "num_conv_layers"):
			if useHyperOpt:
				if (expressionType == "choice"):
					numLayerVals = inputVal
				elif (expressionType == "quniform"):
					numLayerVals = np.arange(inputVal[0], inputVal[1], inputVal[2])
				elif (expressionType == "quniformint"):
					numLayerVals = np.arange(inputVal[0], inputVal[1], inputVal[2])
				else:
					raise ValueError("Can't check num_conv_layers with expression type "+expressionType)
				assert all(x <= len(y) for x in numLayerVals for y in strideList), "all values of num_conv_layers must be less than or equal to lengths of all stride_list's"
			else:
				assert (inputVal <= len(strideList)), "num_conv_layers must be less than or equal to the length of stride_list"

	return space

# 		'latent_dim': hp.choice('latent_dim',[50]),
# 		'num_conv_layers': hp.choice('num_conv_layers',[2,3,4]),
# 		'num_filt_start': hp.choice('num_filt_start',[8]),
# 		# 'filt_growth_mult': hp.quniform('filt_growth_exp',2,4,2),
# 		'filt_growth_mult': hp.choice('filt_growth_exp',[2]),
# 		'kern_size': hp.quniform('kern_size',5,25,5),
# 		# 'learn_rate': hp.loguniform('learn_rate',-9.21034037,-6.90775527),
# 		'learn_rate': hp.choice('learn_rate',[1.0e-4]),
# 		'kernel_reg': hp.choice('kernel_reg', [  {
# 													'type': 'l2',
# 													'reg_val': hp.loguniform('reg_val',-6.90775527,-2.30258509)
# 												 },
# 												# {
# 												#     'type': 'l1',
# 												#     'reg_val': hp.loguniform('reg_val',-6.90775527,-2.30258509)  
# 												# }
# 											  ]),
# 		'centering_scheme': hp.choice('centering_scheme',['init_cond']),
# 		'normal_scheme': hp.choice('norm_scheme',['minmax']),
# 		'activation_func': hp.choice('activation',['elu']),
# 		'final_activation_func': hp.choice('final_activation_func', ['linear']),
# 		'batch_size': hp.choice('batch_size',[50]),

