import numpy as np 


def getVarsFromData(dataList, netIdx, varNetIdxs):
	dataInput = []
	numSpatialDims = dataList[0].ndim - 2
	for dataMat in dataList:
		if (numSpatialDims == 1):
			dataInput.append(dataMat[:,varNetIdxs[netIdx],:])
		elif (numSpatialDims == 2):
			dataInput.append(dataMat[:,varNetIdxs[netIdx],:,:])
		else:
			dataInput.append(dataMat[:,varNetIdxs[netIdx],:,:,:])

	return dataInput
		

def parseValue(expr):
	"""
	Parse read text value into dict value
	"""

	try:
		return eval(expr)
	except:
		return eval(re.sub("\s+", ",", expr))
	else:
		return expr


def parseLine(line):
	"""
	Parse read text line into dict key and value
	"""

	eq = line.find('=')
	if eq == -1: raise Exception()
	key = line[:eq].strip()
	value = line[eq+1:-1].strip()
	return key, parseValue(value)


def readInputFile(inputFile):
	"""
	Read input file
	"""

	# TODO: better exception handling besides just a pass

	readDict = {}
	with open(inputFile) as f:
		contents = f.readlines()

	for line in contents: 
		try:
			key, val = parseLine(line)
			readDict[key] = val
			# convert lists to NumPy arrays
			# if (type(val) == list): 
			# 	readDict[key] = np.asarray(val)
		except:
			pass 

	return readDict

def catchInput(inDict, inKey, defaultVal):

	defaultType = type(defaultVal)
	try:
		# if NoneType passed as default, trust user
		if (defaultType == type(None)):
			outVal = inDict[inKey]
		else:
			outVal = defaultType(inDict[inKey])
	except:
		outVal = defaultVal

	return outVal