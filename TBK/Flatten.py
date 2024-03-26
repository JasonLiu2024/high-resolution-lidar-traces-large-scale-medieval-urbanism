"""flatten"""
import numpy as np
def write_TLC_input(filename, restVert, initVert, triangles, handles):
	"""write data (numpy array) as TLC program input format"""
	with open(filename, 'w') as file:
		# rest vertices
		nV, nDim = restVert.shape
		file.write(f'{nV} {nDim}\n')
		np.savetxt(file, restVert)
		# init vertices
		nV, nDim = initVert.shape
		file.write(f'{nV} {nDim}\n')
		np.savetxt(file, initVert)
		# simplex cells <- triangles
		nF, simplex_size = triangles.shape
		file.write(f'{nF} {simplex_size}\n')
		np.savetxt(file, triangles, "%d")
		# handles
		nhdls = handles.size
		file.write(f'{nhdls}\n')
		np.savetxt(file, handles, "%d")
def read_TLC_result(filename):
	"""read TLC program results into a dictionary"""
	result_dict = {}
	with open(filename) as file:
		while True:
			line = file.readline()
			if line.strip() == "":
				break
			head = line.split()
			data_name = head[0]
			# print(f"data_name: {data_name}")
			array_shape = [int(i) for i in head[1:]]
			# print(f"head: {head}")
			# print(file)
			# if head[0] == "energy":
			# 	break
			data = np.loadtxt(file, dtype=np.float64, max_rows=1)
			if len(array_shape) == 1:
				result_dict[data_name] = data
			else:
				result_dict[data_name] = np.reshape(data, array_shape)
	return result_dict