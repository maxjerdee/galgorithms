import numpy as np
import matplotlib.pyplot as plt
import time

def isotonic_regression(Y):
	res = np.ones_like(Y)*Y.sum()/(Y.shape[0]*Y.shape[1])
	s = np.zeros_like(Y)
	b = np.zeros(Y.shape, dtype=np.int)
	u = np.zeros(Y.shape, dtype=np.int)
	points = []
	for i in range(Y.shape[0]):
		row = []
		for j in range(Y.shape[1]):
			row.append([i,j])
		points.append(tuple(row))
	data = Y - res
	partition(data, s, b, u, res, points, Y.shape[0]*Y.shape[1], 20)
	"""
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
	axs[0][0].imshow(np.log(Y))
	start = time.time()
	end = time.time()
	axs[0][1].imshow(np.log(res))
	plt.show()
	print(end - start)
	"""
	return res

printing = False
def partition(data, s, b, u, res, points, size, depth):
	if printing:
		print("DDDDDDDDDDDDDDDDDDDDDDDDATA:")
		print(np.trunc(data))
		print("Over the points:")
		print(points)
	sum = 0
	for i in range(len(points)):
		for j in range(len(points[i])):
			sum += data[points[i][j][0]][points[i][j][1]]
	if printing:
		print("SUM")
		print(np.trunc(sum*100)/100)
	print("DEPTH: {0}, SIZE: {1}".format(depth, size))
	if size > 1 and depth > 0:
		rlen = []
		clen = len(points)
		for i in range(clen):
			rlen.append(len(points[i]))
		
		max = -10000
		max_index = [0,0]
		for i in range(len(points)):
			s[points[i][rlen[i]-1][0]][points[i][rlen[i]-1][1]] = data[points[i][rlen[i]-1][0]][points[i][rlen[i]-1][1]]
			
			for j in range(rlen[i] - 2, -1, -1):
				s[points[i][j][0]][points[i][j][1]] = s[points[i][j + 1][0]][points[i][j + 1][1]] + data[points[i][j][0]][points[i][j][1]]
			
			if i != 0:
				prev_end = points[i-1][rlen[i-1]-1]
				for j in range(rlen[i]):
					u[points[i][j][0]][points[i][j][1]] = b[prev_end[0]][np.minimum(prev_end[1], points[i][j][1])]
					s[points[i][j][0]][points[i][j][1]] += s[prev_end[0]][u[points[i][j][0]][points[i][j][1]]]
			
			row_max = -10000
			row_max_index = -1
			for j in range(rlen[i]):
				curr = s[points[i][j][0]][points[i][j][1]]
				if curr > row_max:
					row_max = curr
					row_max_index = points[i][j][1]
				b[points[i][j][0]][points[i][j][1]] = row_max_index
			if row_max > max:
				max = row_max
				max_index[0] = i
				max_index[1] = row_max_index
		
		pointers = []
		pointers.append(max_index[1])
		#c_i = points[clen - 1][rlen[clen - 1] - 1][0]
		for i in range(max_index[0], 0, -1):
			pointers.append(u[points[i][0][0]][pointers[max_index[0]-i]])
		"""
		for i in range(c_i, c_i - clen + 1, -1):
			curr = s[i][pointers[c_i - i]]
			if curr > max:
				max = curr
				m_index = i
			pointers.append(u[i][pointers[c_i - i]])
			"""
		pointers = np.flip(pointers, axis=0)
		
		upper_points = []
		lower_points = []
		up_num = 0
		low_num = 0
		if printing:
			print("MAX")
			print(np.trunc(s*100)/100)
			print(max_index)
			print("POINTERS")
			print(pointers)
		for i in range(clen):
			urow = []
			lrow = []
			for j in range(rlen[i]):
				#print("I {0} J {1} P[i] {2} P[i][j] {3}".format(i, j, pointers[i], points[i][j]))
				if i <= max_index[0] and points[i][j][1] >= pointers[i]:
					urow.append(points[i][j])
					up_num += 1
				else:
					lrow.append(points[i][j])
					low_num += 1
			if len(urow) > 0:
				upper_points.append(tuple(urow))
			if len(lrow) > 0:
				lower_points.append(tuple(lrow))
		if printing:
			print("Up: {0}, Low: {1}".format(up_num, low_num))
			#print(max)
			print("Results")
			print(np.trunc(s*100)/100)
			print(b)
			print(u)
			print(upper_points)
			print(lower_points)
			#print(data)
		for i in range(len(upper_points)):
			for j in range(len(upper_points[i])):
				res[upper_points[i][j][0]][upper_points[i][j][1]] += max/up_num
				data[upper_points[i][j][0]][upper_points[i][j][1]] -= max/up_num

		for i in range(len(lower_points)):
			for j in range(len(lower_points[i])):
				res[lower_points[i][j][0]][lower_points[i][j][1]] -= max/low_num
				data[lower_points[i][j][0]][lower_points[i][j][1]] += max/low_num
		#print(np.trunc(data*100)/100)

		if up_num != size:
			partition(data, s, b, u, res, upper_points, up_num, depth-1)
		if low_num != size:
			partition(data, s, b, u, res, lower_points, low_num, depth-1)
	else:
		return
np.random.seed(4)
#Y = np.random.randint(1,20,size=(10,10)).astype(float)
"""
Y = np.load('galaxy1.npy')[15:,:15]
newY = isotonic_regression(Y)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
axs[0][0].imshow(np.log(Y), vmin=4, vmax=8)
col = axs[0][1].imshow(np.log(newY), vmin=4, vmax=8)
fig.colorbar(col)
plt.show()
"""
Y = np.load('galaxy1.npy')
Y1 = Y[15:,:15]
Y2 = np.flip(Y[15:,15:], axis=1)
Y3 = np.flip(Y[:15,:15], axis=0)
Y4 = np.flip(np.flip(Y[:15,15:], axis=0), axis=1)
Y1 = isotonic_regression(Y1)
Y2 = isotonic_regression(Y2)
Y3 = isotonic_regression(Y3)
Y4 = isotonic_regression(Y4)
newY = np.concatenate((np.flip(np.concatenate((Y3, np.flip(Y4, axis=1)), axis=1), axis=0),np.concatenate((Y1, np.flip(Y2, axis=1)), axis=1)), axis=0)
#print(newY)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
axs[0][0].imshow(np.log(Y))
axs[0][1].imshow(np.log(newY))
axs[1][0].imshow(np.log(Y3))
axs[1][1].imshow(np.log(Y4))
plt.show()

