import numpy as np
import matplotlib.pyplot as plt

def peak(original, x, y):
	"""finds the matrix closest to 'original' subject to the "peak" being at 
	(x,y) i.e. any ray eminating from the lower left-hand corner of (x,y) 
	will intersect a series of pixels that are non-increasing in value"""
	Y1 = original[x:,:y]
	Y2 = np.flip(original[x:,y:], axis=1)
	Y3 = np.flip(original[:x,:y], axis=0)
	Y4 = np.flip(np.flip(original[:x,y:], axis=0), axis=1)
	Y1 = isotonic_regression(Y1)
	Y2 = isotonic_regression(Y2)
	Y3 = isotonic_regression(Y3)
	Y4 = isotonic_regression(Y4)
	return np.concatenate((np.flip(np.concatenate((Y3, np.flip(Y4, axis=1)), axis=1), axis=0),np.concatenate((Y1, np.flip(Y2, axis=1)), axis=1)), axis=0)

def isotonic_regression(Y):
	"""Sets up the partition function to find the isotonic regression,
	the values closest (in the l2 sense) to Y under the condition that 
	the result A satifies A_i,j >= A_(i-1),j and A_i,(j-1)"""
	res = np.ones_like(Y)*Y.sum()/(Y.shape[0]*Y.shape[1])
	s = np.zeros(Y.shape, dtype=np.float)
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
	return res

def partition(data, s, b, u, res, points, size, depth):
	"""Finds the maximal set within "data" over the given "points" (which must have mean 0)
	and then partitions again the upper and lower halves, while updating the solution "res"""
	# depth is just for demonstration purposes, terminating the recursion early
	
	# termination conditions
	if size > 1 and depth > 0:

		# variables that keep track of the scope of "points" for iteration purposes
		rlen = []
		clen = len(points)
		for i in range(clen):
			rlen.append(len(points[i]))
		
		# keeps track of which point defines the maximal set
		max = -10000
		max_index = [0,0]

		# each point on the grid defines a potentially maximal set (including that point and the best 
		# choice for higher rows) s[x][y] tracks the value of the set defined by (x, y)
		for i in range(len(points)):
			# calculating s based on current row
			s[points[i][rlen[i]-1][0]][points[i][rlen[i]-1][1]] = data[points[i][rlen[i]-1][0]][points[i][rlen[i]-1][1]]
			for j in range(rlen[i] - 2, -1, -1):
				s[points[i][j][0]][points[i][j][1]] = s[points[i][j + 1][0]][points[i][j + 1][1]] + data[points[i][j][0]][points[i][j][1]]
			
			# if below the first row, factoring in the optimal set from above rows
			if i != 0:
				prev_end = points[i-1][rlen[i-1]-1]
				for j in range(rlen[i]):
					u[points[i][j][0]][points[i][j][1]] = b[prev_end[0]][np.minimum(prev_end[1], points[i][j][1])]
					s[points[i][j][0]][points[i][j][1]] += s[prev_end[0]][u[points[i][j][0]][points[i][j][1]]]
			
			# keeping track of the best sets from the new row for later use (what b and u are for)
			row_max = -10000
			row_max_index = -1
			for j in range(rlen[i]):
				curr = s[points[i][j][0]][points[i][j][1]]
				if curr > row_max:
					row_max = curr
					row_max_index = points[i][j][1]
				b[points[i][j][0]][points[i][j][1]] = row_max_index

			# updating the global optimal set
			if row_max > max:
				max = row_max
				max_index[0] = i
				max_index[1] = row_max_index
		
		# finding the set of points that generated the global optimum
		pointers = []
		pointers.append(max_index[1])
		for i in range(max_index[0], 0, -1):
			pointers.append(u[points[i][0][0]][pointers[max_index[0]-i]])
		pointers = np.flip(pointers, axis=0)
		
		# finding the set of points of the upper and lower partitions defined by the optimal set
		upper_points = []
		lower_points = []
		up_num = 0
		low_num = 0
		for i in range(clen):
			urow = []
			lrow = []
			for j in range(rlen[i]):
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

		# updating the final result and prepping the new datasets to have mean 0
		for i in range(len(upper_points)):
			for j in range(len(upper_points[i])):
				res[upper_points[i][j][0]][upper_points[i][j][1]] += max/up_num
				data[upper_points[i][j][0]][upper_points[i][j][1]] -= max/up_num
		for i in range(len(lower_points)):
			for j in range(len(lower_points[i])):
				res[lower_points[i][j][0]][lower_points[i][j][1]] -= max/low_num
				data[lower_points[i][j][0]][lower_points[i][j][1]] += max/low_num
		
		# recursion (if the optimal set is the current one, stop since at this point 
		# the mean of the selected elements is optimal over them)
		if up_num != size:
			partition(data, s, b, u, res, upper_points, up_num, depth-1)
		if low_num != size:
			partition(data, s, b, u, res, lower_points, low_num, depth-1)
	else:
		return

Y = np.load('galaxy1.npy')

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
axs[0][0].imshow(np.log(Y), vmin=4, vmax=8)
axs[0][1].imshow(np.log(peak(Y,15,15)), vmin=4, vmax=8)
plt.show()
