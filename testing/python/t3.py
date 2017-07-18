import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

dx = [1,1,0,-1,-1,-1,0,1]
dy = [0,1,1,1,0,-1,-1,-1]

def fit(data):
	"""Please, squares only"""
	n = int((data.shape[0] - 1)/2)
	# Making Data structures
	block_sums = np.zeros((8,n-1,n-1))
	best = np.zeros((8,n-1,n-1))
	hot_spots = np.zeros((8,n-1,n-1), dtype='int')
	zip_data = np.zeros((8,n,n))
	size = data.shape[0]*data.shape[1]
	mean = data.sum()/size
	result = np.ones_like(data)*mean
	data = data - mean

	# Restructuring Data
	cData = [data[n][n]]
	cRes = [mean]
	center = True
	axes = []
	aData = []
	for s in range(8):
		axes.append([0,n-1])
		axis = []
		axis2 = []
		for i in range(1,n+1):
			axis.append(data[n + dx[s]*i][n + dy[s]*i])
		aData.append(np.array(axis))
	aData = np.array(aData)
	aRes = np.ones_like(aData)*mean
	sData = np.zeros((8,n-1,n-1))
	points = []
	for s in range(8):
		sector = []
		for i in range(1,n):
			row = []
			for j in range(1,n):
				cx = n + dx[s]*i + dx[(s+1)%8]*j
				cy = n + dy[s]*i + dy[(s+1)%8]*j
				if cx >= 0 and cx < data.shape[0] and cy >= 0 and cy < data.shape[0]:
					sData[s][i-1][j-1] = data[cx][cy]
					row.append((i-1,j-1))
				else:
					sData[s][i-1][j-1] = 0
			sector.append(row)
		points.append(sector)
	sRes = np.ones_like(sData)*mean
	print((data*100).astype('int'))
	print(sData)
	depth = 10
	partition(cData, aData, sData, cRes, aRes, sRes, center, axes, points, block_sums, best, hot_spots, zip_data, size, depth)
	
	# Repack Data
	result = np.zeros_like(data)
	result[n][n] = cRes[0]
	for s in range(8):
		for i in range(1,n+1):
			result[n + dx[s]*i][n + dy[s]*i] = aRes[s][i-1]
		for i in range(1,n):
				for j in range(1,n):
					cx = n + dx[s]*i + dx[(s+1)%8]*j
					cy = n + dy[s]*i + dy[(s+1)%8]*j
					if cx >= 0 and cx < data.shape[0] and cy >= 0 and cy < data.shape[0]:
						result[cx][cy] = sRes[s][i-1][j-1]
	"""
	result = np.zeros_like(data)
	result[n][n] = cData[0]
	for s in range(8):
		for i in range(1,n+1):
			result[n + dx[s]*i][n + dy[s]*i] = aData[s][i-1]
		for i in range(1,n):
				for j in range(1,n):
					cx = n + dx[s]*i + dx[(s+1)%8]*j
					cy = n + dy[s]*i + dy[(s+1)%8]*j
					if cx >= 0 and cx < data.shape[0] and cy >= 0 and cy < data.shape[0]:
						result[cx][cy] = sData[s][i-1][j-1]
	print(result.astype('int'))
	"""
	return result

def partition(cData, aData, sData, cRes, aRes, sRes, center, axes, points, block_sums, best, hot_spots, zip_data, size, depth):
	
	if depth >= 0 and size > 1:
		print('PPPPPPPPPPPPPPPPPPPPPPPPartion Call')
		print("DOMAIN")
		print("Center")
		print(center)
		print("Axes")
		print(axes)
		print("Points")
		print(points)
		print("DATA")
		print(aData)
		print(sData)
		n = aData.shape[1]
		for s in range(8):
			start = np.ones(n)*(n)
			end = np.ones(n, dtype='int')*-1
			for i in range(len(points[s])):
				start[points[s][i][0][0]+1] = points[s][i][0][1]
				end[points[s][i][len(points[s][i]) - 1][0]] = points[s][i][len(points[s][i]) - 1][1]
				for j in range(len(points[s][i])):
					block_sums[s][points[s][i][j][0]][points[s][i][j][1]] = sData[s][points[s][i][j][0]][points[s][i][j][1]]
					best[s][points[s][i][j][0]][points[s][i][j][1]] = 0
					if points[s][i][j][1] >= start[i]:
						block_sums[s][points[s][i][j][0]][points[s][i][j][1]] += block_sums[s][points[s][i][j][0] - 1][points[s][i][j][1]]
						best[s][points[s][i][j][0]][points[s][i][j][1]] += best[s][points[s][i][j][0] - 1][points[s][i][j][1]]
					if points[s][i][j][1] > start[i]:
						block_sums[s][points[s][i][j][0]][points[s][i][j][1]] -= block_sums[s][points[s][i][j][0] - 1][points[s][i][j][1] - 1]
						best[s][points[s][i][j][0]][points[s][i][j][1]] -= best[s][points[s][i][j][0] - 1][points[s][i][j][1] - 1]
					if j > 0:
						block_sums[s][points[s][i][j][0]][points[s][i][j][1]] += block_sums[s][points[s][i][j][0]][points[s][i][j][1] - 1]
						best[s][points[s][i][j][0]][points[s][i][j][1]] += best[s][points[s][i][j][0]][points[s][i][j][1] - 1]
					if block_sums[s][points[s][i][j][0]][points[s][i][j][1]] >= best[s][points[s][i][j][0]][points[s][i][j][1]]:
						best[s][points[s][i][j][0]][points[s][i][j][1]] = block_sums[s][points[s][i][j][0]][points[s][i][j][1]]
						hot_spots[s][points[s][i][j][0]][points[s][i][j][1]] = 1
					else:
						hot_spots[s][points[s][i][j][0]][points[s][i][j][1]] = 0
					#print("{0},{1}".format(points[s][i][j][0], axes[s][0]))
					#print(n)
					if axes[s][0] != axes[s][1]:
						z_i = np.maximum(points[s][i][j][0], axes[s][0])
					else:
						z_i = n-1
					if axes[(s+1)%8][0] != axes[(s+1)%8][1]:
						z_j = np.maximum(points[s][i][j][1], axes[(s+1)%8][0])
					else:
						z_j = n-1
					zip_data[s][z_i][z_j] = np.maximum(zip_data[s][z_i][z_j], best[s][points[s][i][j][0]][points[s][i][j][1]])
					
			#print(s)
			#print(start)
			#print("END")
			#print(end)

			#extend zip_data to full axes
			for i in range(axes[s][0], axes[s][1]):
				for j in range(np.maximum(axes[(s+1)%8][0]+1, end[i]+1), axes[(s+1)%8][1]):
					zip_data[s][i][j] = 0
					if end[i] == -1:
						if i > axes[s][0]:
							zip_data[s][i][j] = zip_data[s][i-1][j]
					else:
						if i > axes[s][0]:
							zip_data[s][i][j] += zip_data[s][i-1][j]
						if j > axes[(s+1)%8][0]:
							zip_data[s][i][j] += zip_data[s][i][j-1]
							if i > axes[s][0]:
								zip_data[s][i][j] -= zip_data[s][i-1][j-1]
		print("zip_data")
		print(zip_data)
		#zipper
		result = []
		for s in range(8):
			result.append(axes[s][1]-1)
		for s_mod in range(16):
			s = s_mod % 8
			max = 0
			axis_sum = 0
			result[s] = -1
			if axes[s][0] != axes[s][1]:
				for r in range(axes[s][0],axes[s][1]):
					axis_sum += aData[s][r]
					if result[(s-1)%8] == -1:
						c1 = 0
					else:
						temp = result[(s-1)%8]
						if axes[(s-1)%8][0] == axes[(s-1)%8][1]:
							temp = n - 1
						c1 = zip_data[(s-1)%8][temp][r]
					if result[(s+1)%8] == -1:
						c2 = 0
					else:
						temp = result[(s+1)%8]
						if axes[(s+1)%8][0] == axes[(s+1)%8][1]:
							temp = n - 1
						c2 = zip_data[s][r][temp]
					curr = c1 + c2 + axis_sum
					#print("{0},{1},{2},{3}".format(result,r,curr,s))
					if curr > max:
						max = curr
						result[s] = r
			else:
				result[s] = axes[s][0]

		#print(result)
		max_sum = 0
		count = 0
		if center:
			count += 1
			max_sum += cData[0]
		for s in range(8):
			if result[s] > -1 and result[(s+1)%8] > -1:
				max_sum += zip_data[s][result[s]][result[(s+1)%8]] 
			for r in range(axes[s][0],result[s]+1):
				count += 1
				max_sum += aData[s][r]

		for s in range(8):
			extent = -1
			for i in range(len(points[s])-1,-1,-1):
				for j in range(len(points[s][i])-1,-1,-1):
					if hot_spots[s][points[s][i][j][0]][points[s][i][j][1]] == 1:
						extent = np.maximum(points[s][i][j][1], extent)
					if points[s][i][j][1] <= extent and points[s][i][j][0] <= result[s] and points[s][i][j][1] <= result[(s+1)%8]:
						count += 1

		for s in range(8):
			for i in range(axes[s][0], axes[s][1]):
				for j in range(axes[(s+1)%8][0], axes[(s+1)%8][1]):
					zip_data[s][i][j] = 0

		print("block_sums")
		print(block_sums)
		print("best")
		print(best)
		#print("max_sum")
		#print(max_sum)
		#New data bounds for next iterations and make new sets mean 0 and update result
		u_center = center
		l_center = False
		
		if center:
			cData[0] -= max_sum/count
			cRes[0] += max_sum/count
		
		l_axes = []
		u_axes = []
		for s in range(8):
			u_axes.append((axes[s][0],result[s]+1))
			for r in range(axes[s][0],result[s]+1):
				aData[s][r] -= max_sum/count
				aRes[s][r] += max_sum/count
			l_axes.append((result[s]+1,axes[s][1]))
			for r in range(result[s]+1,axes[s][1]):
				aData[s][r] += max_sum/(size - count)
				aRes[s][r] -= max_sum/(size - count)

		#print("hot_spots")
		#print(hot_spots)
		l_points = []
		u_points = []
		for s in range(8):
			l_sector = []
			u_sector = []
			extent = -1
			for i in range(len(points[s])-1,-1,-1):
				l_row = []
				u_row = []
				for j in range(len(points[s][i])-1,-1,-1):
					if hot_spots[s][points[s][i][j][0]][points[s][i][j][1]] == 1:
						hot_spots[s][points[s][i][j][0]][points[s][i][j][1]] = 0
						extent = np.maximum(points[s][i][j][1], extent)
					if points[s][i][j][1] <= extent and points[s][i][j][0] <= result[s] and points[s][i][j][1] <= result[(s+1)%8]:
						u_row.append(points[s][i][j])
						sData[s][points[s][i][j][0]][points[s][i][j][1]] -= max_sum/count
						sRes[s][points[s][i][j][0]][points[s][i][j][1]] += max_sum/count
					else:
						l_row.append(points[s][i][j])
						sData[s][points[s][i][j][0]][points[s][i][j][1]] += max_sum/(size - count)
						sRes[s][points[s][i][j][0]][points[s][i][j][1]] -= max_sum/(size - count)
				if len(l_row) > 0:
					l_sector.append(list(reversed(l_row)))
				if len(u_row) > 0:
					u_sector.append(list(reversed(u_row)))
			l_points.append(list(reversed(l_sector)))
			u_points.append(list(reversed(u_sector)))
		#print(u_points)
		print("BLOCK_SUMS")
		#print(block_sums)
		print("BEST")
		#print(best)
		print("HOT_SPOTS")
		#print(hot_spots)
		print("AXES")
		#print(axes)
		print("ZIP_DATA")
		#print(zip_data)
		print("S_DATA")
		#print(sRes)
		if count != size:
			partition(cData, aData, sData, cRes, aRes, sRes, u_center, u_axes, u_points, block_sums, best, hot_spots, zip_data, count, depth-1)
		if count != 0:
			#partition(cData, aData, sData, cRes, aRes, sRes, l_center, l_axes, l_points, block_sums, best, hot_spots, zip_data, count, depth-1)
			pass

def add_noise(Y, sigma):
	"""Adds noise to Y"""
	return Y + np.random.normal(0, sigma, Y.shape)

np.random.seed(2)
"""
x, y = np.mgrid[-1.6:2:.4, -1.6:2:.4]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([0, 0], [[1.0, 0.5], [0.5, 1.0]])
data = add_noise(rv.pdf(pos), 0.05)
data = (data*100).astype(int)
"""
x, y = np.mgrid[-1.6:1.6:.05, -1.6:1.6:.05]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([0, 0], [[1.0, 0.5], [0.5, 1.0]])
data = add_noise(rv.pdf(pos), 0.05)
data = (data*100).astype(int)

result = fit(data)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
axs[0][0].imshow(data, vmin=0, vmax=20)
axs[0][1].imshow(result, vmin=0, vmax=20)
plt.show()
