import numpy as np

def concave_height(a, b, c):
	if any(v is None for v in [a, b, c]): 
		return None
	delta_yx = (c[1]-a[1])/(c[0]-a[0])
	intercept = c[1]-delta_yx*c[0]
	perpendicular = np.abs(-delta_yx*b[0]+b[1]-intercept)/np.sqrt( [np.power(delta_yx,2)+1] )
	return perpendicular[0]

def distance(a, b):
	if any(v is None for v in [a, b]): 
		return None
	return np.linalg.norm(a-b)

def angle(a, b, c):
	if any(v is None for v in [a, b, c]): 
		return None
	ba = a-b
	bc = c-b
	# find b angle
	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
	angle = np.arccos(cosine_angle)
	return np.degrees(angle)

def parabola(data):
	if any(v is None for v in data): 
		return None
	x, y = map(np.array, zip(*data))
	if x.std() != 0:
		x_normalized = (x - x.mean())/x.std()
	else:
		x_normalized = 0

	if y.std() != 0:
		y_normalized = (y - y.mean())/y.std()
	else:
		y_normalized = 0

	fit_normalized = np.polyfit(x_normalized, y_normalized, 2)
	# the higher the narrow
	return fit_normalized[0]

def slope(a,b):
	if any(v is None for v in [a, b]): 
		return None

	slope, intercept = np.polyfit(a,b,1)
	return slope

def ratio (a,b):
	if any(v is None for v in [a,b]): 
		return None
	if b == 0:
		ratio = 0
	else:
		ratio = a/b
	return ratio