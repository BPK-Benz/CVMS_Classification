import os
import cv2
import numpy as np
import pandas as pd
import colorsys
import calculation as calc

# make color dictioanary
def make_colors(variation = 9):
	hues = np.linspace(0, 1, variation+1)[:-1]
	colors = [[int(255*c) for c in colorsys.hsv_to_rgb(h, 1, 1)] for h in hues]
	return colors
colors = make_colors(19)

def read_anno(dataframe):
	labels = {}
	for index, row in dataframe.iterrows():
		if not row['filename'] in labels:
			labels[row['filename']] = [None] * 19
		labels[row['filename']][int(row['region_id'])] = np.array([
					eval(row['region_shape_attributes'])['cx'],
					eval(row['region_shape_attributes'])['cy'],		
				])
	return labels

class App:

	def __init__(self):

		# set title
		self.title = 'default'
		cv2.namedWindow (self.title, flags=cv2.WINDOW_AUTOSIZE)

		# configs
		self.mark = 'index'
		self.scale = 0.5

		# load labels
		labels_path = '1-50.csv'
		self.labels = read_anno(pd.read_csv(labels_path))

		# load images list
		self.images_path = 'labeled/'
		self.files = sorted([f for f in os.listdir(self.images_path) if f.endswith('.jpg')])
		self.total = len(self.files)
		self.index = 0

		# load 1st image
		self.load_image()

	def load_image(self):

		self.filename = self.files[self.index]
		cv2.setWindowTitle(self.title, self.filename)

		image_path = os.path.join(self.images_path, self.filename)
		self.image = cv2.imread(image_path)

		self.h, self.w = self.image.shape[:2]
		self.black1 = np.zeros([self.h, self.w], dtype=np.uint8)

		self.points = self.labels[self.filename]
		self.properties = self.get_data()
		self.refresh()

	def get_data(self):
		p = self.points.copy()
		if any(v is None for v in p): 
			print(key, 'has a missing point.', [i for i in range(19) if p[i] is None])
		data = {
			'distance 1-2': calc.distance(p[12], p[13]),
			'angle 1-2-3': calc.angle(p[0], p[1], p[2]),
			'height c2': calc.concave_height(p[0], p[2], p[4]),
			'parabola c2': calc.parabola([p[0], p[1], p[2], p[3], p[4]])
		}
		print('-'*80)
		print('filename', self.filename)
		for key in data:
			print('{:<15} {:+.2f}'.format(key, data[key]))
		return data

	def refresh(self):
		self.canvas = self.image.copy()
		self.draw()
		self.display()

	def draw(self):
			
		for i in range(len(self.points)):
			point = self.points[i]
			if self.mark == 'index':
				font = cv2.FONT_HERSHEY_SIMPLEX
				fontScale = 0.3
				lineType = 1
				cv2.putText(self.canvas, str(i),
					point, font, fontScale,
					colors[i], lineType)
			elif self.mark == 'circle':
				self.canvas = cv2.circle(self.canvas, point, 1, colors[i], -1)

	def display(self):
		x1 = 0
		y1 = int(self.h * (1 - self.scale))
		x2 = self.w
		y2 = self.h
		display = self.canvas[y1:y2, x1:x2]
		cv2.imshow(self.title, display)


if __name__ == "__main__":

	app = App()

	while True:

		key = cv2.waitKeyEx(0)
		if key in [ord('q'), 27]:
			break
		elif key in [ord('a'), 65361]:
			app.index = sorted([0, app.index-1, app.total-1])[1]
			app.load_image()
		elif key in [ord('d'), 65363]:
			app.index = sorted([0, app.index+1, app.total-1])[1]
			app.load_image()
		elif key in [ord(' ')]:
			app.mark = {'index': 'circle', 'circle': 'index'}[app.mark]
			app.refresh()
