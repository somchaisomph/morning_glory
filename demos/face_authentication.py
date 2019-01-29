import pygame
import pygame.camera
from pygame.locals import *
import sys
from PIL import Image
import numpy as np
from morning_glory import  *
import os,time
import threading
import queue
import signal

WINDOW=(480,360)
center = (WINDOW[0]//2,WINDOW[1]//2)
IMAGE = (92,112)
SCALE = 2
COLOR = (150, 100, 0)
THICK = 3
c_image = 1

class Predictor(threading.Thread):
	def __init__(self,q):
		threading.Thread.__init__(self)
		self.q = q
		self.recognizer = Recognizer()
		self.recognizer.load_parameters('face_params_2019_01_29.pickle')
		self.shutdown_flag = threading.Event()	
		
	def run(self):
		while not self.shutdown_flag.is_set():
			if not self.q.empty() :
				data = q.get_nowait()
				pred = np.argmax(self.recognizer.predict(data))
				print(pred)
        #authentication is down here.
				

	
#-- Signals
class ServiceExit(Exception):
	"""
	Custom exception which is used to trigger the clean exit
	of all running threads and the main program.
	"""
	pass		

def service_shutdown(signum, frame):
	print('Caught signal %d' % signum)
	raise ServiceExit
		

# 1) -- Initialize
pygame.init()
pygame.camera.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption("Morning Glory Face Authentication ")


# 2) -- get camera
cameras = pygame.camera.list_cameras()
camera = pygame.camera.Camera(cameras[0],WINDOW,'RGB')
camera.start()

# 3) -- grab first iamge
img = camera.get_image()

# 4) -- define area for capture data
rect = [(center[0]-IMAGE[0]*SCALE//2,center[1]-IMAGE[1]*SCALE//2),
		(center[0]+IMAGE[0]*SCALE//2,center[1]-IMAGE[1]*SCALE//2),
		(center[0]+IMAGE[0]*SCALE//2,center[1]+IMAGE[1]*SCALE//2),
		(center[0]-IMAGE[0]*SCALE//2,center[1]+IMAGE[1]*SCALE//2)]

# 5) -- utilities
def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return np.array(gray,dtype=np.uint8)
	
def get_2darr(img):
	# clip out the pre-definded area
	clip = img.subsurface((center[0]-IMAGE[0]*SCALE//2,
							center[1]-IMAGE[1]*SCALE//2,
							IMAGE[0]*SCALE,IMAGE[1]*SCALE))
	
	# scale down to prefered size
	scaled = pygame.transform.scale(clip,IMAGE)
	
	# convert to 3D array (RGB)
	arr_3d = pygame.surfarray.array3d(scaled)
	
	# actually 2D array is needed 
	# convert column-major style of pygame array to row-major style of Numpy
	arr_2d = rgb2gray(arr_3d.swapaxes(0,1))
	
	return arr_2d	
	
q = queue.Queue()
job1  = Predictor(q)
job1.start()

while True :
	clock.tick(30)
	for e in pygame.event.get() :
		if e.type == pygame.QUIT :
			camera.stop()
			job1.shutdown_flag.set()
			job1.join()
			sys.exit()
		elif e.type == KEYDOWN :
			if e.key == K_ESCAPE :	
				camera.stop()
				job1.shutdown_flag.set()
				job1.join()
				sys.exit()
			#elif e.key == K_p and pygame.key.get_mods() & pygame.KMOD_SHIFT:
			elif e.key == K_F1:
				arr = get_2darr(img)							
				q.put_nowait(arr)
				q.task_done()				
				
     
	# draw frame
	
	screen.blit(img, (0,0))	
	pygame.draw.lines(screen,COLOR , True, rect, THICK)
	pygame.display.flip()
	# grab next frame    
	img = camera.get_image()

