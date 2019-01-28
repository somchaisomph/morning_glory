import pyximport; pyximport.install()
from bluepy import btle
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
		self.recognizer = FaceRecognizer()
		self.recognizer.load_parameters('test_params.pickle')
		self.shutdown_flag = threading.Event()
		self.init_bt()
	
	def init_bt(self):
		BT_ADDR = '30:AE:A4:F0:3D:9E'
		SERVICE_UUID = "2a24496e-ac00-4390-8bac-18bc7f9d07de"
		CHARACTERISTIC_UUID = "1e3a7428-fd79-49ce-9807-7f6e8a2a9bc9"
		self.p = btle.Peripheral(BT_ADDR,iface=0)
		svc = self.p.getServiceByUUID(SERVICE_UUID)
		self.ch = svc.getCharacteristics(CHARACTERISTIC_UUID)[0]
		
	def run(self):
		while not self.shutdown_flag.is_set():
			if not self.q.empty() :
				data = q.get_nowait()
				#print(data.shape)
				pred = np.argmax(self.recognizer.predict_arr(data))
				print(pred)
				if pred == 40 :
					self.ch.write(b'1')
				else:
					self.ch.write(b'2')
		self.p.disconnect()

	
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
#palette = tuple([(i, i, i) for i in range(256)])


# 2) -- get camera
cameras = pygame.camera.list_cameras()
camera = pygame.camera.Camera(cameras[0],WINDOW,'RGB')
camera.start()

# 3) -- grab first iamge
img = camera.get_image()

#screen = pygame.display.set_mode( ( WIDTH, HEIGHT ) )
pygame.display.set_caption("pyGame Camera View")

rect = [(center[0]-IMAGE[0]*SCALE//2,center[1]-IMAGE[1]*SCALE//2),
		(center[0]+IMAGE[0]*SCALE//2,center[1]-IMAGE[1]*SCALE//2),
		(center[0]+IMAGE[0]*SCALE//2,center[1]+IMAGE[1]*SCALE//2),
		(center[0]-IMAGE[0]*SCALE//2,center[1]+IMAGE[1]*SCALE//2)]

def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return np.array(gray,dtype=np.uint8)


def save_image(img,dim=(92,112),encode="JPEG",name="test.jpg"):
	# set clip on surface
	clip = img.subsurface((center[0]-IMAGE[0]*SCALE//2,
							center[1]-IMAGE[1]*SCALE//2,
							IMAGE[0]*SCALE,IMAGE[1]*SCALE))
	pil_string_image = pygame.image.tostring(clip,"RGBA",False)
	im = Image.frombytes("RGBA",
						(IMAGE[0]*SCALE,IMAGE[1]*SCALE),
						pil_string_image)
	#im = Image.frombytes("L",(276,336),pil_string_image)
	gray = im.convert('L').resize(dim)
	gray.save(name,encode)
	
def get_2darr(img):
	# clip out the pre-definded area
	clip = img.subsurface((center[0]-IMAGE[0]*SCALE//2,
							center[1]-IMAGE[1]*SCALE//2,
							IMAGE[0]*SCALE,IMAGE[1]*SCALE))
	
	# scale down to prefered size
	scaled = pygame.transform.scale(clip,IMAGE)
	
	# convert to 3D array (RGB)
	arr_3d = pygame.surfarray.array3d(scaled)
	
	# actually 2D is needed
	arr_2d = rgb2gray(arr_3d.swapaxes(0,1))
	
	return arr_2d
	
def predict(img):
	# set clip on surface
	clip = img.subsurface((center[0]-IMAGE[0]*SCALE//2,
							center[1]-IMAGE[1]*SCALE//2,
							IMAGE[0]*SCALE,IMAGE[1]*SCALE))
							
	# scale down to prefered size
	scaled = pygame.transform.scale(clip,IMAGE)
							
	#convert to byte						
	pil_string_image = pygame.image.tostring(scaled,"RGBA",False)
	
	#convert to PIL Image
	im = Image.frombytes("RGBA",
						IMAGE,
						pil_string_image)
	
	gray = im.convert('LA')
	#print(gray.size)
	print(recognizer.predict(gray))
	
	
q = queue.Queue()
job1  = Predictor(q)
job1.start()

recognizer = FaceRecognizer()
recognizer.load_parameters('test_params.pickle')
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
				
			elif e.key == K_F2:
				if c_image < 10 : c_image += 1
				else : c_image = 1
				save_image(img,dim=(92,112),encode="PPM",name=str(c_image)+".pgm")
				
			elif e.key == K_F3 :
				save_image(img,dim=(92,112),encode="PPM",name="qry.pgm")
				_img = Image.open('qry.pgm')
				_np_img = np.array(_img)
				print(_np_img.shape)
				
			elif e.key == K_F4 :
				arr = get_2darr(img)	
				print(arr.shape)
				
				
     
	# draw frame
	
	screen.blit(img, (0,0))	
	pygame.draw.lines(screen,COLOR , True, rect, THICK)
	pygame.display.flip()
	# grab next frame    
	img = camera.get_image()

