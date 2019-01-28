import pyximport; pyximport.install()
import numpy as np
from morning_glory import  *
import os,time


recognizer = RNN_FaceRecognizer()
recognizer.load_parameters('params.pickle')
rootDir = './testing/faces'
with open('result.txt','w') as outfile:
	start_time = time.time()
	for root, dirs, files in os.walk(rootDir):
		for _f in files:
			_fname = os.path.join(root,_f)
			if '.pgm' in _fname :
				qry_img = Image.open(_fname)
				pred1 = recognizer.predict(qry_img)
				line = str(np.argmax(pred1)) + ","+_f+'\r\n'
				outfile.write(line)
	total_time = time.time() - start_time
	print("Total time spent = {}".format(total_time))


