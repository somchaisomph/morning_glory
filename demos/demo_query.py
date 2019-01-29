import numpy as np
from PIL import Image
from morning_glory import Recognizer

recognizer = Recognizer()
recognizer.load_parameters('parameters.pickle')
qry = Image.open('/s3/1.pgm')
_np_img = np.array(qry)
pred = recognizer.predict(_np_img)
print(np.argmax(pred))
