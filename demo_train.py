from morning_glory import  *
import numpy as np
import time

Config.training_set_dir = './training'
Config.output_size = 41
epoch = 100
recognizer = FaceRecognizer()
recognizer.init_parameters()
start_time = time.time()
recognizer.train(epoch=epoch)
total_time = time.time() - start_time
recognizer.save_parameters('face_params_2019_01_28.pickle')
print("Done, time spent = {} for {} epoches".format(total_time,epoch))


