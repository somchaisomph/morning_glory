import numpy as np
from morning_glory import Config,Recognizer

conf = Config()
conf.training_set_dir = '../../../orl_faces'
conf.num_image = 8
conf.image_subfix = 'pgm'
conf.image_shape = [112,92]
conf.output_size = 40
conf.hidden_layers = 128
conf.learning_rate = 0.001

recognizer = Recognizer()
recognizer.set_config(conf)
recognizer.init_parameters()
recognizer.train(epoch=10,silence=False)
recognizer.save_parameters("parameters.pickle")
