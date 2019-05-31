from pathlib import Path
from PIL import Image
import numpy as np
import pickle
# -- By Somchai Somphadung
# -- Good idea from http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# -- updated 2019-05-31

class Config():
    training_set_dir = '../faces/att/'
    num_image = 8
    image_subfix = 'pgm'
    image_shape = [92,112]
    output_size = 40

class Activator():
    #activations
    def softmax(X):
        c = np.clip(X,-700,700) #float64 maximum exponentiable value
        e = np.exp(c)
        return e / np.sum(e,axis=1,keepdims=True)

    def cross_entropy(out,label):
        entropy = label * np.log(out + 1e-6) #prevent log value overflow
        return -np.sum(entropy,axis=1,keepdims=True)

    def sigmoid(X):
        c = np.clip(X,-700,700)
        return 1 / (1 + np.exp(-c))

    def deriv_sigmoid(out):
        return out * (1 - out)

    def tanh(X):
        c = np.clip(X,-350,350)
        return 2 / (1 + np.exp(-2 * c)) - 1

    def deriv_tanh(out):
        return 1 - np.square(out)
    
class Helper():
    
    def eigen_vect(gray):
        #to find eigen vector for gray, gray must be sqaure vector 
        
         np.linalg.eig(gray)
    
    def whitening(gray):
        _mean = np.mean(gray)
        _std = np.std(gray)
        return (gray - _mean)/_std
    
    def get_sample_code(folder_name):
        return folder_name[1:]  
    
    def get_np_image(gray_img):
        #convert grayscale to numpy array
        _img_np = np.array(gray_img) 
        _mean = np.mean(_img_np)
        _std = np.std(_img_np)    
        #create zero-center image then send out
        return (_img_np - _mean)/_std 
        #return _img_np
    
    def get_folder_list(path):
        fd_path = Path(Config.training_set_dir)
        return [x for x in fd_path.iterdir() if (x.is_dir()) and ( x.name[0] == 's')] 
    
    def get_sample_code(folder_name):
        return folder_name[1:]   

    def get_training_data(index,folder_list,whitening=True):
        assert index < Config.num_image
        X = []
        y = []
        _num_folder = len(folder_list)
        for f in range(_num_folder): # traverse folder list        
            _folder = folder_list[f]
            _label = np.zeros((Config.output_size))
            _sample_code = Helper.get_sample_code(_folder.name)
            _label[int(_sample_code)-1] = 1
            y.append(_label)  
        
                    
            _img_file = Helper.get_file_list(_folder)[index]        
            _gray = Image.open(_img_file)
            
            # convert PIL image to ndarray
            _np_arr = np.array(_gray)             
            if whitening :
                _whiten_img = Helper.whitening(_np_arr)
                
                X.append(_whiten_img)
            else :
                X.append(_np_arr)
            

        return (np.array(X),np.array(y))
    
    def get_file_list(path_tree):
        fd_files = Path(path_tree)
        file_list = list(fd_files.glob('**/*.{0}'.format(Config.image_subfix)))
        return file_list


class Recognizer():
    def __init__(self):
        self.OUTPUT_SIZE = Config.output_size
        self.HIDDEN = 128
        self.INPUT_SIZE = Config.image_shape[0] + self.HIDDEN # we need to concate HIDDEN to INPUT
        self.ALPHA = 0.001  
        
    def save_parameters(self,param_file):
        p = [self.Wf,self.Wi,self.Wc,self.Wo,self.Wy,self.bf,self.bi,self.bc,self.bo,self.by]
        with open(param_file,'wb') as handle :
            pickle.dump(p,handle)
    
    def load_parameters(self,param_file):    
        p = None
        with open(param_file, 'rb') as handle:
            p = pickle.load(handle)
            self.Wf = p[0]
            self.Wi = p[1]
            self.Wc = p[2]
            self.Wo = p[3]
            self.Wy = p[4]

            self.bf = p[5]
            self.bi = p[6]
            self.bc = p[7]
            self.bo = p[8]
            self.by = p[9]
            
            # for derivative
            self.dWf = np.zeros_like(self.Wf)
            self.dWi = np.zeros_like(self.Wi)
            self.dWc = np.zeros_like(self.Wc)
            self.dWo = np.zeros_like(self.Wo)
            self.dWy = np.zeros_like(self.Wy)

            self.dbf = np.zeros_like(self.bf)
            self.dbi = np.zeros_like(self.bi)
            self.dbc = np.zeros_like(self.bc)
            self.dbo = np.zeros_like(self.bo)
            self.dby = np.zeros_like(self.by)
            
    
    def init_parameters(self):
        self.Wf = np.random.randn(self.INPUT_SIZE,self.HIDDEN)/np.sqrt(self.INPUT_SIZE/2) 
        self.Wi = np.random.randn(self.INPUT_SIZE,self.HIDDEN)/np.sqrt(self.INPUT_SIZE/2) 
        self.Wc = np.random.randn(self.INPUT_SIZE,self.HIDDEN)/np.sqrt(self.INPUT_SIZE/2) 
        self.Wo = np.random.randn(self.INPUT_SIZE,self.HIDDEN)/np.sqrt(self.INPUT_SIZE/2) 
        self.Wy = np.random.randn(self.HIDDEN,self.OUTPUT_SIZE)/np.sqrt(self.HIDDEN/2)

        self.bf = np.zeros(self.HIDDEN)
        self.bi = np.zeros(self.HIDDEN)
        self.bc = np.zeros(self.HIDDEN)
        self.bo = np.zeros(self.HIDDEN)
        self.by = np.zeros(self.OUTPUT_SIZE)

        # for derivative
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dWy = np.zeros_like(self.Wy)

        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        self.dby = np.zeros_like(self.by)
        
    def LSTM_Cell(self,X):
        batch_num = X.shape[1]
        caches = []
        states = []
        states.append([np.zeros([batch_num,self.HIDDEN]),
                    np.zeros([batch_num,self.HIDDEN])])
        for x in X:
            c_prev , h_prev = states[-1]
            x = np.column_stack([x,h_prev])
            hf = Activator.sigmoid(np.dot(x,self.Wf) + self.bf) # forget gate
            hi = Activator.sigmoid(np.dot(x,self.Wi) + self.bi) # new information gate
            ho = Activator.sigmoid(np.dot(x,self.Wo) + self.bo) # output gate
            hc = Activator.tanh(np.dot(x, self.Wc) + self.bc)
        
            c = hf * c_prev + hi * hc # compute current output
            h = ho * Activator.tanh(c) # compute current cell state
        
            states.append([c,h])
            caches.append([x,hf,hi,ho,hc])
        return caches,states
    
    
    def train(self,epoch=100,alpha=0.001):
        self.ALPHA = alpha  
        folder_list = Helper.get_folder_list(Config.training_set_dir)
        
        for e in range(epoch):
            for p in range(Config.num_image ):
                # Forward 
                X, Y = Helper.get_training_data(p,folder_list)
                Xt = X.transpose(1,0,2)
                caches, states = self.LSTM_Cell(Xt)    
                c, h = states[-1]
                out = np.dot(h, self.Wy) + self.by
                
                pred = Activator.softmax(out)    
                entropy = Activator.cross_entropy(pred, Y)
                
                # Backpropagation Through Time
                dout = pred - Y
                self.dWy = np.dot(h.T, dout)
                self.dby = np.sum(dout, axis=0)
    
                dc_next = np.zeros_like(c)
                dh_next = np.zeros_like(h)
            
                for t in range(Xt.shape[0]):
                    c, h = states[-t-1]
                    c_prev, h_prev = states[-t-2]
        
                    x, hf, hi, ho, hc = caches[-t-1]
        
                    tc = Activator.tanh(c)
                    dh = np.dot(dout, self.Wy.T) + dh_next
        
                    dc = dh * ho * Activator.deriv_tanh(tc)
                    dc = dc + dc_next
        
                    dho = dh * tc
                    dho = dho * Activator.deriv_sigmoid(ho)
        
                    dhf = dc * c_prev
                    dhf = dhf * Activator.deriv_sigmoid(hf)
        
                    dhi = dc * hc
                    dhi = dhi * Activator.deriv_sigmoid(hi)
        
                    dhc = dc * hi
                    dhc = dhc * Activator.deriv_tanh(hc)
        
                    self.dWf += np.dot(x.T, dhf)
                    self.dbf += np.sum(dhf, axis=0)
                    dXf = np.dot(dhf, self.Wf.T)
        
                    self.dWi += np.dot(x.T, dhi)
                    self.dbi += np.sum(dhi, axis=0)
                    dXi = np.dot(dhi, self.Wi.T)
        
                    self.dWo += np.dot(x.T, dho)
                    self.dbo += np.sum(dho, axis=0)
                    dXo = np.dot(dho, self.Wo.T)
        
                    self.dWc += np.dot(x.T, dhc)
                    self.dbc += np.sum(dhc, axis=0)
                    dXc = np.dot(dhc, self.Wc.T)
        
                    dX = dXf + dXi + dXo + dXc
        
                    dc_next = hf * dc
                    dh_next = dX[:, -self.HIDDEN:]
                
                self.update_param()
                self.reset()
                print('Iter {}'.format((Config.num_image ) * e + p))
                print('entropy', np.sum(entropy))
                print('----------')
    
    def update_param(self):
        # Update weights
        self.Wf -= self.ALPHA * self.dWf
        self.Wi -= self.ALPHA * self.dWi
        self.Wc -= self.ALPHA * self.dWc
        self.Wo -= self.ALPHA * self.dWo
        self.Wy -= self.ALPHA * self.dWy
    
        self.bf -= self.ALPHA * self.dbf
        self.bi -= self.ALPHA * self.dbi
        self.bc -= self.ALPHA * self.dbc
        self.bo -= self.ALPHA * self.dbo
        self.by -= self.ALPHA * self.dby
    
    def reset(self):    
        # Initialize delta values
        self.dWf *= 0
        self.dWi *= 0
        self.dWc *= 0
        self.dWo *= 0
        self.dWy *= 0
    
        self.dbf *= 0
        self.dbi *= 0
        self.dbc *= 0
        self.dbo *= 0
        self.dby *= 0
        
    def predict(self,query_img):
        # input is gray scale image        
        _np_img = np.array(query_img)
        _mean = np.mean(_np_img)
        _std = np.std(_np_img)
        Xtest =  (_np_img - _mean)/_std 
        Xtest = np.transpose(np.reshape(Xtest,(1,Xtest.shape[0],Xtest.shape[1])),(1,0,2))
        caches, states = self.LSTM_Cell(Xtest)    
        c, h = states[-1]
        out = np.dot(h, self.Wy) + self.by
        pred = Activator.softmax(out)
        return pred

