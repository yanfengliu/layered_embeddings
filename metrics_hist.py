import numpy as np
import os
import matplotlib.pyplot as plt 

class MetricsHist(object):
    avg_w                   = np.array([])
    deeplabv3_avg_w         = np.array([])
    embedding_avg_w         = np.array([])
    loss                    = np.array([])
    ap                      = np.array([])
    ap50                    = np.array([])
    ap75                    = np.array([])
    ar1                     = np.array([])
    ar10                    = np.array([])
    ar100                   = np.array([])
    ar1000                  = np.array([])
    ar_none                 = np.array([])
    ar_heavy                = np.array([])
    ar_partial              = np.array([])
    
    def __init__(self, dt_dir):
        self.dt_dir = dt_dir
    
    def clear(self):
        self.avg_w                  = np.array([])
        self.deeplabv3_avg_w        = np.array([])
        self.embedding_avg_w        = np.array([])
        self.loss                   = np.array([])
        self.ap                     = np.array([])
        self.ap50                   = np.array([])
        self.ap75                   = np.array([])
        self.ar1                    = np.array([])
        self.ar10                   = np.array([])
        self.ar100                  = np.array([])
        self.ar1000                 = np.array([])
        self.ar_none                = np.array([])
        self.ar_heavy               = np.array([])
        self.ar_partial             = np.array([])

    def load(self):
        if (os.path.isfile(self.dt_dir + 'ap.npy')):
            self.avg_w                  = np.load(self.dt_dir + 'avg_w.npy')
            self.deeplabv3_avg_w        = np.load(self.dt_dir + 'deeplabv3_avg_w.npy')
            self.embedding_avg_w        = np.load(self.dt_dir + 'embedding_avg_w.npy')
            self.loss                   = np.load(self.dt_dir + 'loss.npy')
            self.ap                     = np.load(self.dt_dir + 'ap.npy')
            self.ap50                   = np.load(self.dt_dir + 'ap50.npy')
            self.ap75                   = np.load(self.dt_dir + 'ap75.npy')
            self.ar1                    = np.load(self.dt_dir + 'ar1.npy')
            self.ar10                   = np.load(self.dt_dir + 'ar10.npy')
            self.ar100                  = np.load(self.dt_dir + 'ar100.npy')
            self.ar1000                 = np.load(self.dt_dir + 'ar1000.npy')
            self.ar_none                = np.load(self.dt_dir + 'ar_none.npy')
            self.ar_heavy               = np.load(self.dt_dir + 'ar_heavy.npy')
            self.ar_partial             = np.load(self.dt_dir + 'ar_partial.npy')
    
    def save(self):
        np.save(self.dt_dir + 'avg_w.npy',                  self.avg_w)
        np.save(self.dt_dir + 'deeplabv3_avg_w.npy',        self.deeplabv3_avg_w)
        np.save(self.dt_dir + 'embedding_avg_w.npy',        self.embedding_avg_w)
        np.save(self.dt_dir + 'loss.npy',                   self.loss)
        np.save(self.dt_dir + 'ap.npy',                     self.ap)
        np.save(self.dt_dir + 'ap50.npy',                   self.ap50)
        np.save(self.dt_dir + 'ap75.npy',                   self.ap75)
        np.save(self.dt_dir + 'ar1.npy',                    self.ar1)
        np.save(self.dt_dir + 'ar10.npy',                   self.ar10)
        np.save(self.dt_dir + 'ar100.npy',                  self.ar100)
        np.save(self.dt_dir + 'ar1000.npy',                 self.ar1000)
        np.save(self.dt_dir + 'ar_none.npy',                self.ar_none)
        np.save(self.dt_dir + 'ar_heavy.npy',               self.ar_heavy)
        np.save(self.dt_dir + 'ar_partial.npy',             self.ar_partial)
    
    def append(self, metrics):
        self.ap         = np.append(self.ap,         metrics['both'].ap)
        self.ap50       = np.append(self.ap50,       metrics['both'].ap_05)
        self.ap75       = np.append(self.ap75,       metrics['both'].ap_075)
        self.ar1        = np.append(self.ar1,        metrics['both'].ar1)
        self.ar10       = np.append(self.ar10,       metrics['both'].ar10)
        self.ar100      = np.append(self.ar100,      metrics['both'].ar100)
        self.ar1000     = np.append(self.ar1000,     metrics['both'].ar1000)
        self.ar_none    = np.append(self.ar_none,    metrics['both'].ar_none)
        self.ar_heavy   = np.append(self.ar_heavy,   metrics['both'].ar_heavy)
        self.ar_partial = np.append(self.ar_partial, metrics['both'].ar_partial)
    
    def plot(self):
        plt.figure(figsize=(16, 4))
        plt.ylim((0, 1))
        plt.title('performance on val every 5 epochs')
        plt.grid()
        ap_handle,         = plt.plot(self.ap,         label='ap')
        ap50_handle,       = plt.plot(self.ap50,       label='ap50')
        ap75_handle,       = plt.plot(self.ap75,       label='ap75')
        ar10_handle,       = plt.plot(self.ar10,       label='ar10')
        ar100_handle,      = plt.plot(self.ar100,      label='ar100')
        ar1000_handle,     = plt.plot(self.ar1000,     label='ar1000')
        ar_none_handle,    = plt.plot(self.ar_none,    label='ar_none')
        ar_heavy_handle,   = plt.plot(self.ar_heavy,   label='ar_heavy')
        ar_partial_handle, = plt.plot(self.ar_partial, label='ar_partial')

        plt.legend([ap_handle,
                    ap50_handle, 
                    ap75_handle,
                    ar10_handle, 
                    ar100_handle, 
                    ar1000_handle, 
                    ar_none_handle, 
                    ar_heavy_handle, 
                    ar_partial_handle], 
                   ['ap',
                    'ap50', 
                    'ap75',
                    'ar10', 
                    'ar100', 
                    'ar1000', 
                    'ar_none', 
                    'ar_heavy', 
                    'ar_partial'])
        plt.show()

    def plot_weights(self):
        plt.figure(figsize=(16, 4))
        plt.title('weight absolute value sum')
        plt.grid()
        avg_w_handle,              = plt.plot(self.avg_w, label='avg_w')
        deeplabv3_avg_w_handle,    = plt.plot(self.deeplabv3_avg_w, label='deeplabv3_avg_w')
        embedding_avg_w_handle,    = plt.plot(self.embedding_avg_w, label='embedding_avg_w')
        plt.legend([avg_w_handle,
                    deeplabv3_avg_w_handle, 
                    embedding_avg_w_handle], 
                   ['avg_w',
                    'deeplabv3_avg_w',
                    'embedding_avg_w'])
        plt.show()
    
    def print(self):
        print('ap',         self.ap        [-1])
        print('ap50',       self.ap50      [-1])
        print('ap75',       self.ap75      [-1])
        print('ar1',        self.ar1       [-1])
        print('ar10',       self.ar10      [-1])
        print('ar100',      self.ar100     [-1])
        print('ar1000',     self.ar1000    [-1])
        print('ar_none',    self.ar_none   [-1])
        print('ar_heavy',   self.ar_heavy  [-1])
        print('ar_partial', self.ar_partial[-1])

    def print_latex_table(self):
        string_format = "{:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n \\hline"
        try:
            print(string_format.format(
                self.ap[-1], self.ap50[-1], self.ap75[-1], self.ar100[-1], 
                self.ar_none[-1], self.ar_partial[-1], self.ar_heavy[-1]))
        except:
            print("Data not available yet")