import sys 
import os
import re
import json
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QRadioButton, QApplication,
                             QLineEdit, QWidget, QPushButton, QLabel, QGroupBox,
                             QScrollBar, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QComboBox, QButtonGroup, QSizePolicy)

from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot

from datetime import datetime
import numpy as np
from skimage import io
import h5py
import matplotlib.pylab as plt
 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure




#from temp import tomo_test

global tomo

class App(QWidget):
    
    def __init__(self):
        super().__init__()
        self.title =  'Tomo Control'
        self.left = 400
        self.top = 400
        self.width = 1000
        self.height = 600
        self.initUI()
        self.img_bkg = np.array([])
        self.img_tomo = np.array([])
        self.img_align = np.array([])
        self.img_recon = np.array([])
 
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.fpath = os.getcwd()

        
#        lb_empty = QLabel()
        grid = QGridLayout()
        gpbox_prep = self.GP_prepare()
        gpbox_recon = self.recon()
        
        grid.addWidget(gpbox_prep,0,1)
        grid.addWidget(gpbox_recon,1,1)

        
        layout = QVBoxLayout()

        layout.addLayout(grid)
        layout.addWidget(QLabel())

        self.setLayout(layout)
#        self.resize(640,200)
        
    def GP_prepare(self):        
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(30)
        
        gpbox = QGroupBox('Load image')
        gpbox.setFont(self.font1)

        lb_ld = QLabel()
        lb_ld.setFont(self.font2)
        lb_ld.setText('Image type:')
        lb_ld.setFixedWidth(100)
        
        self.pb_ld = QPushButton('Load image stack')
        self.pb_ld.setToolTip('image type: .hdf, .tiff')
        self.pb_ld.setFont(self.font2)
        self.pb_ld.clicked.connect(self.load_image)
        self.pb_ld.setFixedWidth(150)
        
        lb_mod= QLabel()
        lb_mod.setFont(self.font2)
        lb_mod.setText('Image mode:')
        lb_mod.setFixedWidth(100)
        
        file_group = QButtonGroup()
        file_group.setExclusive(True)
        self.rd_hdf = QRadioButton('hdf')
        self.rd_hdf.setChecked(True)
        self.rd_tif = QRadioButton('tif')
        file_group.addButton(self.rd_hdf)
        file_group.addButton(self.rd_tif)
    
        type_group = QButtonGroup()
        type_group.setExclusive(True)
        self.rd_absp = QRadioButton('Absorption')
        self.rd_absp.setFont(self.font2)
        self.rd_absp.setChecked(True)
        self.rd_flrc = QRadioButton('Fluorescence')
        self.rd_flrc.setFont(self.font2)
        type_group.addButton(self.rd_absp)
        type_group.addButton(self.rd_flrc)
        
        
        lb_fp = QLabel()
        lb_fp.setFont(self.font2)
        lb_fp.setText('Image loaded: ')
        lb_fp.setFixedWidth(100)
        
        self.lb_ip = QLabel()
        self.lb_ip.setFont(self.font2)
	       
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_ld)
        hbox1.addWidget(self.rd_hdf)
        hbox1.addWidget(self.rd_tif)
        hbox1.addWidget(lb_empty2)
        hbox1.addWidget(self.pb_ld)
        hbox1.addWidget(lb_empty2)
        hbox1.addWidget(self.lb_ip)
        hbox1.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_mod)
        hbox2.addWidget(self.rd_absp)
        hbox2.addWidget(self.rd_flrc)
        hbox2.addWidget(lb_empty)
        hbox2.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)                
        vbox.setAlignment(QtCore.Qt.AlignLeft)
        
        gpbox.setLayout(vbox)
    
        return gpbox

 
    def recon(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(30)
        
        gpbox = QGroupBox('Tomo reconstruction')
        gpbox.setFont(self.font1)
        
        self.canvas1 = MyCanvas()
        self.canvas2 = MyCanvas()
        
        pb_rmbg = QPushButton('Remove Bkg')
        pb_rmbg.setFont(self.font2)
        pb_rmbg.clicked.connect(self.rm_bkg)
        pb_rmbg.setFixedWidth(150)
        
        pb_align = QPushButton('Align image')
        pb_align.setFont(self.font2)
        pb_align.clicked.connect(self.align_img)
        pb_align.setFixedWidth(150)        
        
        
        self.sl1 = QScrollBar(QtCore.Qt.Horizontal)
        self.sl1.setMaximum(0)
        self.sl1.setMinimum(0)
#        self.sl1.hide()
        self.sl1.valueChanged.connect(lambda: self.sliderval('left'))
        
        self.sl2 = QScrollBar(QtCore.Qt.Horizontal)
        self.sl2.setMaximum(0)
        self.sl2.setMinimum(0)
#        self.sl2.hide()
        self.sl2.valueChanged.connect(lambda: self.sliderval('right'))
        
        self.cb1 = QComboBox()
        self.cb1.setFont(self.font2)
        self.cb1.addItem('Raw image')
        self.cb1.addItem('Background')
        self.cb1.addItem('Aligned')
        self.cb1.addItem('Reconstructed')
        
        self.cb2 = QComboBox()
        self.cb2.setFont(self.font2)
        self.cb2.addItem('Raw image')
        self.cb2.addItem('Background')
        self.cb2.addItem('Aligned')
        self.cb2.addItem('Reconstructed')
        
        pb_sh1 = QPushButton('Update')
        pb_sh1.setToolTip('update left image')
        pb_sh1.setFont(self.font2)
        pb_sh1.clicked.connect(lambda: self.update_canvas_img('left'))
        pb_sh1.setFixedWidth(150)
        
        pb_sh2 = QPushButton('Update')
        pb_sh2.setToolTip('update right image')
        pb_sh2.setFont(self.font2)
        pb_sh2.clicked.connect(lambda: self.update_canvas_img('right'))
        pb_sh2.setFixedWidth(150)        
        
        hbox = QHBoxLayout()
        
        vbox_pb = QVBoxLayout()
        vbox_pb.addWidget(pb_rmbg)
        vbox_pb.addWidget(pb_align)
        vbox_pb.addWidget(lb_empty)
        vbox_pb.setAlignment(QtCore.Qt.AlignTop)
        
        hbox_can_l = QHBoxLayout()
        hbox_can_l.addWidget(self.cb1)
        hbox_can_l.addWidget(pb_sh1)
 
        hbox_can_r = QHBoxLayout()
        hbox_can_r.addWidget(self.cb2)
        hbox_can_r.addWidget(pb_sh2)
        
        vbox_can1 = QVBoxLayout()
        vbox_can1.addWidget(self.canvas1)
        vbox_can1.addWidget(self.sl1)
        vbox_can1.addLayout(hbox_can_l)
        vbox_can1.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox_can2 = QVBoxLayout()
        vbox_can2.addWidget(self.canvas2)
        vbox_can2.addWidget(self.sl2)
        vbox_can2.addLayout(hbox_can_r)
        vbox_can2.setAlignment(QtCore.Qt.AlignLeft)        
        
        hbox.addLayout(vbox_pb)
        hbox.addLayout(vbox_can1)
        hbox.addWidget(lb_empty2)
        hbox.addLayout(vbox_can2)
        hbox.setAlignment(QtCore.Qt.AlignLeft)
        
        gpbox.setLayout(hbox)
        return gpbox
    
 
    def load_image(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        if self.rd_hdf.isChecked() == True:     file_type = 'hdf files (*.h5)'
        else: file_type = 'tiff files (*.tif)'
        
        fn, _ = QFileDialog.getOpenFileName(tomo, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        self.img_bkg = np.array([])
        self.img_tomo = np.array([])
        self.img_align = np.array([])
        self.img_recon = np.array([])
        if fn:            
            print(fn)
            fn_relative = fn.split('/')[-1]
            self.lb_ip.setText('{} loaded'.format(fn_relative))
            if self.rd_hdf.isChecked() == True:  # read hdf file
                f = h5py.File(fn, 'r')
                try:
                    self.img_tomo = np.array(f['img_tomo'])
                except:
                    self.img_tomo = np.zeros([1, 100, 100])
                    print('bkg image not exist')
                try:
                    self.img_bkg = np.array(f['img_bkg_raw'])
                except:
                    self.img_bkg = np.zeros([self.img_tomo.shape])
                    print('tomo image not exist')
                    
                f.close()
            else: # read tiff file
                self.img_tomo = np.array(io.imread(fn))
                
            if self.img_tomo.shape[0] > 0: 
                print('total images: '+str(self.img_tomo.shape[0]))
#                self.sl1.show()
                
#            test_img = self.img_tomo[0]
#            self.canvas1.update_img_one(test_img)
#            self.show_image(test_img)
            
#    def show_image(self, test_img):
#        ax = self.figure.add_subplot(111)
#        ax.clear()
#        ax.imshow(test_img)
#        self.canvas.draw()
        
        
#    def show_image(self, test_img):
#        colormap = []
##        for i in range(255):
##            colormap.append(QtGui.qRgb(i,i,i))
#        w, h = test_img.shape
#    
#        pp = QtGui.QImage(test_img.data, h,w, QtGui.QImage.Format_RGB32)
##        if (max(pp.width(), pp.height()) > 600)
##        pixmap = QtGui.QPixmap(pp.scaled(pp.width()/2, pp.height()/2))
##        else:
#        pixmap = QtGui.QPixmap(pp)
#        self.lb_pic.setPixmap(pixmap)

   
    def sliderval(self, slide_type):
        
        if slide_type == 'left':        
            canvas = self.canvas1
            img_index = self.sl1.value()

        else: 
            canvas = self.canvas2
            img_index = self.sl2.value()

        img = canvas.img_stack[img_index]
        img = np.array(img)
#        self.show_image(img)
        canvas.update_img_one(img)
        
    def rm_bkg(self):
        pass
    
    def align_img(self):
        pass
    
    def update_canvas_img(self, canv):
        
        if canv == 'left': 
            canvas = self.canvas1
            slide = self.sl1
            index = self.cb1.currentIndex()
            print(str(1)+ ' ' + str(index))
        else: 
            canvas = self.canvas2
            slide = self.sl2
            index = self.cb2.currentIndex()
            print(str(2) + ' ' + str(index))
        
        if index == 0:
            canvas.img_stack = self.img_tomo
            canvas.update_img_stack()
            slide.setMaximum(max(self.img_tomo.shape[0]-1, 0))
        elif index == 1:
            canvas.img_stack = self.img_bkg
            canvas.update_img_stack()
            slide.setMaximum(max(self.img_bkg.shape[0]-1, 0))
        elif index == 2:
            canvas.img_stack = self.img_align
            canvas.update_img_stack()
            slide.setMaximum(max(self.img_align.shape[0]-1, 0))
        else:
            canvas.img_stack = self.img_recon
            canvas.update_img_stack()
            slide.setMaximum(max(self.img_recon.shape[0]-1, 0))
           

#    
        
class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, width = 5, height = 3, dpi = 100):
        fig = Figure(figsize=(width, height), dpi=dpi)      
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')
        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)        
        self.img_stack = np.zeros([1, 100, 100])
    def update_img_stack(self):
        if self.img_stack.shape[0] == 0:
            img_blank = np.zeros([100,100])
            return self.update_img_one(img_blank)
        return self.update_img_one(self.img_stack[0])
        
    def update_img_one(self, img):
        self.axes.imshow(img)
        self.draw()
    
if __name__ == '__main__':

    app = QApplication(sys.argv)
    tomo = App()
    tomo.show()    
    sys.exit(app.exec_())
