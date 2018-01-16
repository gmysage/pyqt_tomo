#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:07:42 2018

@author: mingyuan
"""

import sys 
import os
#import re
#import json
import tomopy
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QRadioButton, QApplication,
                             QLineEdit, QWidget, QPushButton, QLabel, QGroupBox,
                             QScrollBar, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QComboBox, QCheckBox, QButtonGroup, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5 import QtCore
#from PyQt5.QtCore import pyqtSlot

import numpy as np
from skimage import io
import h5py
import matplotlib.pylab as plt 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from align_class import dftregistration
from image_binning import bin_ndarray
from scipy.ndimage.interpolation import shift 

global tomo

class App(QWidget):
    
    def __init__(self):
        super().__init__()
        self.title =  'Tomo Control'
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()        
        self.width = 1000
        self.height = 650
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2
        
        self.initUI()
        self.img_bkg = np.array([])
        self.img_tomo = np.array([])
        self.img_align = np.array([])
        self.img_recon = np.array([])
        self.msg = ''
 
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.fpath = os.getcwd()
        
        grid = QGridLayout()
        gpbox_prep = self.layout_GP_prepare()
        gpbox_msg = self.layout_msg()
        gpbox_recon = self.layout_recon()
        
        grid.addWidget(gpbox_prep,0,1)
        grid.addLayout(gpbox_msg, 1,1)
        grid.addWidget(gpbox_recon,2,1)
       
        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(QLabel())

        self.setLayout(layout)
#        self.resize(640,200)
        
    def layout_msg(self):
#        lb_empty = QLabel()
#        lb_empty1 = QLabel()
#        lb_empty1.setFixedWidth(40)
        
        self.lb_ip = QLabel()
        self.lb_ip.setFont(self.font2)  
        self.lb_ip.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_ip.setText('File loaded:')
#        self.lb_ip.setFixedWidth(300)
        self.lb_msg = QLabel()
        self.lb_msg.setFont(self.font2)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_msg.setText('Note:')
        
        vbox_msg = QVBoxLayout()
        vbox_msg.addWidget(self.lb_ip)
#        vbox_msg.addWidget(lb_empty1)
        vbox_msg.addWidget(self.lb_msg)
#        vbox_msg.addWidget(lb_empty)
        vbox_msg.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_msg
        
    def layout_GP_prepare(self):        
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(30)
        lb_empty3 = QLabel()
        lb_empty3.setFixedWidth(100)
        
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
        
        lb_ang = QLabel()
        lb_ang.setFont(self.font2)
        lb_ang.setText('Tomo angle:')
        lb_ang.setFixedWidth(100)
        
        self.lb_ang1 = QLabel()
        self.lb_ang1.setFont(self.font2)
        self.lb_ang1.setText('No angle data ...')
        self.lb_ang1.setFixedWidth(300)
        
        self.lb_ang2 = QLabel()
        self.lb_ang2.setFont(self.font2)
        self.lb_ang2.setText('Manual input (python command):')
        self.lb_ang2.setFixedWidth(250)
        self.lb_ang2.setVisible(False)
        
        self.tx_ang = QLineEdit()
        self.tx_ang.setFixedWidth(300)
        self.tx_ang.setFont(self.font2)
        self.tx_ang.setVisible(False)
        
        self.pb_ang = QPushButton('Execute')
        self.pb_ang.setFont(self.font2)
        self.pb_ang.clicked.connect(self.manu_angle_input)
        self.pb_ang.setFixedWidth(150)
        self.pb_ang.setVisible(False)        
        
        lb_mod= QLabel()
        lb_mod.setFont(self.font2)
        lb_mod.setText('Image mode:')
        lb_mod.setFixedWidth(100)
        
        self.file_group = QButtonGroup()
        self.file_group.setExclusive(True)
        self.rd_hdf = QRadioButton('hdf')
        self.rd_hdf.setFixedWidth(60)
        self.rd_hdf.setChecked(True)
        self.rd_hdf.toggled.connect(self.select_hdf_file)
        
        self.rd_tif = QRadioButton('tif')
        self.rd_tif.setFixedWidth(60)
        self.rd_tif.toggled.connect(self.select_tif_file)
        self.file_group.addButton(self.rd_hdf)
        self.file_group.addButton(self.rd_tif)
        
        
        lb_hdf_tomo = QLabel()
        lb_hdf_tomo.setFont(self.font2)
        lb_hdf_tomo.setText('Dataset for tomo scan:')
        lb_hdf_tomo.setFixedWidth(150)
        
        lb_hdf_bkg = QLabel()
        lb_hdf_bkg.setFont(self.font2)
        lb_hdf_bkg.setText('bkg:')
        lb_hdf_bkg.setFixedWidth(40)
        
        lb_hdf_dark = QLabel()
        lb_hdf_dark.setFont(self.font2)
        lb_hdf_dark.setText('dark:')
        lb_hdf_dark.setFixedWidth(40)
        
        self.tx_hdf_tomo = QLineEdit()
        self.tx_hdf_tomo.setText('img_tomo')
        self.tx_hdf_tomo.setFixedWidth(80)
        self.tx_hdf_tomo.setFont(self.font2)
        self.tx_hdf_tomo.setVisible(True)
        
        self.tx_hdf_bkg = QLineEdit()
        self.tx_hdf_bkg.setText('img_bkg')
        self.tx_hdf_bkg.setFixedWidth(80)
        self.tx_hdf_bkg.setFont(self.font2)
        self.tx_hdf_bkg.setVisible(True)
        
        self.tx_hdf_dark = QLineEdit()
        self.tx_hdf_dark.setText('img_dark')
        self.tx_hdf_dark.setFixedWidth(80)
        self.tx_hdf_dark.setFont(self.font2)
        self.tx_hdf_dark.setVisible(True)
    
        self.type_group = QButtonGroup()
        self.type_group.setExclusive(True)
        self.rd_absp = QRadioButton('Absorption')
        self.rd_absp.setFont(self.font2)
        self.rd_absp.setFixedWidth(100)
        self.rd_absp.setChecked(True)
        self.rd_flrc = QRadioButton('Fluorescence')
        self.rd_flrc.setFont(self.font2)
        self.rd_flrc.setFixedWidth(100)
        self.rd_flrc.setChecked(False)
        self.type_group.addButton(self.rd_absp)
        self.type_group.addButton(self.rd_flrc)
        
        lb_fp = QLabel()
        lb_fp.setFont(self.font2)
        lb_fp.setText('Image loaded: ')
        lb_fp.setFixedWidth(100)
        
#        self.lb_ip = QLabel()
#        self.lb_ip.setFont(self.font2)  
##        self.lb_msg = QLabel()
##        self.lb_msg.setFont(self.font2)
##        self.lb_msg.setText('')
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_ld)
        hbox1.addWidget(self.rd_tif)
        hbox1.addWidget(self.rd_hdf)
        hbox1.addWidget(lb_hdf_tomo)
        hbox1.addWidget(self.tx_hdf_tomo)
        hbox1.addWidget(lb_hdf_bkg)
        hbox1.addWidget(self.tx_hdf_bkg)
        hbox1.addWidget(lb_hdf_dark)
        hbox1.addWidget(self.tx_hdf_dark)
#        hbox1.addWidget(lb_empty2)
#        hbox1.addWidget(self.chbx_bin)
        hbox1.addWidget(self.pb_ld)
        hbox1.addWidget(lb_empty2)
        
#        hbox_msg = QHBoxLayout()
#        hbox_msg.addWidget(self.lb_ip)
#        hbox_msg.addWidget(lb_empty)
##        hbox_msg.addWidget(self.lb_msg)
#        hbox_msg.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_mod)
        hbox2.addWidget(self.rd_absp)
        hbox2.addWidget(self.rd_flrc)
        hbox2.addWidget(lb_empty3)        
#        hbox2.addLayout(hbox_msg)
#        hbox2.addWidget(self.lb_ip)
#        hbox2.addWidget(lb_empty3)
#        hbox2.addWidget(self.lb_msg)
        hbox2.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_ang = QHBoxLayout()
        hbox_ang.addWidget(lb_ang)
        hbox_ang.addWidget(self.lb_ang1)        
        hbox_ang.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_manul_input = QHBoxLayout()
        hbox_manul_input.addWidget(self.lb_ang2)
        hbox_manul_input.addWidget(self.tx_ang)
        hbox_manul_input.addWidget(self.pb_ang) 
        hbox_manul_input.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
#        vbox.addLayout(hbox_msg)
        vbox.addLayout(hbox2)    
        vbox.addLayout(hbox_ang)  
        vbox.addLayout(hbox_manul_input)     
        vbox.setAlignment(QtCore.Qt.AlignLeft)
        
        gpbox.setLayout(vbox)
    
        return gpbox

 
    def layout_recon(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(5)
        
        gpbox = QGroupBox('Tomo reconstruction')
        gpbox.setFont(self.font1)
 
        tomo_alg_layout = self.layout_tomo_algorithm()
        tomo_rec_layout = self.layout_tomo_recon()
        tomo_prep_layout = self.layout_tomo_prep()
        tomo_rot_cent_layout = self.layout_rot_center()
        canvas_layout = self.layout_canvas()
               
        vbox_tomo_recon = QVBoxLayout()
        vbox_tomo_recon.addLayout(tomo_prep_layout)
        vbox_tomo_recon.addLayout(tomo_rot_cent_layout)
        vbox_tomo_recon.addWidget(lb_empty)
        vbox_tomo_recon.addLayout(tomo_alg_layout)
        vbox_tomo_recon.addWidget(lb_empty)
        vbox_tomo_recon.addLayout(tomo_rec_layout)
        vbox_tomo_recon.setAlignment(QtCore.Qt.AlignTop)
           
        hbox = QHBoxLayout()
        hbox.addLayout(vbox_tomo_recon)
        hbox.addLayout(canvas_layout)
        hbox.addWidget(lb_empty2)  
        hbox.setAlignment(QtCore.Qt.AlignLeft)        
        gpbox.setLayout(hbox)
        return gpbox
    
    def layout_tomo_prep(self):
        lb_empty = QLabel()
        lb_prep = QLabel()
        lb_prep.setFont(self.font1)
        lb_prep.setText('Preparation')
        lb_prep.setFixedWidth(150)
        
        self.pb_rmbg = QPushButton('Remove Bkg')
        self.pb_rmbg.setFont(self.font2)
        self.pb_rmbg.clicked.connect(self.rm_bkg)
        self.pb_rmbg.setEnabled(False)
        self.pb_rmbg.setFixedWidth(150)
        
        self.pb_align = QPushButton('Align Img')
        self.pb_align.setFont(self.font2)
        self.pb_align.clicked.connect(self.tomo_align_img)
        self.pb_align.setEnabled(False) 
        self.pb_align.setFixedWidth(150)   
        
        self.pb_norm1 = QPushButton('Norm. Bkg (opt.)')
        self.pb_norm1.setFont(self.font2)
        self.pb_norm1.clicked.connect(self.tomo_norm_bkg)
        self.pb_norm1.setEnabled(False) 
        self.pb_norm1.setFixedWidth(150)   
        
        self.pb_norm2 = QPushButton('Norm. Inten. (opt.)')
        self.pb_norm2.setFont(self.font2)
        self.pb_norm2.clicked.connect(self.tomo_norm_intensity)
        self.pb_norm2.setEnabled(False) 
        self.pb_norm2.setFixedWidth(150)
        
        
        self.lb_ali = QLabel()
        self.lb_ali.setFont(self.font2)
        self.lb_ali.setText('Aligning slice: ')
        self.lb_ali.setFixedWidth(150)
        
        self.lb_rsft= QLabel()
        self.lb_rsft.setFont(self.font2)
        self.lb_rsft.setText('Row shift: ')
        self.lb_rsft.setFixedWidth(150)
        
        self.lb_csft = QLabel()
        self.lb_csft.setFont(self.font2)
        self.lb_csft.setText('Col shift: ')
        self.lb_csft.setFixedWidth(150)       
        
        hbox_prep = QHBoxLayout()
        hbox_prep.addWidget(self.pb_rmbg)
        hbox_prep.addWidget(self.pb_norm1)        
        hbox_prep.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_norm = QHBoxLayout()
        hbox_norm.addWidget(self.pb_norm2)
        hbox_norm.addWidget(self.pb_align)
        hbox_norm.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_img_shift = QHBoxLayout()
        hbox_img_shift.addWidget(self.lb_rsft)
        hbox_img_shift.addWidget(self.lb_csft)
        hbox_img_shift.setAlignment(QtCore.Qt.AlignLeft)
                        
        vbox_prep = QVBoxLayout()
        vbox_prep.addWidget(lb_prep)
        vbox_prep.addLayout(hbox_prep)
        vbox_prep.addLayout(hbox_norm)
        vbox_prep.addWidget(self.lb_ali)
        vbox_prep.addLayout(hbox_img_shift)
        vbox_prep.addWidget(lb_empty)
        
        return vbox_prep
    
    def layout_rot_center(self):
        lb_rot_title = QLabel()
        lb_rot_title.setFont(self.font1)
        lb_rot_title.setText('Find rotation center ')
        
        lb_note = QLabel()
        lb_note.setFont(self.font2)
        lb_note.setStyleSheet('color: rgb(200, 50, 50);')
        lb_note.setText('(use coordinates showing in display)')
        
        lb_rot_guess = QLabel()
        lb_rot_guess.setFont(self.font2)
        lb_rot_guess.setText('Init. guess: ')
        lb_rot_guess.setFixedWidth(80)
        
        self.tx_rot_guess = QLineEdit(self)
        self.tx_rot_guess.setFont(self.font2)
        self.tx_rot_guess.setValidator(QDoubleValidator())
        self.tx_rot_guess.setFixedWidth(60)
        
        self.pb_rot = QPushButton('Find')
        self.pb_rot.setToolTip('rotation center (coarse)')
        self.pb_rot.setFont(self.font2)
        self.pb_rot.clicked.connect(self.find_rotation_center)
        self.pb_rot.setEnabled(False)
        self.pb_rot.setFixedWidth(80)
        
        lb_rc_test = QLabel()
        lb_rc_test.setFont(self.font2)
        lb_rc_test.setText('Test rotation center: ')
        lb_rc_test.setFixedWidth(150)
        
        lb_rc_test_s = QLabel()
        lb_rc_test_s.setFont(self.font2)
        lb_rc_test_s.setText('R.C start: ')
        lb_rc_test_s.setFixedWidth(80)
        
        self.tx_rc_test_s = QLineEdit(self)
        self.tx_rc_test_s.setFont(self.font2)
        self.tx_rc_test_s.setValidator(QDoubleValidator())
        self.tx_rc_test_s.setFixedWidth(60)
        
        lb_rc_test_e = QLabel()
        lb_rc_test_e.setFont(self.font2)
        lb_rc_test_e.setText('R.C. end: ')
        lb_rc_test_e.setFixedWidth(80)
        
        self.tx_rc_test_e = QLineEdit(self)
        self.tx_rc_test_e.setFont(self.font2)
        self.tx_rc_test_e.setValidator(QDoubleValidator())
        self.tx_rc_test_e.setFixedWidth(60)
        
        lb_which_sli = QLabel()
        lb_which_sli.setFont(self.font2)
        lb_which_sli.setText('Which slice: ')
        lb_which_sli.setFixedWidth(80)
        
        self.tx_which_sli = QLineEdit(self)
        self.tx_which_sli.setFont(self.font2)
        self.tx_rc_test_s.setValidator(QIntValidator())
        self.tx_which_sli.setFixedWidth(60)
        
        self.pb_rot_test = QPushButton('Test')
        self.pb_rot_test.setToolTip('test rotation center at 1 pix interval')
        self.pb_rot_test.setFont(self.font2)
        self.pb_rot_test.clicked.connect(self.test_rotation_center)
        self.pb_rot_test.setEnabled(False)
        self.pb_rot_test.setFixedWidth(80)
        
        self.lb_rot_user = QLabel()
        self.lb_rot_user.setFont(self.font2)
        self.lb_rot_user.setText('Input R.C.: ')
        self.lb_rot_user.setFixedWidth(80)
        
        self.tx_rot = QLineEdit(self)
        self.tx_rot.setFont(self.font2)
        self.tx_rot.setValidator(QDoubleValidator())
        self.tx_rot.setFixedWidth(60)
        
        self.chbx_rc_l = QCheckBox('draw R.C')
        self.chbx_rc_l.setFont(self.font2)
        self.chbx_rc_l.setFixedWidth(100)
        self.chbx_rc_l.setChecked(False)
        self.chbx_rc_l.stateChanged.connect(self.draw_RC)
        self.chbx_rc_l.setEnabled(False) 
        
        hbox_rc_test1 = QHBoxLayout() 
        hbox_rc_test1.addWidget(lb_rc_test_s)
        hbox_rc_test1.addWidget(self.tx_rc_test_s)
        hbox_rc_test1.addWidget(lb_rc_test_e)
        hbox_rc_test1.addWidget(self.tx_rc_test_e)
        hbox_rc_test1.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_rc_test2 = QHBoxLayout()
        hbox_rc_test2.addWidget(lb_which_sli)
        hbox_rc_test2.addWidget(self.tx_which_sli)
        hbox_rc_test2.addWidget(self.pb_rot_test)
        hbox_rc_test2.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_rot = QHBoxLayout()  # user input rotation-center
        hbox_rot.addWidget(self.lb_rot_user)
        hbox_rot.addWidget(self.tx_rot)
        hbox_rot.addWidget(self.chbx_rc_l)
        hbox_rot.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_rot_guess = QHBoxLayout() # guess the rotation-center
        hbox_rot_guess.addWidget(lb_rot_guess)
        hbox_rot_guess.addWidget(self.tx_rot_guess)
        hbox_rot_guess.addWidget(self.pb_rot)
        hbox_rot_guess.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox_rot_cent = QVBoxLayout()
        vbox_rot_cent.addWidget(lb_rot_title)
        vbox_rot_cent.addWidget(lb_note)
        vbox_rot_cent.addLayout(hbox_rot_guess)
        vbox_rot_cent.addWidget(lb_rc_test)
        vbox_rot_cent.addLayout(hbox_rc_test1)
        vbox_rot_cent.addLayout(hbox_rc_test2)
        vbox_rot_cent.addLayout(hbox_rot)
        
        return vbox_rot_cent
    
    def layout_tomo_algorithm(self):
        lb_empty = QLabel()
        lb_empty.setFixedWidth(5)
            
        lb_note = QLabel()
        lb_note.setFont(self.font2)
        lb_note.setStyleSheet('color: rgb(200, 50, 50);')
        lb_note.setText('(use coordinates showing in display)')
        
        lb_alg = QLabel()
        lb_alg.setFont(self.font1)
        lb_alg.setText('Algorithm')
        lb_alg.setFixedWidth(100)
        
        lb_chose = QLabel()
        lb_chose.setFont(self.font2)
        lb_chose.setText('select Alg. ')
        lb_chose.setFixedWidth(80)        
        
        self.cb_alg = QComboBox()
        self.cb_alg.setFont(self.font2)
        self.cb_alg.addItem('gridrec')
        self.cb_alg.addItem('mlem')
        self.cb_alg.addItem('hybrid')
        self.cb_alg.currentIndexChanged.connect(self.select_algorithm)
        self.cb_alg.setFixedWidth(80)
        
        self.lb_iter = QLabel()
        self.lb_iter.setFont(self.font2)
        self.lb_iter.setText('iter.')
        self.lb_iter.setFixedWidth(30)
                
        self.tx_iter = QLineEdit(self)
        self.tx_iter.setFont(self.font2)
        self.tx_iter.setFixedWidth(60)
        self.tx_iter.setEnabled(False)
        
        hbox_alg = QHBoxLayout()
        hbox_alg.addWidget(lb_chose)
        hbox_alg.addWidget(self.cb_alg)
#        hbox_alg.addWidget(lb_empty)
        hbox_alg.addWidget(self.lb_iter)
        hbox_alg.addWidget(self.tx_iter)
        hbox_alg.setAlignment(QtCore.Qt.AlignLeft)
        
        lb_sli = QLabel()
        lb_sli.setFont(self.font2)
        lb_sli.setText('reconstruct: ')
        lb_sli.setFixedWidth(80)
        self.rd_sli_1 = QRadioButton('multi. slice')
        self.rd_sli_1.setChecked(True)
        self.rd_sli_1.setFont(self.font2)
        self.rd_sli_1.setFixedWidth(100)
        
        self.rd_sli_2 = QRadioButton('all')
        self.rd_sli_2.setFont(self.font2)
        self.rd_sli_2.toggled.connect(self.select_recon_slice)
        self.rd_sli_2.setFixedWidth(100)
        self.sli_select_group = QButtonGroup()
        self.sli_select_group.setExclusive(True)
        self.sli_select_group.addButton(self.rd_sli_1)
        self.sli_select_group.addButton(self.rd_sli_2)
        
        lb_sli_st = QLabel()
        lb_sli_st.setFont(self.font2)
        lb_sli_st.setText('slice start: ')
        lb_sli_st.setFixedWidth(80)
        
        lb_sli_end = QLabel()
        lb_sli_end.setFont(self.font2)
        lb_sli_end.setText('end: ')
        lb_sli_end.setFixedWidth(40)
        
        self.tx_sli_st = QLineEdit(self)
        self.tx_sli_st.setFont(self.font2)
        self.tx_sli_st.setValidator(QIntValidator())
        self.tx_sli_st.setFixedWidth(60)
        
        self.tx_sli_end = QLineEdit(self)
        self.tx_sli_end.setFont(self.font2)
        self.tx_sli_end.setValidator(QIntValidator())
        self.tx_sli_end.setFixedWidth(60)
        
        hbox_sli_select_1 = QHBoxLayout()
        hbox_sli_select_1.addWidget(lb_sli)
        hbox_sli_select_1.addWidget(self.rd_sli_1)
        hbox_sli_select_1.addWidget(self.rd_sli_2)
        hbox_sli_select_1.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_sli_select_2 = QHBoxLayout()
        hbox_sli_select_2.addWidget(lb_sli_st)
        hbox_sli_select_2.addWidget(self.tx_sli_st)
        hbox_sli_select_2.addWidget(lb_empty)
        hbox_sli_select_2.addWidget(lb_sli_end)
        hbox_sli_select_2.addWidget(self.tx_sli_end)
        hbox_sli_select_2.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox_sli = QVBoxLayout()
        vbox_sli.addLayout(hbox_sli_select_1)
        vbox_sli.addLayout(hbox_sli_select_2)
        vbox_sli.setAlignment(QtCore.Qt.AlignLeft)        
        
        vbox_alg_tot = QVBoxLayout()
        vbox_alg_tot.addWidget(lb_alg)
        vbox_alg_tot.addWidget(lb_note)
        vbox_alg_tot.addLayout(hbox_alg)
        vbox_alg_tot.addLayout(vbox_sli)
        vbox_alg_tot.setAlignment(QtCore.Qt.AlignLeft)
        
        return vbox_alg_tot
        
        
    def layout_tomo_recon(self):        
        lb_rec = QLabel()
        lb_rec.setFont(self.font1)
        lb_rec.setText('Reconstruction')
        lb_rec.setFixedWidth(120)
        
        self.pb_recon = QPushButton('Run and save')
        self.pb_recon.setFont(self.font2)
        self.pb_recon.clicked.connect(self.tomo_recon)
        self.pb_recon.setEnabled(False)
        self.pb_recon.setFixedWidth(160)
        
        self.cb_bin = QComboBox()
        self.cb_bin.setFont(self.font2)
        self.cb_bin.addItem('raw image')
        self.cb_bin.addItem('2x2 binned image')
        self.cb_bin.addItem('4x4 binned image')
        self.cb_bin.setCurrentText('2x2 binned image')
        self.cb_bin.setFixedWidth(160)
        
        
        self.rd_rec_bin = QRadioButton('binned image')
        self.rd_rec_bin.setChecked(True)
        self.rd_rec_bin.setFont(self.font2)
        self.rd_rec_bin.setFixedWidth(120)
        
        self.rd_rec_raw = QRadioButton('raw image')
        self.rd_rec_raw.setFont(self.font2)
        self.rd_rec_raw.setFixedWidth(120)
        
        self.rec_bin_img_group = QButtonGroup()
        self.rec_bin_img_group.setExclusive(True)
        self.rec_bin_img_group.addButton(self.rd_rec_bin)
        self.rec_bin_img_group.addButton(self.rd_rec_raw)
        
        hbox_rec = QHBoxLayout()
        hbox_rec.addWidget(self.cb_bin)
        hbox_rec.addWidget(self.pb_recon)
#        hbox_rec.addWidget(self.rd_rec_bin)
#        hbox_rec.addWidget(self.rd_rec_raw)
        hbox_rec.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox_rec = QVBoxLayout()
        vbox_rec.addWidget(lb_rec)
        vbox_rec.addLayout(hbox_rec)
#        vbox_rec.addWidget(self.pb_recon)
        vbox_rec.setAlignment(QtCore.Qt.AlignLeft)
        
        return vbox_rec
    
    def layout_canvas(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)
        
        self.canvas1 = MyCanvas(obj = self)       
        
        self.sl1 = QScrollBar(QtCore.Qt.Horizontal)
        self.sl1.setMaximum(0)
        self.sl1.setMinimum(0)
        self.sl1.valueChanged.connect(self.sliderval)
        
        self.cb1 = QComboBox()
        self.cb1.setFont(self.font2)
        self.cb1.addItem('Raw image')
        self.cb1.addItem('Background')
        self.cb1.currentIndexChanged.connect(self.update_canvas_img)
 
        
#        self.pb_sh1 = QPushButton('Update')
#        self.pb_sh1.setToolTip('update left image')
#        self.pb_sh1.setFont(self.font2)
#        self.pb_sh1.clicked.connect(self.update_canvas_img)
#        self.pb_sh1.setEnabled(False)
#        self.pb_sh1.setFixedWidth(150)
        
        self.pb_del = QPushButton('Del. single image')
        self.pb_del.setToolTip('Delete single image, it will delete the same slice in other images (e.g., "raw image", "aligned image", "background removed" ')
        self.pb_del.setFont(self.font2)
        self.pb_del.clicked.connect(self.delete_single_img)
        self.pb_del.setEnabled(False)
        self.pb_del.setFixedWidth(150)

        hbox_can_l = QHBoxLayout()
        hbox_can_l.addWidget(self.cb1)
#        hbox_can_l.addWidget(self.pb_sh1)
        hbox_can_l.addWidget(self.pb_del)
        
        self.lb_x_l = QLabel()
        self.lb_x_l.setFont(self.font2)
        self.lb_x_l.setText('x: ')
        self.lb_x_l.setFixedWidth(80)
        
        self.lb_y_l = QLabel()
        self.lb_y_l.setFont(self.font2)
        self.lb_y_l.setText('y: ')
        self.lb_y_l.setFixedWidth(80)
        
        self.lb_z_l = QLabel()
        self.lb_z_l.setFont(self.font2)
        self.lb_z_l.setText('intensity: ')
        self.lb_z_l.setFixedWidth(120)
        
        lb_cmap = QLabel()
        lb_cmap.setFont(self.font2)
        lb_cmap.setText('colormap: ')
        lb_cmap.setFixedWidth(80)
        
        cmap = ['gray', 'bone', 'viridis', 'terrain', 'gnuplot','bwr','plasma', 'PuBu', 'summer', 'rainbow', 'jet']
        self.cb_cmap = QComboBox()
        self.cb_cmap.setFont(self.font2)
        for i in cmap:
            self.cb_cmap.addItem(i)
        self.cb_cmap.setCurrentText('bone')
        self.cb_cmap.currentIndexChanged.connect(self.change_colormap)
        self.cb_cmap.setFixedWidth(80)

        self.pb_adj_cmap = QPushButton('Auto Contrast')
        self.pb_adj_cmap.setFont(self.font2)
        self.pb_adj_cmap.clicked.connect(self.auto_contrast)
        self.pb_adj_cmap.setEnabled(True)
        self.pb_adj_cmap.setFixedWidth(120)
        
        lb_cmax = QLabel()
        lb_cmax.setFont(self.font2)
        lb_cmax.setText('cmax: ')
        lb_cmax.setFixedWidth(40)
        lb_cmin = QLabel()
        lb_cmin.setFont(self.font2)
        lb_cmin.setText('cmin: ')
        lb_cmin.setFixedWidth(40)
             
        self.tx_cmax = QLineEdit(self)
        self.tx_cmax.setFont(self.font2)
        self.tx_cmax.setFixedWidth(60)
        self.tx_cmax.setText('1.')
        self.tx_cmax.setValidator(QDoubleValidator())
        self.tx_cmax.setEnabled(True)
        
        self.tx_cmin = QLineEdit(self)
        self.tx_cmin.setFont(self.font2)
        self.tx_cmin.setFixedWidth(60)
        self.tx_cmin.setText('0.')
        self.tx_cmin.setValidator(QDoubleValidator())
        self.tx_cmin.setEnabled(True)
        
        self.pb_set_cmap = QPushButton('Set')
        self.pb_set_cmap.setFont(self.font2)
        self.pb_set_cmap.clicked.connect(self.set_contrast)
        self.pb_set_cmap.setEnabled(True)
        self.pb_set_cmap.setFixedWidth(60)
        
        hbox_chbx_l = QHBoxLayout()
        hbox_chbx_l.addWidget(self.lb_x_l)
        hbox_chbx_l.addWidget(self.lb_y_l)
        hbox_chbx_l.addWidget(self.lb_z_l)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_chbx_l.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox_cmap = QHBoxLayout()
        hbox_cmap.addWidget(lb_cmap)
        hbox_cmap.addWidget(self.cb_cmap)
        hbox_cmap.addWidget(self.pb_adj_cmap)
        hbox_cmap.addWidget(lb_cmin)
        hbox_cmap.addWidget(self.tx_cmin)
        hbox_cmap.addWidget(lb_cmax)
        hbox_cmap.addWidget(self.tx_cmax)
        hbox_cmap.addWidget(self.pb_set_cmap)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_cmap.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox_can1 = QVBoxLayout()
        vbox_can1.addWidget(self.canvas1)
        vbox_can1.addWidget(self.sl1)
        vbox_can1.addLayout(hbox_can_l)
        vbox_can1.addLayout(hbox_chbx_l)
        vbox_can1.addLayout(hbox_cmap)
        vbox_can1.setAlignment(QtCore.Qt.AlignLeft)
        
        return vbox_can1
    
    
    def select_hdf_file(self):
        self.tx_hdf_tomo.setEnabled(True)
        self.tx_hdf_bkg.setEnabled(True)
        self.tx_hdf_dark.setEnabled(True)
        
    def select_tif_file(self):
        self.tx_hdf_tomo.setEnabled(False)
        self.tx_hdf_bkg.setEnabled(False)
        self.tx_hdf_dark.setEnabled(False)
        
    def tomo_norm_bkg(self):
        self.pb_norm1.setText('normalizing ..')
        self.pb_norm1.setEnabled(False)
        QApplication.processEvents()
        
        prj = deepcopy(self.img_bkg_removed)
        prj[np.isnan(prj)] = 0
        prj[np.isinf(prj)] = 0
        prj[prj<0] = 0
        s = prj.shape
        prj_sort = np.sort(prj) # sort each slice
        prj_sort = prj_sort[:, :, 0:int(s[1]*s[2]*0.1)]
        slice_avg = np.mean(np.mean(prj_sort, axis=2), axis=1) # average for each slice
        prj = np.array([prj[i] - slice_avg[i] for i in range(s[0])])
        prj[prj < 0] = 0
        self.img_bkg_removed = deepcopy(prj)
        self.img_align = deepcopy(prj) # synchronize self.img_align and self.img_bkg_removed
        del prj, slice_avg
        self.pb_norm1.setEnabled(True)
        self.pb_norm1.setText('Norm. Bkg. (opt.) ')
        QApplication.processEvents()
        self.cb1.setCurrentText('Background removed')
        self.update_canvas_img()
        
        
    def tomo_norm_intensity(self): 
        self.pb_norm2.setText('Waiting ...')
        QApplication.processEvents()
        canvas = self.canvas1
        
        prj_sum = np.sum(np.sum(self.img_align, axis=2), axis=1) # summed intensity for each projection image
        x = np.arange(len(prj_sum)).reshape(prj_sum.shape)
        canvas.x, canvas.y = x, prj_sum
#        self.sl1.setMaximum(0)
        if self.cb1.findText('Intensity plot') < 0:
            self.cb1.addItem('Intensity plot')
        self.cb1.setCurrentText('Intensity plot')    
        self.pb_norm2.setText('Norm. Inten. (Opt.)')
        
    
    def delete_single_img(self):
        msg = QMessageBox()
        msg.setText('This will delete the image in \"raw data\", \"aligned image\", and \"background removed\" images')
        msg.setWindowTitle('Delete single image')
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    
        reply = msg.exec_()
        if reply == QMessageBox.Ok:
            img_type = self.cb1.currentText()
            current_slice = self.sl1.value()            
            if img_type == 'Reconstructed' or img_type == 'R.C. test':
                self.pb_del.setEnabled(False)
            elif img_type == 'Background':
                try:  self.img_bkg = np.delete(self.img_bkg, current_slice, axis=0)
                except: pass
            else:            
                try:   self.img_tomo = np.delete(self.img_tomo, current_slice, axis=0)
                except:  print('cannot delete img_tomo')
            
                try:   self.img_bkg_removed = np.delete(self.img_bkg_removed, current_slice, axis=0)
                except:  print('cannot delete img_bkg_removed')
            
                try:  self.img_align = np.delete(self.img_align, current_slice, axis=0)
                except: print('cannot delete img_align')
                
                try:  
                    self.theta = np.delete(self.theta, current_slice, axis=0)
                    tomo_angle = self.theta / np.pi * 180
                    st = '{0:3.1f}, {1:3.1f}, ..., {2:3.1f}  (totally, {3} angles)'.format(tomo_angle[0], tomo_angle[1], tomo_angle[-1],len(tomo_angle))
                    self.lb_ang1.setText(st) # update angle information showing on the label
                except:  print('cannot delete theta')            
            self.msg = 'image #{} has been deleted'.format(current_slice)
            self.update_msg()
            self.update_canvas_img()
    
    def tomo_recon(self):
        self.pb_recon.setEnabled(False)        
        QApplication.processEvents()
        sli_start = int(self.tx_sli_st.text())
        sli_end = int(self.tx_sli_end.text())
        rot_cen = float(self.tx_rot.text())
        prj = deepcopy(self.img_align)
        sh = prj.shape
        if self.cb_bin.currentText() == '2x2 binned image': 
            self.msg = 'reconstructing 2x2 binned image ...'
            self.update_msg()
            prj = bin_ndarray(prj, (sh[0], sh[1]//2, sh[2]//2), 'mean')
            sli_start = sli_start // 2
            sli_end = sli_end // 2
            rot_cen = rot_cen / 2
        elif self.cb_bin.currentText() == '4x4 binned image': 
            self.msg = 'reconstructing 2x2 binned image ...'
            self.update_msg()
            prj = bin_ndarray(prj, (sh[0], sh[1]//4, sh[2]//4), 'mean')
            sli_start = sli_start // 4
            sli_end = sli_end // 4
            rot_cen = rot_cen / 4
        tomo_prj = prj[:, sli_start:max(sli_end, sli_start+1), :]
        del prj
        
                
        print('Reconstrucing ...')
        # select algorithm
        if self.rot_cen and tomo_prj.shape[0]: # all paremeter are set            
            if self.cb_alg.currentText() == 'gridrec':
                rec = tomopy.recon(tomo_prj, self.theta, center=rot_cen, algorithm='gridrec')
            elif self.cb_alg.currentText() == 'mlem':
                num_iter = min(max(int(self.tx_iter.text()), 10), 30) # 10 < iter < 30
                rec = tomopy.recon(tomo_prj, self.theta, center=rot_cen, num_iter=num_iter, algorithm='mlem')
            else: # ospml_hybrid
                num_iter = min(max(int(self.tx_iter.text()), 10), 30)
                rec = tomopy.recon(tomo_prj, self.theta, center=rot_cen, num_iter=num_iter, algorithm='ospml_hybrid', reg_par=[0.01, 0.001])
            rec [rec < 0] = 0
            self.msg = 'reconstruction finished, now saving...'
            self.update_msg()
            # save results
            fbin = self.cb_bin.currentText()
            if fbin == 'raw image':
                fimg = '_recon_'
            elif fbin == '2x2 binned image':
                fimg = '_recon_2x2_'
            elif fbin == '4x4 binned image':
                fimg = '_recon_4x4_'

            fn = deepcopy(self.fn)
            fn_r = self.fn_relative.split('.')[:-1][0]
            pos = fn.find(fn_r)
            fname = fn[:pos] + fn_r + fimg + 'v' + str(self.save_version) + '.h5'
            try:
                with h5py.File(fname, 'w') as hf:
                    hf.create_dataset('recon', data = rec)
                    hf.create_dataset('slice_start', data = self.tx_sli_st.text())
                    hf.create_dataset('slice_end', data = self.tx_sli_end.text())
                    hf.create_dataset('recon_algorithm', data = self.cb_alg.currentText())
                self.save_version += 1
                self.msg = fname + '   has been saved!'
                self.update_msg()
            except:
                self.msg = 'file save fails ...'
                self.update_msg()
                    
                    
#            with h5py.File('')
            
        else:
            print('parameters setting are wrong ...')
            self.msg = 'parameters setting are wrong ...'
            self.update_msg()
        self.pb_recon.setEnabled(True)
        print('Reconstruction finished.')
        # showing the results 
        self.img_recon = rec
        t = self.cb1.findText('Reconstructed')
        if t < 0:
            self.cb1.addItem('Reconstructed')
            self.cb1.setCurrentText('Reconstructed')
        self.canvas1.img_stack = self.img_recon
        self.update_canvas_img()
        
    
    def select_recon_slice(self):
        t = self.img_tomo.shape[1]
        s = t // 2

        if self.rd_sli_2.isChecked():           # all slice
            self.tx_sli_st.setEnabled(False)
            self.tx_sli_end.setEnabled(False)
            self.tx_sli_st.setText(str(0))
            self.tx_sli_end.setText(str(t))            
        else:                                   # multi_slice
            self.tx_sli_st.setEnabled(True)
            self.tx_sli_end.setEnabled(True)
            self.tx_sli_st.setText(str(s))
            self.tx_sli_end.setText(str(s+1))
        QApplication.processEvents()
 
    def select_algorithm(self):
        if self.cb_alg.currentText() == 'gridrec': 
            self.tx_iter.setEnabled(False)
        else: 
            self.tx_iter.setEnabled(True)
            self.tx_iter.setText('20')

        
    def load_image(self):
        self.default_layout()
        self.pb_ld.setEnabled(False)
        
        if len(self.tx_hdf_tomo.text()): dataset_tomo = self.tx_hdf_tomo.text()
        else: dataset_tomo = 'img_tomo'
        
        if len(self.tx_hdf_bkg.text()): dataset_bkg = self.tx_hdf_bkg.text()
        else: dataset_bkg = 'img_bkg'
        
        if len(self.tx_hdf_dark.text()): dataset_dark = self.tx_hdf_dark.text()
        else: dataset_dark = 'img_dark'
        
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        if self.rd_hdf.isChecked() == True:     file_type = 'hdf files (*.h5)'
        else: file_type = 'tiff files (*.tif)'
        
        fn, _ = QFileDialog.getOpenFileName(tomo, "QFileDialog.getOpenFileName()", "", file_type, options=options)
       
        if fn:            
            print(fn)
            fn_relative = fn.split('/')[-1]
            print(fn_relative)
            self.fn = fn
            self.fn_relative = fn_relative
            self.lb_ip.setText('File loaded:   {}'.format(fn))
            self.lb_ip.setStyleSheet('color: rgb(200, 50, 50);')
            if self.rd_hdf.isChecked() == True:  # read hdf file
                f = h5py.File(fn, 'r')
                # read tomo-scan image
                try: 
#                    self.img_tomo = np.array(f['img_tomo'])
                    self.img_tomo = np.array(f[dataset_tomo])
                    self.rot_cen = self.img_tomo.shape[2]/2
                    print('Image size: '+str(self.img_tomo.shape))
                    self.tx_rot_guess.setText(str(self.rot_cen))
                    self.tx_rot.setText(str(self.rot_cen))
                    self.tx_sli_st.setText(str(self.img_tomo.shape[1]//2))
                    self.tx_sli_end.setText(str(self.img_tomo.shape[1]//2+1))
                    self.pb_rmbg.setEnabled(True) # remove background
                    self.pb_rot_test.setEnabled(True) # test rotation center
                    self.pb_del.setEnabled(True) # delete single image
                    
                    self.msg = 'image shape: {0}'.format(self.img_tomo.shape)
                    self.update_msg()
                except:
                    self.img_tomo = np.zeros([2, 100, 100])
                    print('tomo image not exist')
                finally: 
                    self.img_bkg_removed = deepcopy(self.img_tomo) 
                    self.img_align = deepcopy(self.img_bkg_removed)
                    self.update_canvas_img()
                    
                # read background image    
                try:  
#                    self.img_bkg = np.array(f['img_bkg_raw'])
                    self.img_bkg = np.array(f[dataset_bkg])
                    
                except:
                    self.img_bkg = np.ones(self.img_tomo.shape)
                    print('\nbkg image not exist')
                    
                # read dark image    
                try:  
#                    self.img_dark = np.array(f['img_dark_raw'])   
                    self.img_dark = np.array(f[dataset_dark])  
                except:
                    self.img_dark = np.zeros(self.img_tomo.shape)
                    print('\ndark image not exist')
                
                # read theta
                try:    
                    self.theta = np.array(f['theta'])
                    st = '{0:3.1f}, {1:3.1f}, ..., {2:3.1f}  (totally, {3} angles)'.format(self.theta[0], self.theta[1], self.theta[-1],len(self.theta))
                    self.lb_ang1.setText(st)
                    if max(np.abs(self.theta)) > np.pi:
                        self.theta = self.theta / 180 * np.pi
                    assert (len(self.theta) == self.img_tomo.shape[0]), 'number of angles does not match number of images'
                    
                except:  
                    self.theta=np.array([])  
                    self.lb_ang1.setText('No angle data ...')
                    self.lb_ang2.setVisible(True)
                    self.tx_ang.setVisible(True)
                    self.pb_ang.setVisible(True)
                f.close()
                
            else: # read tiff file
                try:
                    self.img_tomo = np.array(io.imread(fn))  
                    self.tx_sli_st.setText(str(self.img_tomo.shape[1]//2))
                    self.tx_sli_end.setText(str(self.img_tomo.shape[1]//2+1))
                    self.msg = 'image shape: {0}'.format(self.img_tomo.shape)
                    self.update_msg()
                    self.pb_rmbg.setEnabled(True) # remove background
                    self.pb_rot_test.setEnabled(True) # test rotation center
                    self.pb_del.setEnabled(True) # delete single image
                    QApplication.processEvents()
                except:
                    self.img_tomo = np.zeros([2, 100, 100])
                    print('image not exist')
                finally:
                    self.img_bkg = np.ones(self.img_tomo.shape)   # (tomo - dark)/(bkg - dark)
                    self.img_bkg_removed = deepcopy(self.img_tomo)
                    self.img_align = deepcopy(self.img_bkg_removed)
                    self.img_dark = np.zeros(self.img_tomo.shape)
                    self.update_canvas_img()
                    self.theta=np.array([])  
                    self.lb_ang1.setText('No angle data ...')
                    self.lb_ang2.setVisible(True)
                    self.tx_ang.setVisible(True)
                    self.pb_ang.setVisible(True)
        self.pb_ld.setEnabled(True) 
                       

    def update_msg(self):
        self.lb_msg.setText('Note: ' + self.msg)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')
        
    def manu_angle_input(self):
        com = 'self.theta = np.array(' + self.tx_ang.text() + ')'
        try: 
            exec(com)
            st = str(self.theta[0]) + ',  ' + str(self.theta[1]) + ',  ...  , ' + str(self.theta[-1])
            st = '{0}  (totally: {1} angles)'.format(st, len(self.theta))
            self.lb_ang1.setText(st)
        except: self.lb_ang1.setText('Invalid angle')
        

    def default_layout(self):
        try:
            del self.img_tomo, self.img_bkg, self.img_bkg_removed, self.img_bkg_update
            del self.img_align, self.img_recon, self.img_rc_test, self.theta
        except: pass
        self.save_version = 0
        self.theta=np.array([])  
        self.img_tomo = np.array([])
        self.img_bkg = np.array([])
        self.img_bkg_update = np.array([])
        self.img_bkg_removed = np.array([])
        self.img_align = np.array([])
        self.lb_ang1.setText('No angle data ...')
        self.lb_ang2.setVisible(False)
        self.tx_ang.setVisible(False)
        self.pb_ang.setVisible(False)
        
#        self.pb_rmbg.setEnabled(True)
#        self.pb_align.setEnabled(True) 
#        self.pb_rot.setEnabled(True)
        
#        self.pb_sh1.setEnabled(True)
        self.chbx_rc_l.setEnabled(True)       
        t = self.cb1.findText('Background removed')
        if t:  self.cb1.removeItem(t)            
        t = self.cb1.findText('Aligned image')
        if t:  self.cb1.removeItem(t)            
        t = self.cb1.findText('Reconstructed')
        if t:  self.cb1.removeItem(t)

   
    def sliderval(self): 
        canvas = self.canvas1
        img_index = self.sl1.value()
        img = canvas.img_stack[img_index]
        img = np.array(img)
        canvas.update_img_one(img, img_index=img_index)
        
        
    def rm_bkg(self):
        
        self.pb_rmbg.setText('wait ...')
#        self.pb_rmbg.setEnabled(False)
        QApplication.processEvents()
        if len(self.img_bkg.shape) == 3:
            img_bkg_update = np.sum(self.img_bkg, axis=0)/self.img_bkg.shape[0]
            img_dark_update = np.sum(self.img_dark, axis=0)/self.img_dark.shape[0]
        else:
            img_bkg_update = deepcopy(self.img_bkg)
            img_dark_update = deepcopy(self.img_dark)
            
        n = self.img_tomo.shape[0]
        img_bkg_ext = np.repeat(img_bkg_update[np.newaxis,:,:], n, axis=0)
        img_dark_ext = np.repeat(img_dark_update[np.newaxis,:,:], n, axis=0)

        temp = (self.img_tomo - img_dark_ext)/(img_bkg_ext - img_dark_ext)
        if self.rd_absp.isChecked() == True:       # Absorption image. e.g. TXM             
            temp[temp<0]=1 # 1 means no absorption
            temp[np.isnan(temp)] = 1
            temp[np.isinf(temp)] = 1
            temp = -np.log(temp)           
        else:                                       # Fluorescence image
            temp[temp<0] = 0
            
        temp[np.isnan(temp)] = 0
        temp[np.isinf(temp)] = 0  
        self.img_bkg_removed = temp
        self.img_align = deepcopy(self.img_bkg_removed) # synchronize img_align and img_bkg_removed
        self.pb_rmbg.setText('Remove Bkg')
        self.pb_align.setEnabled(True) # align image
        self.pb_norm1.setEnabled(True) # norm background
        self.pb_norm2.setEnabled(True) # norm total intensity
        self.pb_recon.setEnabled(True) # run reconstruction
        self.pb_rot.setEnabled(True) # find rotation center
        del temp, img_bkg_update, img_dark_update, img_bkg_ext, img_dark_ext
        QApplication.processEvents()
       
        if self.cb1.findText('Background removed') < 0:
            self.cb1.addItem('Background removed')
        print('Background removed\n. Item "Background removed" has been added. ')
        self.msg = 'Background removed.'
        self.update_msg()
        
        self.cb1.setCurrentText('Background removed')
        self.update_canvas_img()


    def align_img(self, img_ref, img):
        img1_fft = np.fft.fft2(img_ref)
        img2_fft = np.fft.fft2(img)
        output = dftregistration(img1_fft, img2_fft)
        row_shift = output[2]
        col_shift = output[3]
        img_shift = shift(img, [row_shift, col_shift], mode='constant', cval=0.0)
        return img_shift, row_shift, col_shift
    

    def tomo_align_img(self):
        self.pb_align.setText('Aligning ...')
        QApplication.processEvents()
        self.pb_align.setEnabled(False)
        prj = deepcopy(self.img_bkg_removed)
        n = self.img_align.shape[0]
        for i in range(1, n):
            print('Aligning image slice ' + str(i))            
            img_temp, rsft, csft = self.align_img(prj[i-1], prj[i])
            prj[i]= deepcopy(img_temp)
            
            self.lb_ali.setText('Aligning slice: ' + str(i))
            QApplication.processEvents()
            self.lb_rsft.setText('Row shfit: {:3.1f} pix'.format(rsft))
            QApplication.processEvents()
            self.lb_csft.setText('Col shfit: {:3.1f} pix'.format(csft))
            QApplication.processEvents()           
            
#            self.canvas1.update_img_one(img_temp, i)
        self.img_align = deepcopy(prj)    
        del prj
        if self.cb1.findText('Aligned image') < 0:
            self.cb1.addItem('Aligned image')
            self.update_canvas_img()
        self.pb_align.setText('Align Img')
        self.pb_align.setEnabled(True)
        print('Image aligned.\n Item "Aligned Image" has been added.')
        self.msg = 'Image aligning finished'
        self.update_msg()
        
    
    def find_rotation_center(self):
        # should "remove background first", to change the image to be like "fluorescence image"
        print('Finding rotation center...')
        self.pb_rot.setEnabled(False)
        QApplication.processEvents()
        init = np.float(self.tx_rot_guess.text()) # suppose to read from image display
        prj = np.max(self.img_align) - self.img_align # treat as Fluorescence image           
        ind = self.img_align.shape[1]//2 # index of slice to be used
        theta = self.theta
        self.rot_cen = tomopy.find_center(prj, theta, ind, init, tol=0.5)
        self.rot_cen = float(self.rot_cen)

        self.tx_rot_guess.setText('{:4.1f}'.format(self.rot_cen)) 
        self.msg = 'Rotation center is found at: x = {:3.1f}'.format(self.rot_cen)
        self.update_msg()
        print('Rotation center is found at: x = {:3.1f}'.format(self.rot_cen))
        self.pb_rot.setEnabled(True)
        del prj
       
    def test_rotation_center(self):
        self.pb_rot_test.setEnabled(False)
        QApplication.processEvents()
        
        print('testing rotation center')
        
        rs = int(self.tx_rc_test_s.text())
        re = int(self.tx_rc_test_e.text())
        ind = int(self.tx_which_sli.text())
        sh = self.img_align.shape

        prj = self.img_align[:, ind:ind+1, :]
        rot_test = np.zeros([re-rs, sh[2], sh[2]])
        self.rot_cen_list = np.arange(rs, re)
        for i in range(re-rs):
            rot_test[i] = tomopy.recon(prj, self.theta, center=self.rot_cen_list[i], algorithm='gridrec')
        self.img_rc_test = rot_test

        if self.cb1.findText('R.C test') < 0:
            self.cb1.addItem('R.C test')
        self.pb_rot_test.setEnabled(True)   
        QApplication.processEvents()
        print('Rotation center test finished. "R.C test" has been added.')
        self.msg = 'Rotation center test finished.'
        self.update_msg()
            
            
            
    def update_canvas_img(self):
        canvas = self.canvas1
        slide = self.sl1
        type_index = self.cb1.currentText()
#        self.pb_sh1.setEnabled(False)
        QApplication.processEvents()

        if self.chbx_rc_l.checkState():    canvas.draw_line = True
        else:   canvas.draw_line = False
     
        if type_index == 'Raw image':
            sh = self.img_tomo.shape
            canvas.img_stack = self.img_tomo
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0]-1, 0))
            self.current_image = self.img_tomo
        elif type_index == 'Background':
            sh = self.img_bkg.shape
            canvas.img_stack = self.img_bkg
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0]-1, 0))
            self.current_image = self.img_bkg
        elif type_index == 'Aligned image':
            sh = self.img_align.shape
            canvas.img_stack = self.img_align
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0]-1, 0))
            self.current_image = self.img_align
        elif type_index == 'Reconstructed':
            sh = self.img_recon.shape
            canvas.img_stack = self.img_recon
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0]-1, 0))
        elif type_index == 'Background removed':
            sh = self.img_bkg_removed.shape
            canvas.img_stack = self.img_bkg_removed
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0]-1, 0)) 
            self.current_image = self.img_bkg_removed
        elif type_index == 'R.C test':
            sh = self.img_rc_test.shape
            canvas.img_stack = self.img_rc_test
            canvas.special_info = self.rot_cen_list # sent rotation center list to the canvas, showing in display image title
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0]-1, 0))   
            self.current_image = self.img_rc_test
            
        elif type_index == 'Intensity plot':
            self.sl1.setMaximum(0)
            canvas.draw_line = True
            canvas.overlay_flag = False
            canvas.add_line()
            canvas.draw_line = False
            canvas.overlay_flag = True
            canvas.colorbar_on_flag = True
            
        
#        self.pb_sh1.setEnabled(True)
        QApplication.processEvents()

        
    def draw_RC(self):
        canvas = self.canvas1
        stat = True if self.chbx_rc_l.checkState() else False
        canvas.draw_line = stat
        x1 = float(self.tx_rot.text())
        x2 = x1
        y1 = 1
        y2 = float(self.img_align.shape[1])-1
        canvas.x, canvas.y = [x1, x2], [y1, y2]      
        
    def change_colormap(self):
        canvas = self.canvas1
        cmap = self.cb_cmap.currentText()
        canvas.colormap = cmap
        canvas.colorbar_on_flag = True
        canvas.update_img_one(canvas.current_img, canvas.current_img_index)
        
    def auto_contrast(self):
        canvas = self.canvas1
        cmin, cmax = canvas.auto_contrast()
        self.tx_cmax.setText('{:6.3f}'.format(cmax))
        self.tx_cmin.setText('{:6.3f}'.format(cmin))
        
    def set_contrast(self):
        canvas = self.canvas1
        cmax = np.float(self.tx_cmax.text())
        cmin = np.float(self.tx_cmin.text())
        canvas.set_contrast(cmin, cmax)
        
        
        
class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, width = 5, height = 3, dpi = 120, obj=[]):
        self.obj = obj
        self.fig = Figure(figsize=(width, height), dpi=dpi)      
        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')
        self.cmax = 1
        self.cmin = 0
        self.current_img = np.zeros([100, 100])
        self.current_img_index = 0
        self.colorbar_on_flag = True
        self.colormap = 'bone'
        self.draw_line = False    
        self.overlay_flag = True
        self.x, self.y, = [0., 0.], [0., 0.] 
        self.special_info = None
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)        
        self.img_stack = np.zeros([1, 100, 100])
        self.mpl_connect('motion_notify_event', self.mouse_moved)
        
    def mouse_moved(self, mouse_event):
        if mouse_event.inaxes:
            x, y = mouse_event.xdata, mouse_event.ydata 
            row = int(np.max([np.min([self.current_img.shape[0],y]),0]))
            col = int(np.max([np.min([self.current_img.shape[1],x]),0]))
            try:
                z = self.current_img[row][col]
                self.obj.lb_x_l.setText('x: {:3.1f}'.format(x))
                self.obj.lb_y_l.setText('y: {:3.1f}'.format(y))
                self.obj.lb_z_l.setText('intensity: {:3.3f}'.format(z))
            except:
                 print(self.current_img.shape)       
            
            
    def update_img_stack(self):
        if self.img_stack.shape[0] == 0:
            img_blank = np.zeros([100,100])
            return self.update_img_one(img_blank, img_index=self.current_img_index)
        return self.update_img_one(self.img_stack[0], img_index=0)
        
    def update_img_one(self, img=np.array([]), img_index=0):
        if len(img) == []: img = self.current_img
        add_title = ''
        self.current_img = img
        self.current_img_index = img_index
        self.im = self.axes.imshow(img, cmap=self.colormap, vmin=self.cmin, vmax=self.cmax)
        self.axes.axis('on')
        self.axes.set_aspect('equal','box')  
        if not(self.special_info is None):
            temp = self.special_info
            add_title = ' rot_cen: {:3.1f}'.format(temp[img_index]) 
            
        self.axes.set_title('current image: '+ str(img_index) + add_title)
        self.axes.title.set_fontsize(10)
        
        if self.colorbar_on_flag:
            self.add_colorbar()
            self.colorbar_on_flag = False
        self.add_line()
        self.draw()
        
    def add_line(self):
        if self.draw_line:
            if self.overlay_flag:
                self.axes.hold(True)
                self.axes.plot(self.x, self.y, 'y:')
                self.axes.hold(False)
            else:
                self.rm_colorbar()
                self.axes.hold(False)
                self.axes.plot(self.x, self.y, 'r.-')
                
                self.axes.hold(True)
                self.axes.axis('on')
                self.axes.set_aspect('auto') 
                self.draw()
        else: pass

    def set_contrast(self, cmin, cmax):
        self.cmax = cmax
        self.cmin = cmin
        self.colorbar_on_flag = True
        self.update_img_one(self.current_img, self.current_img_index)
        
    def auto_contrast(self):
        img = self.current_img
        self.cmax = np.max(img)
        self.cmin = np.min(img)
        self.colorbar_on_flag = True
        self.update_img_one(self.current_img, self.current_img_index)
        return self.cmin, self.cmax
        

    def rm_colorbar(self):
        try:
            self.cb.remove()
            self.draw()
        except: pass
    
    def add_colorbar(self):        
        if self.colorbar_on_flag:
            try:
                self.cb.remove()
                self.draw()
            except: pass
            self.divider = make_axes_locatable(self.axes)
            self.cax = self.divider.append_axes('right', size='3%', pad=0.06)
            self.cb = self.fig.colorbar(self.im, cax=self.cax, orientation='vertical')
            self.cb.ax.tick_params(labelsize=10)
            self.draw()      
            
            
           
if __name__ == '__main__':

    app = QApplication(sys.argv)
    tomo = App()
    tomo.show()    
    sys.exit(app.exec_())
