#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:39:15 2018

@author: mingyuan
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:07:42 2018

@author: mingyuan
"""

import sys
import os
# import re
# import json

from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QRadioButton, QApplication,
                             QLineEdit, QWidget, QPushButton, QLabel, QGroupBox,
                             QScrollBar, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QListWidget, QListWidgetItem, QAbstractItemView,
                             QComboBox, QCheckBox, QButtonGroup, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5 import QtCore
# from PyQt5.QtCore import pyqtSlot

import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
from skimage import io
import h5py
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from align_class import dftregistration
from image_binning import bin_ndarray
from scipy.ndimage.interpolation import shift
from scipy.signal import medfilt2d
import pandas as pd

global xanes


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'XANES Control'
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
        self.width = 1000
        self.height = 800
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2

        self.initUI()
        self.img_bkg = np.array([])
        self.img_xanes = np.array([])
        self.img_align = np.array([])
        self.img_recon = np.array([])
        self.msg = ''

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.fpath = os.getcwd()
        self.roi_file_id = 0

        grid = QGridLayout()
        gpbox_prep = self.layout_GP_prepare()
        gpbox_msg = self.layout_msg()
        gpbox_recon = self.layout_recon()

        grid.addWidget(gpbox_prep, 0, 1)
        grid.addLayout(gpbox_msg, 1, 1)
        grid.addWidget(gpbox_recon, 2, 1)

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
        self.lb_msg.setText('Message:')

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
        lb_ang.setText('XANES energy:')
        lb_ang.setFixedWidth(120)

        self.lb_ang1 = QLabel()
        self.lb_ang1.setFont(self.font2)
        self.lb_ang1.setText('No energy data ...')
        self.lb_ang1.setFixedWidth(350)

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

        lb_mod = QLabel()
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

        lb_hdf_xanes = QLabel()
        lb_hdf_xanes.setFont(self.font2)
        lb_hdf_xanes.setText('Dataset for XANES:')
        lb_hdf_xanes.setFixedWidth(130)

        lb_hdf_bkg = QLabel()
        lb_hdf_bkg.setFont(self.font2)
        lb_hdf_bkg.setText('  bkg:')
        lb_hdf_bkg.setFixedWidth(40)

        lb_hdf_dark = QLabel()
        lb_hdf_dark.setFont(self.font2)
        lb_hdf_dark.setText('  dark:')
        lb_hdf_dark.setFixedWidth(40)

        lb_hdf_eng = QLabel()
        lb_hdf_eng.setFont(self.font2)
        lb_hdf_eng.setText('  energy:')
        lb_hdf_eng.setFixedWidth(60)

        self.tx_hdf_xanes = QLineEdit()
        self.tx_hdf_xanes.setText('img_xanes')
        self.tx_hdf_xanes.setFixedWidth(85)
        self.tx_hdf_xanes.setFont(self.font2)
        self.tx_hdf_xanes.setVisible(True)

        self.tx_hdf_bkg = QLineEdit()
        self.tx_hdf_bkg.setText('img_bkg')
        self.tx_hdf_bkg.setFixedWidth(85)
        self.tx_hdf_bkg.setFont(self.font2)
        self.tx_hdf_bkg.setVisible(True)

        self.tx_hdf_dark = QLineEdit()
        self.tx_hdf_dark.setText('img_dark')
        self.tx_hdf_dark.setFixedWidth(85)
        self.tx_hdf_dark.setFont(self.font2)
        self.tx_hdf_dark.setVisible(True)

        self.tx_hdf_eng = QLineEdit()
        self.tx_hdf_eng.setText('xanes_eng')
        self.tx_hdf_eng.setFixedWidth(85)
        self.tx_hdf_eng.setFont(self.font2)
        self.tx_hdf_eng.setVisible(True)

        self.type_group = QButtonGroup()
        self.type_group.setExclusive(True)
        self.rd_absp = QRadioButton('Absorption')
        self.rd_absp.setFont(self.font2)
        self.rd_absp.setFixedWidth(100)
        self.rd_absp.setChecked(True)
        self.rd_flrc = QRadioButton('Fluorescence')
        self.rd_flrc.setFont(self.font2)
        self.rd_flrc.setFixedWidth(120)
        self.rd_flrc.setChecked(False)
        self.type_group.addButton(self.rd_absp)
        self.type_group.addButton(self.rd_flrc)

        lb_fp = QLabel()
        lb_fp.setFont(self.font2)
        lb_fp.setText('Image loaded: ')
        lb_fp.setFixedWidth(100)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_ld)
        hbox1.addWidget(self.rd_tif)
        hbox1.addWidget(self.rd_hdf)
        hbox1.addWidget(lb_hdf_xanes)
        hbox1.addWidget(self.tx_hdf_xanes)
        hbox1.addWidget(lb_hdf_bkg)
        hbox1.addWidget(self.tx_hdf_bkg)
        hbox1.addWidget(lb_hdf_dark)
        hbox1.addWidget(self.tx_hdf_dark)
        hbox1.addWidget(lb_hdf_eng)
        hbox1.addWidget(self.tx_hdf_eng)
        #        hbox1.addWidget(self.pb_ld)
        hbox1.addWidget(lb_empty2)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_mod)
        hbox2.addWidget(self.rd_absp)
        hbox2.addWidget(self.rd_flrc)
        hbox2.addWidget(lb_empty2)
        hbox2.addWidget(self.pb_ld)
        hbox2.addWidget(lb_empty3)

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

        gpbox = QGroupBox('XANES fitting')
        gpbox.setFont(self.font1)

        xanes_prep_layout = self.layout_xanes_prep()
        xanes_roi_layout = self.layout_plot_spec
        xanes_fit_edge_layout = self.layout_fit_pre_post_edges()
        canvas_layout = self.layout_canvas()

        vbox_xanes_recon = QVBoxLayout()
        vbox_xanes_recon.addLayout(xanes_prep_layout)
        vbox_xanes_recon.addLayout(xanes_roi_layout)
        vbox_xanes_recon.addLayout(xanes_fit_edge_layout)
        vbox_xanes_recon.addWidget(lb_empty)
        vbox_xanes_recon.setAlignment(QtCore.Qt.AlignTop)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox_xanes_recon)
        hbox.addLayout(canvas_layout)
        hbox.addWidget(lb_empty2)
        hbox.setAlignment(QtCore.Qt.AlignLeft)
        gpbox.setLayout(hbox)
        return gpbox

    @property
    def layout_plot_spec(self):
        lb_empty = QLabel()
        lb_roi = QLabel()
        lb_roi.setFont(self.font1)
        lb_roi.setText('ROI for Spec.')
        lb_roi.setFixedWidth(100)

        lb_info = QLabel()
        lb_info.setFont(self.font2)
        lb_info.setStyleSheet('color: rgb(200, 50, 50);')
        lb_info.setText('Spectrum calc. based on current image stack')

        lb_roi_x1 = QLabel()
        lb_roi_x1.setText('Top-left  x:')
        lb_roi_x1.setFont(self.font2)
        lb_roi_x1.setFixedWidth(80)

        lb_roi_y1 = QLabel()
        lb_roi_y1.setText('y:')
        lb_roi_y1.setFont(self.font2)
        lb_roi_y1.setFixedWidth(20)

        lb_roi_x2 = QLabel()
        lb_roi_x2.setText('Bot-right x:')
        lb_roi_x2.setFont(self.font2)
        lb_roi_x2.setFixedWidth(80)

        lb_roi_y2 = QLabel()
        lb_roi_y2.setText('y:')
        lb_roi_y2.setFont(self.font2)
        lb_roi_y2.setFixedWidth(20)

        self.tx_roi_x1 = QLineEdit()
        self.tx_roi_x1.setText('0')
        self.tx_roi_x1.setFont(self.font2)
        self.tx_roi_x1.setFixedWidth(50)

        self.tx_roi_y1 = QLineEdit()
        self.tx_roi_y1.setText('0')
        self.tx_roi_y1.setFont(self.font2)
        self.tx_roi_y1.setFixedWidth(50)

        self.tx_roi_x2 = QLineEdit()
        self.tx_roi_x2.setText('1')
        self.tx_roi_x2.setFont(self.font2)
        self.tx_roi_x2.setFixedWidth(50)

        self.tx_roi_y2 = QLineEdit()
        self.tx_roi_y2.setText('1')
        self.tx_roi_y2.setFont(self.font2)
        self.tx_roi_y2.setFixedWidth(50)

        self.pb_roi_draw = QPushButton('Draw ROI')
        self.pb_roi_draw.setFont(self.font2)
        self.pb_roi_draw.clicked.connect(self.draw_roi)
        self.pb_roi_draw.setFixedWidth(105)
        self.pb_roi_draw.setVisible(True)

        self.pb_roi_plot = QPushButton('Plot Spec.')
        self.pb_roi_plot.setFont(self.font2)
        self.pb_roi_plot.clicked.connect(self.plot_spectrum)
        self.pb_roi_plot.setFixedWidth(105)
        self.pb_roi_plot.setVisible(True)

        self.pb_roi_hide = QPushButton('Hide ROI')
        self.pb_roi_hide.setFont(self.font2)
        self.pb_roi_hide.clicked.connect(self.hide_roi)
        self.pb_roi_hide.setFixedWidth(105)
        self.pb_roi_hide.setVisible(True)

        self.pb_roi_show = QPushButton('Show ROI')
        self.pb_roi_show.setFont(self.font2)
        self.pb_roi_show.clicked.connect(self.show_roi)
        self.pb_roi_show.setFixedWidth(105)
        self.pb_roi_show.setVisible(True)

        self.pb_roi_reset = QPushButton('Reset ROI')
        self.pb_roi_reset.setFont(self.font2)
        self.pb_roi_reset.clicked.connect(self.reset_roi)
        self.pb_roi_reset.setFixedWidth(105)
        self.pb_roi_reset.setVisible(True)

        self.pb_roi_export = QPushButton('Export Spec.')
        self.pb_roi_export.setFont(self.font2)
        self.pb_roi_export.clicked.connect(self.export_spectrum)
        self.pb_roi_export.setFixedWidth(105)
        self.pb_roi_export.setVisible(True)

        lb_file_index = QLabel()
        lb_file_index.setFont(self.font2)
        lb_file_index.setText('  File index for export:')
        lb_file_index.setFixedWidth(155)

        self.tx_file_index = QLineEdit()
        self.tx_file_index.setFixedWidth(50)
        self.tx_file_index.setFont(self.font2)
        self.tx_file_index.setText(str(self.roi_file_id))

        self.lst_roi = QListWidget()
        self.lst_roi.setFont(self.font2)
        self.lst_roi.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_roi.setFixedWidth(80)
        self.lst_roi.setFixedHeight(100)

        lb_lst_roi = QLabel()
        lb_lst_roi.setFont(self.font2)
        lb_lst_roi.setText('ROI list:')
        lb_lst_roi.setFixedWidth(80)

        # hbox_roi_tl = QHBoxLayout()
        # hbox_roi_tl.addWidget(lb_roi_x1)
        # hbox_roi_tl.addWidget(self.tx_roi_x1)
        # hbox_roi_tl.addWidget(lb_roi_y1)
        # hbox_roi_tl.addWidget(self.tx_roi_y1)
        # hbox_roi_tl.setAlignment(QtCore.Qt.AlignLeft)
        #
        # hbox_roi_bd = QHBoxLayout()
        # hbox_roi_bd.addWidget(lb_roi_x2)
        # hbox_roi_bd.addWidget(self.tx_roi_x2)
        # hbox_roi_bd.addWidget(lb_roi_y2)
        # hbox_roi_bd.addWidget(self.tx_roi_y2)
        # hbox_roi_bd.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button1 = QHBoxLayout()
        hbox_roi_button1.addWidget(self.pb_roi_draw)
        hbox_roi_button1.addWidget(self.pb_roi_reset)
        hbox_roi_button1.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button2 = QHBoxLayout()
        hbox_roi_button2.addWidget(self.pb_roi_show)
        hbox_roi_button2.addWidget(self.pb_roi_hide)
        hbox_roi_button2.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button3 = QHBoxLayout()
        hbox_roi_button3.addWidget(self.pb_roi_plot)
        hbox_roi_button3.addWidget(self.pb_roi_export)
        hbox_roi_button3.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button4 = QHBoxLayout()
        hbox_roi_button4.addWidget(lb_file_index)
        hbox_roi_button4.addWidget(self.tx_file_index)
        hbox_roi_button4.setAlignment(QtCore.Qt.AlignLeft)

        vbox_roi = QVBoxLayout()
        vbox_roi.setContentsMargins(0, 0, 0, 0)
        vbox_roi.addLayout(hbox_roi_button1)
        vbox_roi.addLayout(hbox_roi_button2)
        vbox_roi.addLayout(hbox_roi_button3)
        vbox_roi.addLayout(hbox_roi_button4)
        vbox_roi.setAlignment(QtCore.Qt.AlignLeft)

        vbox_lst = QVBoxLayout()
        vbox_lst.addWidget(lb_lst_roi, 0, QtCore.Qt.AlignTop)
        vbox_lst.addWidget(self.lst_roi, 0, QtCore.Qt.AlignTop)
        vbox_lst.addWidget(lb_empty)
        vbox_lst.setAlignment(QtCore.Qt.AlignLeft)

        box_roi = QHBoxLayout()
        box_roi.addLayout(vbox_roi)
        box_roi.addLayout(vbox_lst)
        box_roi.addWidget(lb_empty, 0, QtCore.Qt.AlignLeft)
        box_roi.setAlignment(QtCore.Qt.AlignLeft)
        # box_roi.setAlignment(QtCore.Qt.AlignTop)

        box_roi_tot = QVBoxLayout()
        box_roi_tot.addWidget(lb_roi)
        box_roi_tot.addWidget(lb_info)
        box_roi_tot.addLayout(box_roi)
        # box_roi_tot.addWidget(lb_empty)
        box_roi_tot.setAlignment(QtCore.Qt.AlignLeft)

        return box_roi_tot

    def layout_fit_pre_post_edges(self):
        lb_empty = QLabel()
        lb_fit_edge = QLabel()
        lb_fit_edge.setFont(self.font1)
        lb_fit_edge.setText('Fit pre-post edges')
        lb_fit_edge.setFixedWidth(150)

        lb_fit_pre_s = QLabel()
        lb_fit_pre_s.setText('Pre -edge start:')
        lb_fit_pre_s.setFont(self.font2)
        lb_fit_pre_s.setFixedWidth(120)

        lb_fit_pre_e = QLabel()
        lb_fit_pre_e.setText('end:')
        lb_fit_pre_e.setFont(self.font2)
        lb_fit_pre_e.setFixedWidth(40)

        lb_fit_post_s = QLabel()
        lb_fit_post_s.setText('Post-edge start:')
        lb_fit_post_s.setFont(self.font2)
        lb_fit_post_s.setFixedWidth(120)

        lb_fit_post_e = QLabel()
        lb_fit_post_e.setText('end:')
        lb_fit_post_e.setFont(self.font2)
        lb_fit_post_e.setFixedWidth(40)

        self.tx_fit_pre_s = QLineEdit()
        self.tx_fit_pre_s.setFont(self.font2)
        self.tx_fit_pre_s.setFixedWidth(50)

        self.tx_fit_pre_e = QLineEdit()
        self.tx_fit_pre_e.setFont(self.font2)
        self.tx_fit_pre_e.setFixedWidth(50)

        self.tx_fit_post_s = QLineEdit()
        self.tx_fit_post_s.setFont(self.font2)
        self.tx_fit_post_s.setFixedWidth(50)

        self.tx_fit_post_e = QLineEdit()
        self.tx_fit_post_e.setFont(self.font2)
        self.tx_fit_post_e.setFixedWidth(50)

        lb_fit_norm = QLabel()
        lb_fit_norm.setFont(self.font2)
        lb_fit_norm.setText('For selected ROI:')
        lb_fit_norm.setFixedWidth(120)

        self.pb_fit_norm = QPushButton('For and Norm. (ROI)')
        self.pb_fit_norm.setFont(self.font2)
        self.pb_fit_norm.clicked.connect(self.fit_edge)
        self.pb_fit_norm.setFixedWidth(150)
        self.pb_fit_norm.setVisible(True)

        lb_fit_norm_px = QLabel()
        lb_fit_norm_px.setFont(self.font2)
        lb_fit_norm_px.setText('Fit single pixel:')
        lb_fit_norm_px.setFixedWidth(120)

        self.pb_fit_norm_px = QPushButton('Fit and Norm. (Pix)')
        self.pb_fit_norm_px.setFont(self.font2)
        self.pb_fit_norm_px.clicked.connect(self.fit_edge_px)
        self.pb_fit_norm_px.setFixedWidth(150)
        self.pb_fit_norm_px.setVisible(True)

        # self.pb_fit_edge = QPushButton('Fit pre-post edges')
        # self.pb_fit_edge.setFont(self.font2)
        # self.pb_fit_edge.clicked.connect(self.fit_edge)
        # self.pb_fit_edge.setFixedWidth(150)
        # self.pb_fit_edge.setVisible(True)
        #
        # self.pb_norm_edge = QPushButton('Normalization')
        # self.pb_norm_edge.setFont(self.font2)
        # self.pb_norm_edge.clicked.connect(self.norm_edge)
        # self.pb_norm_edge.setFixedWidth(150)
        # self.pb_norm_edge.setVisible(True)

        hbox_fit_pre = QHBoxLayout()
        hbox_fit_pre.addWidget(lb_fit_pre_s)
        hbox_fit_pre.addWidget(self.tx_fit_pre_s)
        hbox_fit_pre.addWidget(lb_fit_pre_e)
        hbox_fit_pre.addWidget(self.tx_fit_pre_e)
        hbox_fit_pre.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_post = QHBoxLayout()
        hbox_fit_post.addWidget(lb_fit_post_s)
        hbox_fit_post.addWidget(self.tx_fit_post_s)
        hbox_fit_post.addWidget(lb_fit_post_e)
        hbox_fit_post.addWidget(self.tx_fit_post_e)
        hbox_fit_post.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_pb = QHBoxLayout()
        hbox_fit_pb.addWidget(lb_fit_norm)
        hbox_fit_pb.addWidget(self.pb_fit_norm)
        hbox_fit_pb.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_pb_px = QHBoxLayout()
        hbox_fit_pb_px.addWidget(lb_fit_norm_px)
        hbox_fit_pb_px.addWidget(self.pb_fit_norm_px)
        hbox_fit_pb_px.setAlignment(QtCore.Qt.AlignLeft)

        vbox_fit = QVBoxLayout()
        vbox_fit.addWidget(lb_fit_edge)
        vbox_fit.addLayout(hbox_fit_pre)
        vbox_fit.addLayout(hbox_fit_post)
        vbox_fit.addLayout(hbox_fit_pb)
        vbox_fit.addLayout(hbox_fit_pb_px)
        vbox_fit.setAlignment(QtCore.Qt.AlignLeft)

        return vbox_fit

    def layout_xanes_prep(self):
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
        self.pb_align.clicked.connect(self.xanes_align_img)
        self.pb_align.setEnabled(False)
        self.pb_align.setFixedWidth(150)

        self.pb_norm1 = QPushButton('Norm. Bkg (opt.)')
        self.pb_norm1.setFont(self.font2)
        self.pb_norm1.clicked.connect(self.xanes_norm_bkg)
        self.pb_norm1.setEnabled(False)
        self.pb_norm1.setFixedWidth(150)

        self.lb_ali = QLabel()
        self.lb_ali.setFont(self.font2)
        self.lb_ali.setText('  Aligning slice: ')
        self.lb_ali.setFixedWidth(150)

        self.pb_filt = QPushButton('Median filter')
        self.pb_filt.setFont(self.font2)
        self.pb_filt.clicked.connect(self.xanes_img_smooth)
        self.pb_filt.setEnabled(False)
        self.pb_filt.setFixedWidth(150)

        lb_filt = QLabel()
        lb_filt.setFont(self.font2)
        lb_filt.setText('  kernal size: ')
        lb_filt.setFixedWidth(80)

        self.tx_filt = QLineEdit(self)
        self.tx_filt.setFont(self.font2)
        self.tx_filt.setText('3')
        self.tx_filt.setValidator(QIntValidator())
        self.tx_filt.setFixedWidth(50)

        hbox_filt = QHBoxLayout()
        hbox_filt.addWidget(self.pb_filt)
        hbox_filt.addWidget(lb_filt)
        hbox_filt.addWidget(self.tx_filt)
        hbox_filt.setAlignment(QtCore.Qt.AlignLeft)

        hbox_prep = QHBoxLayout()
        hbox_prep.addWidget(self.pb_rmbg)
        hbox_prep.addWidget(self.pb_norm1)
        hbox_prep.setAlignment(QtCore.Qt.AlignLeft)

        hbox_norm = QHBoxLayout()
        hbox_norm.addWidget(self.pb_align)
        hbox_norm.addWidget(self.lb_ali)
        hbox_norm.setAlignment(QtCore.Qt.AlignLeft)

        vbox_prep = QVBoxLayout()
        vbox_prep.addWidget(lb_prep)
        vbox_prep.addLayout(hbox_prep)
        vbox_prep.addLayout(hbox_filt)
        vbox_prep.addLayout(hbox_norm)
        vbox_prep.addWidget(lb_empty)

        return vbox_prep

    def layout_canvas(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)

        self.canvas1 = MyCanvas(obj=self)

        self.sl1 = QScrollBar(QtCore.Qt.Horizontal)
        self.sl1.setMaximum(0)
        self.sl1.setMinimum(0)
        self.sl1.valueChanged.connect(self.sliderval)

        self.cb1 = QComboBox()
        self.cb1.setFont(self.font2)
        self.cb1.addItem('Raw image')
        self.cb1.addItem('Background')
        self.cb1.currentIndexChanged.connect(self.update_canvas_img)

        self.pb_del = QPushButton('Del. single image')
        self.pb_del.setToolTip(
            'Delete single image, it will delete the same slice in other images (e.g., "raw image", "aligned image", "background removed" ')
        self.pb_del.setFont(self.font2)
        self.pb_del.clicked.connect(self.delete_single_img)
        self.pb_del.setEnabled(False)
        self.pb_del.setFixedWidth(150)

        hbox_can_l = QHBoxLayout()
        hbox_can_l.addWidget(self.cb1)
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

        cmap = ['gray', 'bone', 'viridis', 'terrain', 'gnuplot', 'bwr', 'plasma', 'PuBu', 'summer', 'rainbow', 'jet']
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
        self.tx_hdf_xanes.setEnabled(True)
        self.tx_hdf_bkg.setEnabled(True)
        self.tx_hdf_dark.setEnabled(True)
        self.tx_hdf_eng.setEnabled(True)

    def select_tif_file(self):
        self.tx_hdf_xanes.setEnabled(False)
        self.tx_hdf_bkg.setEnabled(False)
        self.tx_hdf_dark.setEnabled(False)
        self.tx_hdf_eng.setEnabled(False)

    def xanes_norm_bkg(self):
        '''
        Treat if it is fluorescent image.
        calculate the mean value of 10% of the lowest pixel value, and substract this value from the whole image
        '''
        self.pb_norm1.setText('normalizing ..')
        self.pb_norm1.setEnabled(False)
        QApplication.processEvents()

        prj = deepcopy(self.img_bkg_removed)
        prj[np.isnan(prj)] = 0
        prj[np.isinf(prj)] = 0
        prj[prj < 0] = 0
        s = prj.shape
        prj_sort = np.sort(prj)  # sort each slice
        prj_sort = prj_sort[:, :, 0:int(s[1] * s[2] * 0.1)]
        slice_avg = np.mean(np.mean(prj_sort, axis=2), axis=1)  # average for each slice
        prj = np.array([prj[i] - slice_avg[i] for i in range(s[0])])
        prj[prj < 0] = 0
        self.img_bkg_removed = deepcopy(prj)
        self.img_align = deepcopy(prj)  # synchronize self.img_align and self.img_bkg_removed
        del prj, slice_avg
        self.pb_norm1.setEnabled(True)
        self.pb_norm1.setText('Norm. Bkg. (opt.) ')
        QApplication.processEvents()
        self.cb1.setCurrentText('Background removed')
        self.update_canvas_img()

    def xanes_img_smooth(self):
        self.pb_filt.setEnabled(False)
        self.pb_filt.setText('Smoothing ...')
        QApplication.processEvents()
        canvas = self.canvas1
        img_stack = canvas.img_stack
        kernal_size = int(self.tx_filt.text())
        for i in range(img_stack.shape[0]):
            img_stack[i] = medfilt2d(img_stack[i], kernal_size)

        self.img_smooth = deepcopy(img_stack)
        if self.cb1.findText('Image smoothed') < 0:
            self.cb1.addItem('Image smoothed')
        self.msg = 'Image smoothed'
        self.update_msg()
        self.cb1.setCurrentText('Image smoothed')
        self.update_canvas_img()
        self.pb_filt.setEnabled(True)
        self.pb_filt.setText('Median filter')
        QApplication.processEvents()

    def plot_spectrum(self):
        # canvas = self.canvas1
        # x1 = int(float((self.tx_roi_x1.text())))
        # y1 = int(float((self.tx_roi_y1.text())))
        # x2 = int(float((self.tx_roi_x2.text())))
        # y2 = int(float((self.tx_roi_y2.text())))
        # assert(x2 >= x1), "bottom x should larger than top x"
        # assert(y2 >= y1), "right y should larger than left y"
        # img_stack = deepcopy(canvas.img_stack)
        # self.roi_spec = np.mean(np.mean(img_stack[:,y1:y2,x1:x2,], axis=1), axis=1)
        # if self.cb1.findText('ROI spectrum') < 0:
        #     self.cb1.addItem('ROI spectrum')
        # print('Generate ROI spectrum\n. Item "ROI spectrum" has been added. ')
        # self.msg = 'ROI spectrum added.'
        # self.update_msg()
        # self.cb1.setCurrentText('ROI spectrum')
        #
        # self.update_canvas_img()

        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)
        plt.figure();
        roi_color = canvas.roi_color
        roi_list = canvas.roi_list
        x = self.xanes_eng
        legend = []
        # print('roi list: {}'.format(roi_list) )
        for item in self.lst_roi.selectedItems():
            plt.hold(True)
            print(item.text())
            plot_color = roi_color[item.text()]
            roi_cord = np.int32(np.array(roi_list[item.text()]))
            plot_label = item.text()
            x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x1 = min(x1, x2)
            x2 = max(x1, x2)
            y1 = min(y1, y2)
            y2 = max(y1, y2)
            roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)
            line, = plt.plot(x, roi_spec, marker='.', color=plot_color, label=plot_label)
            legend.append(line)
        print(legend)
        plt.legend(handles=legend)
        plt.show()

    def show_roi(self):
        plt.figure()
        canvas = self.canvas1
        current_img = canvas.current_img
        current_colormap = canvas.colormap
        cmin, cmax = canvas.cmin, canvas.cmax
        s = current_img.shape
        plt.imshow(current_img, cmap=current_colormap, vmin=cmin, vmax=cmax)
        plt.axis('equal')
        plt.axis('off')
        plt.colorbar()
        roi_color = canvas.roi_color
        roi_list = canvas.roi_list
        for item in self.lst_roi.selectedItems():
            plt.hold(True)
            print(item.text())
            plot_color = roi_color[item.text()]
            roi_cord = np.int32(np.array(roi_list[item.text()]))
            plot_label = item.text()
            x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x = [x1, x2, x2, x1, x1]
            y = [y1, y1, y2, y2, y1]
            plt.plot(x, y, '-', color=plot_color, linewidth=1.0, label=plot_label)
            roi_name = '#' + plot_label.split('_')[-1]
            plt.annotate(roi_name, xy=(x1, y1 - 40),
                         bbox={'facecolor': plot_color, 'alpha': 0.5, 'pad': 2},
                         fontsize=10)
        # self.pb_roi_showhide.setText('Hide ROI')
        plt.show()

    def hide_roi(self):
        # self.pb_roi_showhide.text() == 'Hide ROI':
        self.update_canvas_img()
        # self.pb_roi_showhide.setText('Show ROI')

    def export_spectrum(self):
        self.show_roi()
        self.tx_file_index
        try:
            os.makedirs(self.fpath + '/ROI')
        except:
            # print(self.fpath + '/ROI failed')
            pass
        fn_spec = 'roi_spectrum_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
        fn_spec = self.fpath + '/ROI/' + fn_spec

        fn_cord = 'roi_coordinates_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
        fn_cord = self.fpath + '/ROI/' + fn_cord

        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)
        roi_list = canvas.roi_list
        x = self.xanes_eng
        roi_dict_spec = {'X-Eng(eV)': pd.Series(x)}
        roi_dict_cord = {}
        for item in self.lst_roi.selectedItems():
            roi_cord = np.int32(np.array(roi_list[item.text()]))
            plot_label = item.text()
            x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x1 = min(x1, x2)
            x2 = max(x1, x2)
            y1 = min(y1, y2)
            y2 = max(y1, y2)
            area = (x2 - x1) * (y2 - y1)
            roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)
            roi_spec = np.around(roi_spec, 3)
            roi_dict_spec[plot_label] = pd.Series(roi_spec)
            roi_dict_cord[plot_label] = pd.Series([x1, y1, x2, y2, area], index=['x1', 'y1', 'x2', 'y2', 'area'])
        df_spec = pd.DataFrame(roi_dict_spec)
        df_cord = pd.DataFrame(roi_dict_cord)

        with open(fn_spec, 'w') as f:
            df_spec.to_csv(f, sep='\t', index=False)
        with open(fn_cord, 'w') as f:
            df_cord.to_csv(f, sep='\t')

        self.roi_file_id += 1
        self.tx_file_index.setText(str(self.roi_file_id))
        print(fn_spec + '  saved')
        self.msg = 'ROI spectrum file saved:   ' + fn_spec
        self.update_msg()

    def reset_roi(self):
        canvas = self.canvas1
        self.lst_roi.clear()
        canvas.current_roi = [0, 0, 0, 0]
        canvas.current_color = 'red'
        canvas.roi_list = {}
        canvas.roi_count = 0
        canvas.roi_color = {}
        self.update_canvas_img()
        s = canvas.current_img.shape
        self.tx_roi_x1.setText('0.0')
        self.tx_roi_x2.setText('{:3.1f}'.format(s[1]))
        self.tx_roi_y1.setText('{:3.1f}'.format(0))
        self.tx_roi_y2.setText('{:3.1f}'.format(s[0]))
        pass

    def draw_roi(self):
        self.pb_roi_draw.setEnabled(False)
        QApplication.processEvents()
        canvas = self.canvas1
        canvas.draw_roi()
        pass

    def fit_edge(self):
        pre_s = float(self.tx_fit_pre_s.text())
        pre_e = float(self.tx_fit_pre_e.text())
        post_s = float(self.tx_fit_post_s.text())
        post_e = float(self.tx_fit_post_e.text())
        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)
        roi_list = canvas.roi_list
        x_eng = deepcopy(self.xanes_eng)
        for item in self.lst_roi.selectedItems():
            plt.figure()  # generate figure for each roi
            plt.subplot(211)
            roi_cord = np.int32(np.array(roi_list[item.text()]))
            plot_label = item.text()
            x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)

            # fit pre-edge
            x, y = x_eng[x_eng < pre_e], roi_spec[x_eng < pre_e]
            x, y = x[x > pre_s], y[x > pre_s]
            if len(x) == 1:
                x = np.squeeze([x, x])
                y = np.squeeze([y, y])
            p = polyfit(x, y, 1)
            y_pre_fit = polyval(p, x_eng)

            # fit post_edge
            x, y = x_eng[x_eng < post_e], roi_spec[x_eng < post_e]
            x, y = x[x > post_s], y[x > post_s]
            if len(x) == 1:
                x = np.squeeze([x, x])
                y = np.squeeze([y, y])
            p = polyfit(x, y, 1)
            y_post_fit = polyval(p, x_eng)

            plt.plot(x_eng, roi_spec, '.', color='gray')
            plt.plot(x_eng, y_pre_fit, 'b', linewidth=1)
            plt.plot(x_eng, y_post_fit, 'r', linewidth=1)
            plt.title('pre-post edge fitting for ' + plot_label)

            plt.subplot(212)  # plot normalized spectrum
            roi_spec_norm = (roi_spec - y_pre_fit) / y_post_fit
            plt.plot(x_eng, roi_spec_norm)
            plt.title('normalized spectrum for ' + plot_label)
            plt.show()

        pass



    # ##########################################
    def fit_edge_px(self): # not finished yet
        pre_s = float(self.tx_fit_pre_s.text())
        pre_e = float(self.tx_fit_pre_e.text())
        post_s = float(self.tx_fit_post_s.text())
        post_e = float(self.tx_fit_post_e.text())
        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)
        x_eng = deepcopy(self.xanes_eng)

        # pre_edge
        x_pre, y_pre = x_eng[x_eng < pre_e], img_stack[x_eng < pre_e, : , :]
        x_pre, y_pre = x_pre[x_pre > pre_s], y_pre[x_pre > pre_s, :, :]
        s = y_pre.shape
        y = y_pre.reshape([s[0], s[1]*s[2]])
        p = polyfit(x_pre,y,1)
        y_fit_pre = polyval(x_eng, p)
        y_fit_pre = y_fit_pre.T
        y_fit_pre = y_fit_pre.reshape(img_stack.shape)

        # post_edge
        x_post, y_post = x_eng[x_eng < post_e], img_stack[x_eng < post_e, :, :]
        x_post, y_post = x_post[x_post > post_s], y_post[x_post > post_s, :, :]
        s = y_post.shape
        y = y_post.reshape([s[0], s[1] * s[2]])
        p = polyfit(x_post, y, 1)
        y_fit_post = polyval(x_eng, p)
        y_fit_post = y_fit_post.T
        y_fit_post = y_fit_post.reshape(img_stack.shape)


        pass

    def norm_edge(self):
        pass

    def delete_single_img(self):
        msg = QMessageBox()
        msg.setText('This will delete the image in \"raw data\", \"aligned image\", and \"background removed\" images')
        msg.setWindowTitle('Delete single image')
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        reply = msg.exec_()
        if reply == QMessageBox.Ok:
            img_type = self.cb1.currentText()
            current_slice = self.sl1.value()
            if img_type == 'Background':
                try:
                    self.img_bkg = np.delete(self.img_bkg, current_slice, axis=0)
                except:
                    pass
            else:
                try:
                    self.img_xanes = np.delete(self.img_xanes, current_slice, axis=0)
                except:
                    print('cannot delete img_xanes')

                try:
                    self.img_bkg_removed = np.delete(self.img_bkg_removed, current_slice, axis=0)
                except:
                    print('cannot delete img_bkg_removed')

                try:
                    self.img_align = np.delete(self.img_align, current_slice, axis=0)
                except:
                    print('cannot delete img_align')

                try:
                    self.img_smooth = np.delete(self.img_smooth, current_slice, axis=0)
                except:
                    print('cannot delete img_smooth')

                try:
                    self.xanes_eng = np.delete(self.xanes_eng, current_slice, axis=0)
                    st = '{0:3.1f}, {1:3.1f}, ..., {2:3.1f}  (totally, {3} angles)'.format(self.xanes_eng[0],
                                                                                           self.xanes_eng[1],
                                                                                           self.xanes_eng[-1],
                                                                                           len(self.xanes_eng))
                    self.lb_ang1.setText(st)  # update angle information showing on the label
                except:
                    print('cannot delete energy')
            self.msg = 'image #{} has been deleted'.format(current_slice)
            self.update_msg()
            self.update_canvas_img()

    def select_recon_slice(self):
        t = self.img_xanes.shape[1]
        s = t // 2

        if self.rd_sli_2.isChecked():  # all slice
            self.tx_sli_st.setEnabled(False)
            self.tx_sli_end.setEnabled(False)
            self.tx_sli_st.setText(str(0))
            self.tx_sli_end.setText(str(t))
        else:  # multi_slice
            self.tx_sli_st.setEnabled(True)
            self.tx_sli_end.setEnabled(True)
            self.tx_sli_st.setText(str(s))
            self.tx_sli_end.setText(str(s + 1))
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

        if len(self.tx_hdf_xanes.text()):
            dataset_xanes = self.tx_hdf_xanes.text()
        else:
            dataset_xanes = 'img_xanes'

        if len(self.tx_hdf_bkg.text()):
            dataset_bkg = self.tx_hdf_bkg.text()
        else:
            dataset_bkg = 'img_bkg'

        if len(self.tx_hdf_dark.text()):
            dataset_dark = self.tx_hdf_dark.text()
        else:
            dataset_dark = 'img_dark'

        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        if self.rd_hdf.isChecked() == True:
            file_type = 'hdf files (*.h5)'
        else:
            file_type = 'tiff files (*.tif)'

        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)

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
                # read xanes-scan image
                try:
                    #                    self.img_xanes = np.array(f['img_xanes'])
                    self.img_xanes = np.array(f[dataset_xanes])
                    self.rot_cen = self.img_xanes.shape[2] / 2
                    print('Image size: ' + str(self.img_xanes.shape))
                    self.pb_rmbg.setEnabled(True)  # remove background
                    self.pb_del.setEnabled(True)  # delete single image
                    self.pb_filt.setEnabled(True)  # delete single image
                    self.msg = 'image shape: {0}'.format(self.img_xanes.shape)
                    self.update_msg()
                except:
                    self.img_xanes = np.zeros([2, 100, 100])
                    print('xanes image not exist')
                finally:
                    self.img_bkg_removed = deepcopy(self.img_xanes)
                    self.img_align = deepcopy(self.img_bkg_removed)
                    self.update_canvas_img()

                # read background image    
                try:
                    self.img_bkg = np.array(f[dataset_bkg])

                except:
                    self.img_bkg = np.ones(self.img_xanes.shape)
                    print('\nbkg image not exist')

                # read dark image    
                try:
                    self.img_dark = np.array(f[dataset_dark])
                except:
                    self.img_dark = np.zeros(self.img_xanes.shape)
                    print('\ndark image not exist')

                # read theta
                try:
                    self.xanes_eng = np.array(f['xanes_eng'])
                    if min(self.xanes_eng) < 4000:  # make sure it is in unit of eV
                        self.xanes_eng = self.xanes_eng * 1000
                    st = '{0:3.1f}, {1:3.1f}, ..., {2:3.1f}  (totally, {3} energies)'.format(self.xanes_eng[0],
                                                                                             self.xanes_eng[1],
                                                                                             self.xanes_eng[-1],
                                                                                             len(self.xanes_eng))
                    self.tx_fit_pre_s.setText(str(min(self.xanes_eng)-1))
                    self.tx_fit_pre_e.setText(str(min(self.xanes_eng) +10))
                    self.tx_fit_post_e.setText(str(max(self.xanes_eng)+1))
                    self.tx_fit_post_s.setText(str(max(self.xanes_eng) -10))
                    self.lb_ang1.setText(st)

                    assert (len(self.xanes_eng) == self.img_xanes.shape[
                        0]), 'number of energy does not match number of images'

                except:
                    self.xanes_eng = np.array([])
                    self.lb_ang1.setText('No energy data ...')
                    self.lb_ang2.setVisible(True)
                    self.tx_ang.setVisible(True)
                    self.pb_ang.setVisible(True)
                f.close()

            else:  # read tiff file
                try:
                    self.img_xanes = np.array(io.imread(fn))
                    self.tx_sli_st.setText(str(self.img_xanes.shape[1] // 2))
                    self.tx_sli_end.setText(str(self.img_xanes.shape[1] // 2 + 1))
                    self.msg = 'image shape: {0}'.format(self.img_xanes.shape)
                    self.update_msg()
                    self.pb_rmbg.setEnabled(True)  # remove background
                    self.pb_rot_test.setEnabled(True)  # test rotation center
                    self.pb_del.setEnabled(True)  # delete single image
                    QApplication.processEvents()
                except:
                    self.img_xanes = np.zeros([2, 100, 100])
                    print('image not exist')
                finally:
                    self.img_bkg = np.ones(self.img_xanes.shape)  # (xanes - dark)/(bkg - dark)
                    self.img_bkg_removed = deepcopy(self.img_xanes)
                    self.img_align = deepcopy(self.img_bkg_removed)
                    self.img_dark = np.zeros(self.img_xanes.shape)
                    self.update_canvas_img()
                    self.theta = np.array([])
                    self.lb_ang1.setText('No energy data ...')
                    self.lb_ang2.setVisible(True)
                    self.tx_ang.setVisible(True)
                    self.pb_ang.setVisible(True)
        self.pb_ld.setEnabled(True)

    def update_msg(self):
        self.lb_msg.setText('Message: ' + self.msg)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')

    def manu_angle_input(self):
        com = 'self.theta = np.array(' + self.tx_ang.text() + ')'
        try:
            exec(com)
            st = str(self.theta[0]) + ',  ' + str(self.theta[1]) + ',  ...  , ' + str(self.theta[-1])
            st = '{0}  (totally: {1} angles)'.format(st, len(self.theta))
            self.lb_ang1.setText(st)
        except:
            self.lb_ang1.setText('Invalid angle')

    def default_layout(self):
        try:
            del self.img_xanes, self.img_bkg, self.img_bkg_removed, self.img_bkg_update
            del self.img_align, self.img_recon, self.img_rc_test, self.theta
        except:
            pass
        self.save_version = 0
        self.theta = np.array([])
        self.img_xanes = np.array([])
        self.img_bkg = np.array([])
        self.img_bkg_update = np.array([])
        self.img_bkg_removed = np.array([])
        self.img_align = np.array([])
        self.lb_ang1.setText('No energy data ...')
        self.lb_ang2.setVisible(False)
        self.tx_ang.setVisible(False)
        self.pb_ang.setVisible(False)

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
        QApplication.processEvents()
        if len(self.img_bkg.shape) == 3:
            img_bkg_update = np.sum(self.img_bkg, axis=0) / self.img_bkg.shape[0]
            img_dark_update = np.sum(self.img_dark, axis=0) / self.img_dark.shape[0]
        else:
            img_bkg_update = deepcopy(self.img_bkg)
            img_dark_update = deepcopy(self.img_dark)

        n = self.img_xanes.shape[0]
        img_bkg_ext = np.repeat(img_bkg_update[np.newaxis, :, :], n, axis=0)
        img_dark_ext = np.repeat(img_dark_update[np.newaxis, :, :], n, axis=0)

        temp = (self.img_xanes - img_dark_ext) / (img_bkg_ext - img_dark_ext)
        if self.rd_absp.isChecked() == True:  # Absorption image. e.g. TXM
            temp[temp < 0] = 1  # 1 means no absorption
            temp[np.isnan(temp)] = 1
            temp[np.isinf(temp)] = 1
            temp = -np.log(temp)
        else:  # Fluorescence image
            temp[temp < 0] = 0

        temp[np.isnan(temp)] = 0
        temp[np.isinf(temp)] = 0
        self.img_bkg_removed = temp
        self.img_align = deepcopy(self.img_bkg_removed)  # synchronize img_align and img_bkg_removed
        self.pb_rmbg.setText('Remove Bkg')
        self.pb_align.setEnabled(True)  # align image
        self.pb_norm1.setEnabled(True)  # norm background

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

    def xanes_align_img(self):
        self.pb_align.setText('Aligning ...')
        QApplication.processEvents()
        self.pb_align.setEnabled(False)
        prj = deepcopy(self.img_bkg_removed)
        n = self.img_align.shape[0]
        for i in range(1, n):
            print('Aligning image slice ' + str(i))
            img_temp, rsft, csft = self.align_img(prj[i - 1], prj[i])
            prj[i] = deepcopy(img_temp)

            self.lb_ali.setText('Aligning slice: ' + str(i))
            QApplication.processEvents()
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

    def update_canvas_img(self):
        canvas = self.canvas1
        slide = self.sl1
        type_index = self.cb1.currentText()
        #        self.pb_sh1.setEnabled(False)
        QApplication.processEvents()

        canvas.draw_line = False
        self.pb_adj_cmap.setEnabled(True)
        self.pb_set_cmap.setEnabled(True)
        self.pb_del.setEnabled(True)
        if type_index == 'Raw image':
            self.pb_roi_draw.setEnabled(True)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            sh = self.img_xanes.shape
            canvas.img_stack = self.img_xanes
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = self.img_xanes
        elif type_index == 'Background':
            self.pb_roi_draw.setEnabled(False)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            sh = self.img_bkg.shape
            canvas.img_stack = self.img_bkg
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = self.img_bkg
        elif type_index == 'Aligned image':
            self.pb_roi_draw.setEnabled(True)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            sh = self.img_align.shape
            canvas.img_stack = self.img_align
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = self.img_align
        elif type_index == 'Background removed':
            self.pb_roi_draw.setEnabled(True)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            sh = self.img_bkg_removed.shape
            canvas.img_stack = self.img_bkg_removed
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = self.img_bkg_removed
        elif type_index == 'Intensity plot':
            self.pb_roi_draw.setEnabled(False)
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            self.pb_adj_cmap.setEnabled(True)
            self.pb_set_cmap.setEnabled(True)
            self.pb_del.setEnabled(True)
            self.sl1.setMaximum(0)
            canvas.draw_line = True
            canvas.overlay_flag = False

            canvas.add_line()
            canvas.draw_line = False
            canvas.overlay_flag = True
            canvas.colorbar_on_flag = True
        elif type_index == 'Image smoothed':
            self.pb_roi_draw.setEnabled(False)
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            self.pb_adj_cmap.setEnabled(True)
            self.pb_set_cmap.setEnabled(True)
            self.pb_del.setEnabled(True)
            sh = self.img_smooth.shape
            canvas.img_stack = self.img_smooth
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = self.img_smooth


        # elif type_index == 'ROI spectrum':
        #     self.pb_roi_draw.setEnabled(False)
        #     canvas.axes.clear()  # this is important, to clear the current image before another imshow()
        #     self.pb_adj_cmap.setEnabled(False)
        #     self.pb_set_cmap.setEnabled(False)
        #     self.pb_del.setEnabled(False)
        #     self.sl1.setMaximum(0)
        #     canvas.x = self.xanes_eng
        #     canvas.y = self.roi_spec
        #     canvas.draw_line = True
        #     canvas.overlay_flag = False
        #     canvas.plot_label = 'roi1'
        #     canvas.legend_flag = True
        #     canvas.add_line()
        #     canvas.legend_flag = False
        #     canvas.draw_line = False
        #     canvas.overlay_flag = True
        #     canvas.colorbar_on_flag = True
        #     canvas.x, canvas.y = [], []

        QApplication.processEvents()

    def update_roi_list(self, mode='add', item_name=''):
        # de-select all the existing selection
        if mode == 'add':
            for i in range(self.lst_roi.count()):
                item = self.lst_roi.item(i)
                item.setSelected(False)

            item = QListWidgetItem(item_name)
            self.lst_roi.addItem(item)
            self.lst_roi.setCurrentItem(item)
        elif mode == 'del_all':
            self.lst_roi.clear()
        elif mode == 'del':
            for selectItem in self.lst_roi.selectedItems():
                self.lst_roi.removeItemWidget(selectItem)
        else:
            pass

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
    def __init__(self, parent=None, width=5, height=3, dpi=120, obj=[]):
        self.obj = obj
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        #        self.roi = xanes_roi(fig=self.fig, ax=self.axes)
        self.axes.axis('off')
        self.cmax = 1
        self.cmin = 0
        self.current_img = np.zeros([100, 100])
        self.current_img_index = 0
        self.colorbar_on_flag = True
        self.colormap = 'bone'
        self.draw_line = False
        self.overlay_flag = True
        self.x, self.y, = [], []
        self.plot_label = ''
        self.legend_flag = False
        self.roi_list = {}
        self.roi_color = {}
        self.roi_count = 0
        self.current_roi = [0, 0, 0, 0]
        self.color_list = ['red', 'brown', 'orange', 'olive', 'green', 'cyan', 'blue', 'pink', 'purple', 'gray']
        self.current_color = 'red'
        self.special_info = None
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)
        self.img_stack = np.zeros([1, 100, 100])
        self.mpl_connect('motion_notify_event', self.mouse_moved)

    def mouse_moved(self, mouse_event):
        if mouse_event.inaxes:
            x, y = mouse_event.xdata, mouse_event.ydata
            self.obj.lb_x_l.setText('x: {:3.2f}'.format(x))
            self.obj.lb_y_l.setText('y: {:3.2f}'.format(y))
            row = int(np.max([np.min([self.current_img.shape[0], y]), 0]))
            col = int(np.max([np.min([self.current_img.shape[1], x]), 0]))
            try:
                z = self.current_img[row][col]
                self.obj.lb_z_l.setText('intensity: {:3.3f}'.format(z))
            except:
                self.obj.lb_z_l.setText('')

    def update_img_stack(self):
        #        self.axes.hold(False)
        self.axes = self.fig.add_subplot(111)
        if self.img_stack.shape[0] == 0:
            img_blank = np.zeros([100, 100])
            return self.update_img_one(img_blank, img_index=self.current_img_index)
        return self.update_img_one(self.img_stack[0], img_index=0)

    def update_img_one(self, img=np.array([]), img_index=0):
        if len(img) == []: img = self.current_img
        self.current_img = img
        self.current_img_index = img_index
        self.im = self.axes.imshow(img, cmap=self.colormap, vmin=self.cmin, vmax=self.cmax)
        self.axes.axis('on')
        self.axes.set_aspect('equal', 'box')

        self.axes.set_title('current image: ' + str(img_index))
        self.axes.title.set_fontsize(10)
        if self.colorbar_on_flag:
            self.add_colorbar()
            self.colorbar_on_flag = False
        self.add_line()
        self.draw()

    def add_line(self):
        if self.draw_line:
            if self.overlay_flag:
                # self.axes.hold(True)
                self.axes.plot(self.x, self.y, '-', color=self.current_color, linewidth=1.0, label=self.plot_label)
                # self.axes.hold(False)
            else:
                self.rm_colorbar()
                line, = self.axes.plot(self.x, self.y, '.-', color=self.current_color, linewidth=1.0,
                                       label=self.plot_label)
                if self.legend_flag:
                    self.axes.legend(handles=[line])
                # self.axes.hold(True)
                self.axes.axis('on')
                self.axes.set_aspect('auto')
                self.draw()
        else:
            pass

    def draw_roi(self):
        self.cidpress = self.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        x1, y1 = event.xdata, event.ydata
        self.current_roi[0] = x1
        self.current_roi[1] = y1

    def on_release(self, event):
        x2, y2 = event.xdata, event.ydata
        self.current_roi[2] = x2
        self.current_roi[3] = y2
        self.roi_add_to_list()
        self.roi_display(self.current_roi)
        self.roi_disconnect()

    def roi_disconnect(self):
        self.mpl_disconnect(self.cidpress)
        self.mpl_disconnect(self.cidrelease)

    def roi_display(self, selected_roi):
        x1, y1 = selected_roi[0], selected_roi[1]
        x2, y2 = selected_roi[2], selected_roi[3]
        self.x = [x1, x2, x2, x1, x1]
        self.y = [y1, y1, y2, y2, y1]
        self.draw_line = True
        self.add_line()
        self.draw_line = False
        roi_name = '#' + str(self.roi_count - 1)
        self.axes.annotate(roi_name, xy=(x1, y1 - 40),
                           bbox={'facecolor': self.current_color, 'alpha': 0.5, 'pad': 2},
                           fontsize=10)
        self.draw()
        # self.roi_count += 1
        self.obj.tx_roi_x1.setText('{:4.1f}'.format(x1))
        self.obj.tx_roi_y1.setText('{:4.1f}'.format(y1))
        self.obj.tx_roi_x2.setText('{:4.1f}'.format(x2))
        self.obj.tx_roi_y2.setText('{:4.1f}'.format(y2))
        self.obj.pb_roi_draw.setEnabled(True)

        QApplication.processEvents()

    #
    def roi_add_to_list(self):
        roi_name = 'roi_' + str(self.roi_count)
        self.roi_list[roi_name] = deepcopy(self.current_roi)
        self.current_color = self.color_list[self.roi_count % 10]
        self.roi_color[roi_name] = self.current_color
        self.roi_count += 1
        self.obj.update_roi_list(mode='add', item_name=roi_name)

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
        except:
            pass

    def add_colorbar(self):
        if self.colorbar_on_flag:
            try:
                self.cb.remove()
                self.draw()
            except:
                pass
            self.divider = make_axes_locatable(self.axes)
            self.cax = self.divider.append_axes('right', size='3%', pad=0.06)
            self.cb = self.fig.colorbar(self.im, cax=self.cax, orientation='vertical')
            self.cb.ax.tick_params(labelsize=10)
            self.draw()

        # class xanes_roi():


#    def __init__(self, fig, ax):
#        self.fig = fig
#        self.ax = ax
#        self.roi_list = {}
#        self.roi_count = 0
#        self.current_roi = [0,0,0,0]
#        
#    def connect(self):
#        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
#        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
#    
#    def on_press(self, event):
#        x1, y1 = event.xdata, event.ydata
#        self.current_roi[0] = x1
#        self.current_roi[1] = y1
#        
#    def on_release(self, event):
#        x2, y2 = event.xdata, event.ydata
#        self.current_roi[2] = x2
#        self.current_roi[3] = y2
#        roi_name = 'roi_' + str(self.roi_count)
#        self.roi_list[roi_name] = self.current_roi
#        self.roi_count += 1
#
#        self.display()
#        self.disconnect()
#        
#    def display(self):
#        x1, y1, x2, y2 = self.current_roi
#        print('y2='+str(y2))
#        self.ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])
#
#        self.fig.canvas.draw()
##        plt.show()
#        
#    def disconnect(self):
#        self.fig.canvas.mpl_disconnect(self.cidpress)
#        self.fig.canvas.mpl_disconnect(self.cidrelease) 
#        self.ax.hold(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    xanes = App()
    xanes.show()
    sys.exit(app.exec_())
