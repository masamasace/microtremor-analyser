import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pathlib import Path
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix" 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import datetime

import pandas as pd
import numpy as np
from scipy.signal.windows import parzen

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, num_row=1, num_col=1, width=6, height=12, left=0.125, right=0.9, hspace=0.2, wspace=0.2):
        fig, self.axes = plt.subplots(num_row, num_col, figsize=(width, height))
        fig.subplots_adjust(left=left, right=right, hspace=hspace, wspace=wspace)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QWidget):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.list_file_path = []
        self.import_file_index = 0
        self.analysing_file_path = ""
        self.running_spectra = []
        self.running_spectra_arrows = []
        self.representative_spectrum = []
        self.representative_spectrum_geomean = []
        self.representative_spectrum_geomean_parzen = []


        self.section_width = 2048
        self.parzen_band_freqency_sectioning = 0.05
        self.parzen_band_freqency_smoothing = 0.2
 
        self.initUI()

    
    def initUI(self):
        
        self.grid_entire = QGridLayout()
        
        self.groupbox_file = QGroupBox()
        self.grid_file = QGridLayout()

        # Define select button
        self.button_file_dialog = QPushButton("Select", self)
        self.button_file_dialog.clicked.connect(self.open_dialog)
        self.grid_file.addWidget(self.button_file_dialog, 0, 0)

        # Define file path form
        self.lineedit_file_path = QLineEdit("file path", self)
        self.grid_file.addWidget(self.lineedit_file_path, 0, 1)

        # Define Register data button
        self.button_file_register = QPushButton("Register Data", self)
        self.button_file_register.clicked.connect(self.register_file)
        self.grid_file.addWidget(self.button_file_register, 0, 2)

        self.label01 = QLabel("Import File", self)
        self.label01.setAlignment(Qt.AlignCenter)
        self.grid_file.addWidget(self.label01, 1, 0)

        self.combobox_file_list = QComboBox(self)
        self.grid_file.addWidget(self.combobox_file_list, 1, 1)

        self.button_file_import = QPushButton("Import", self)
        self.button_file_import.clicked.connect(self.import_file)
        self.grid_file.addWidget(self.button_file_import, 1, 2)

        self.groupbox_file.setLayout(self.grid_file)
        self.groupbox_file.setTitle("File")

        self.groupbox_graph_control = QGroupBox()
        self.grid_graph_control = QGridLayout()

        self.groupbox_graph_time_data = QGroupBox()

        self.groupbox_graph_time_data.setTitle("Time-series record")
        self.grid_graph_time_data = QGridLayout()
        
        self.label02 = QLabel("Min", self)
        self.label03 = QLabel("Max", self)
        self.label04 = QLabel("Time ", self)
        self.label05 = QLabel("Y", self)
        self.label02.setAlignment(Qt.AlignCenter)
        self.label03.setAlignment(Qt.AlignCenter)
        self.label04.setAlignment(Qt.AlignCenter)
        self.label05.setAlignment(Qt.AlignCenter)

        self.lineedit_min_t_1 = QLineEdit(self)
        self.lineedit_max_t_1 = QLineEdit(self)
        self.lineedit_min_y_1 = QLineEdit(self)
        self.lineedit_max_y_1 = QLineEdit(self)

        self.button_update_graph_time_data = QPushButton("Update", self)
        self.button_update_graph_time_data.clicked.connect(lambda: self.update_graph_time_data("UpdateGraph"))

        self.button_reset_graph_time_data = QPushButton("Reset", self)
        self.button_reset_graph_time_data.clicked.connect(lambda: self.update_graph_time_data("Reset"))

        self.grid_graph_time_data.addWidget(self.label02, 1, 0, 1, 1)
        self.grid_graph_time_data.addWidget(self.label03, 2, 0, 1, 1)
        self.grid_graph_time_data.addWidget(self.label04, 0, 1, 1, 1)
        self.grid_graph_time_data.addWidget(self.label05, 0, 2, 1, 1)
        self.grid_graph_time_data.addWidget(self.lineedit_min_t_1, 1, 1, 1, 1)
        self.grid_graph_time_data.addWidget(self.lineedit_max_t_1, 2, 1, 1, 1)
        self.grid_graph_time_data.addWidget(self.lineedit_min_y_1, 1, 2, 1, 1)
        self.grid_graph_time_data.addWidget(self.lineedit_max_y_1, 2, 2, 1, 1)
        self.grid_graph_time_data.addWidget(self.button_update_graph_time_data, 3, 0, 1, 3)
        self.grid_graph_time_data.addWidget(self.button_reset_graph_time_data, 4, 0, 1, 3)


        self.groupbox_graph_time_data.setLayout(self.grid_graph_time_data)
        self.grid_graph_control.addWidget(self.groupbox_graph_time_data, 0, 0, 1, 1)

        self.groupbox_running_spectra = QGroupBox()
        self.groupbox_running_spectra.setTitle("Running Spectrum")
        self.grid_running_spectra = QGridLayout()

        self.label06 = QLabel("Min", self)
        self.label07 = QLabel("Max", self)
        self.label08 = QLabel("Time", self)
        self.label09 = QLabel("Freqency", self)
        self.label10 = QLabel("H/V Amp.", self)
        self.label11 = QLabel("Section Width", self)
        self.label12 = QLabel("Apply Parzen", self)
        self.label13 = QLabel("Parzen Band", self)

        self.label06.setAlignment(Qt.AlignCenter)
        self.label07.setAlignment(Qt.AlignCenter)
        self.label08.setAlignment(Qt.AlignCenter)
        self.label09.setAlignment(Qt.AlignCenter)
        self.label10.setAlignment(Qt.AlignCenter)
        self.label11.setAlignment(Qt.AlignCenter)
        self.label12.setAlignment(Qt.AlignCenter)
        self.label13.setAlignment(Qt.AlignCenter)

        self.lineedit_min_t_2    = QLineEdit(self)
        self.lineedit_max_t_2    = QLineEdit(self)
        self.lineedit_min_freq_2 = QLineEdit(self)
        self.lineedit_max_freq_2 = QLineEdit(self)
        self.lineedit_min_amp_2  = QLineEdit(self)
        self.lineedit_max_amp_2  = QLineEdit(self)
        self.lineedit_section_width_2          = QLineEdit(self)
        self.button_toggle_parzen_sectioning_2 = QPushButton("Sectioning", self)
        self.button_toggle_parzen_smoothing_2  = QPushButton("Smoothing", self)
        self.lineedit_parzen_band_freqency_sectioning_2 = QLineEdit(self)
        self.lineedit_parzen_band_freqency_smoothing_2  = QLineEdit(self)

        self.button_toggle_parzen_sectioning_2.setCheckable(True)
        self.button_toggle_parzen_smoothing_2.setCheckable(True)

        self.button_compute_running_spetrum = QPushButton("Compute Running Spectrum", self)
        self.button_compute_running_spetrum.clicked.connect(lambda: self.compute_update_running_spectra(only_update=False))

        self.button_graph_update_running_spetrum = QPushButton("Update Graph", self)
        self.button_graph_update_running_spetrum.clicked.connect(lambda: self.compute_update_running_spectra(only_update=True))

        self.grid_running_spectra.addWidget(self.label06, 1, 0, 1, 1)
        self.grid_running_spectra.addWidget(self.label07, 2, 0, 1, 1)
        self.grid_running_spectra.addWidget(self.label08, 0, 1, 1, 1)
        self.grid_running_spectra.addWidget(self.label09, 0, 2, 1, 1)
        self.grid_running_spectra.addWidget(self.label10, 0, 3, 1, 1)
        self.grid_running_spectra.addWidget(self.label11, 0, 4, 1, 1)
        self.grid_running_spectra.addWidget(self.label12, 0, 5, 1, 1)
        self.grid_running_spectra.addWidget(self.label13, 0, 6, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_min_t_2    , 1, 1, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_max_t_2    , 2, 1, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_min_freq_2 , 1, 2, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_max_freq_2 , 2, 2, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_min_amp_2  , 1, 3, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_max_amp_2  , 2, 3, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_section_width_2          , 1, 4, 1, 1)
        self.grid_running_spectra.addWidget(self.button_toggle_parzen_sectioning_2 , 1, 5, 1, 1)
        self.grid_running_spectra.addWidget(self.button_toggle_parzen_smoothing_2  , 2, 5, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_parzen_band_freqency_sectioning_2 , 1, 6, 1, 1)
        self.grid_running_spectra.addWidget(self.lineedit_parzen_band_freqency_smoothing_2  , 2, 6, 1, 1)

        self.grid_running_spectra.addWidget(self.button_compute_running_spetrum  , 3, 0, 1, 7)
        self.grid_running_spectra.addWidget(self.button_graph_update_running_spetrum  , 4, 0, 1, 7)

        self.groupbox_running_spectra.setLayout(self.grid_running_spectra)
        self.grid_graph_control.addWidget(self.groupbox_running_spectra, 0, 1, 1, 1)
        self.groupbox_graph_control.setLayout(self.grid_graph_control)

        self.groupbox_graph_control.setTitle("Graph Control")

        self.groupbox_export= QGroupBox()
        self.groupbox_export.setTitle("Export")
        self.grid_export = QGridLayout()

        self.button_export_data   = QPushButton("Export Data", self) 
        self.button_export_figure = QPushButton("Export Figure", self)
        self.button_export_data.clicked.connect(self.export_data)
        self.button_export_figure.clicked.connect(self.export_figure)

        self.grid_export.addWidget(self.button_export_data, 0, 0, 1, 1)
        self.grid_export.addWidget(self.button_export_figure, 1, 0, 1, 1)

        self.groupbox_export.setLayout(self.grid_export)
        self.grid_graph_control.addWidget(self.groupbox_export, 0, 2, 1, 1)

        self.grid_entire.addWidget(self.groupbox_file, 0, 0, 1, 1)
        self.grid_entire.addWidget(self.groupbox_graph_control, 1, 0, 1, 1)

        self.mpl_canvas_xyz = MplCanvas(self, num_row=3, num_col=1, hspace=0)
        self.t_x_plot, = self.mpl_canvas_xyz.axes[0].plot([0, 100], [0, 0], "k", linewidth=0.5)
        self.t_y_plot, = self.mpl_canvas_xyz.axes[1].plot([0, 100], [0, 0], "k", linewidth=0.5)
        self.t_z_plot, = self.mpl_canvas_xyz.axes[2].plot([0, 100], [0, 0], "k", linewidth=0.5)
        self.mpl_canvas_xyz.axes[0].tick_params(labelbottom=False)
        self.mpl_canvas_xyz.axes[1].tick_params(labelbottom=False)
        self.mpl_canvas_xyz.axes[0].set_title("Time-series Data")
        self.mpl_canvas_xyz.axes[0].text(0.825, 0.9, "X Component", transform=self.mpl_canvas_xyz.axes[0].transAxes, fontsize=8)
        self.mpl_canvas_xyz.axes[1].text(0.825, 0.9, "Y Component", transform=self.mpl_canvas_xyz.axes[1].transAxes, fontsize=8)
        self.mpl_canvas_xyz.axes[2].text(0.825, 0.9, "Z Component", transform=self.mpl_canvas_xyz.axes[2].transAxes, fontsize=8)
        self.mpl_canvas_xyz.axes[2].set_xlabel("Time (sec)")

        self.grid_entire.addWidget(self.mpl_canvas_xyz, 2, 0, 1, 1)

        # Draw figure of running spectrum
        self.mpl_canvas_running_spectra = MplCanvas(self, num_row=2, num_col=1)

        pos = self.mpl_canvas_running_spectra.axes[0].imshow([[0]], aspect=0.45)
        self.cbar = self.mpl_canvas_running_spectra.figure.colorbar(
            pos, 
            ax=self.mpl_canvas_running_spectra.axes[1], 
            orientation = 'horizontal',
            aspect=40)
        self.mpl_canvas_running_spectra.axes[1].get_xaxis().set_visible(False)
        self.mpl_canvas_running_spectra.axes[1].get_yaxis().set_visible(False)
        self.mpl_canvas_running_spectra.axes[1].set_frame_on(False)

        bbox_running_spectra = self.mpl_canvas_running_spectra.axes[0].get_position()
        corner_xyz = self.mpl_canvas_xyz.axes[0].get_position().get_points()
        temp_position = [[corner_xyz[0][0], 0.1], 
                         [corner_xyz[1][0], 1]]
        bbox_running_spectra.set_points(temp_position)
        self.mpl_canvas_running_spectra.axes[0].set_position(bbox_running_spectra, which="original")

        self.mpl_canvas_running_spectra.axes[0].set_title("Running Spectra", y=1.2)
        self.mpl_canvas_running_spectra.axes[0].set_xlabel("Time (sec)")
        self.mpl_canvas_running_spectra.axes[0].set_ylabel("Freqency (Hz)")
        
        self.mpl_canvas_running_spectra.mpl_connect("button_press_event", self.onclick_running_spectra)
        self.grid_entire.addWidget(self.mpl_canvas_running_spectra, 3, 0, 1, 1)

        # Draw figure of representative H/V spetra 
        self.mpl_canvas_representative_spectrum = MplCanvas(self, num_row=1, num_col=1)
        self.mpl_canvas_representative_spectrum.axes.plot([1, 10], [1, 1], label="Sample")
        self.mpl_canvas_representative_spectrum.axes.set_xscale("log")
        self.mpl_canvas_representative_spectrum.axes.set_yscale("log")
        self.mpl_canvas_representative_spectrum.axes.set_xlabel("Freqency (Hz)")
        self.mpl_canvas_representative_spectrum.axes.set_ylabel("H/V Ratio")
        self.mpl_canvas_representative_spectrum.axes.set_title("Representative Spectrum")
        self.mpl_canvas_representative_spectrum.axes.legend(ncols=3, loc="upper left", fontsize=8, fancybox=False, frameon=False)
        self.grid_entire.addWidget(self.mpl_canvas_representative_spectrum, 2, 1, 1, 1)

        # 
        self.mpl_canvas_representative_spectrum_parzen = MplCanvas(self, num_row=1, num_col=1)
        self.mpl_canvas_representative_spectrum_parzen.axes.plot([1, 10], [1, 1], label="Sample")
        self.mpl_canvas_representative_spectrum_parzen.axes.set_xscale("log")
        self.mpl_canvas_representative_spectrum_parzen.axes.set_yscale("log")
        self.mpl_canvas_representative_spectrum_parzen.axes.set_xlabel("Freqency (Hz)")
        self.mpl_canvas_representative_spectrum_parzen.axes.set_ylabel("H/V Ratio")
        self.mpl_canvas_representative_spectrum_parzen.axes.set_title("Representative Spectrum (Applying Parzen)")
        self.mpl_canvas_representative_spectrum_parzen.axes.legend(ncols=3, loc="upper left", fontsize=8, fancybox=False, frameon=False)
        self.grid_entire.addWidget(self.mpl_canvas_representative_spectrum_parzen, 3, 1, 1, 1)
        
        self.grid_entire.setColumnStretch(0, 1)
        self.grid_entire.setColumnStretch(1, 1)
        self.setLayout(self.grid_entire)

        self.setGeometry(300, 50, 1600, 1200)
        self.setWindowTitle('Microtremor Analyser')
        
    def open_dialog(self):
        opened_file_path, check = QFileDialog.getOpenFileName(None,"Select csv file","","All Files (*)")

        if check:
            self.lineedit_file_path.setText(str(opened_file_path))
    
    def register_file(self):
        temp = self.lineedit_file_path.text()

        if os.path.isfile(temp):
            self.list_file_path.append(Path(temp))
            self.update_combobox()
            print("Registered!")
    
    def update_combobox(self):
        if len(self.list_file_path):
            self.combobox_file_list.clear()
            for temp_file_path in self.list_file_path:
                self.combobox_file_list.addItem(temp_file_path.name)
    
    def import_file(self):
        temp = self.combobox_file_list.currentIndex()

        if temp < len(self.list_file_path):
            self.analysing_file_path = self.list_file_path[temp]
            print("Imported!")
            self.draw_canvas()
    
    def draw_canvas(self):
        
        if self.analysing_file_path.suffix == ".csv":
            self.acc_data = pd.read_csv(self.analysing_file_path)

            abs_max = self.acc_data.iloc[:, 1:4].max().max() * 1.1

            self.t_x_plot.set_xdata(self.acc_data["time"])
            self.t_x_plot.set_ydata(self.acc_data["x"])
            self.t_y_plot.set_xdata(self.acc_data["time"])
            self.t_y_plot.set_ydata(self.acc_data["y"])
            self.t_z_plot.set_xdata(self.acc_data["time"])
            self.t_z_plot.set_ydata(self.acc_data["z"])
            self.mpl_canvas_xyz.axes[0].relim()
            self.mpl_canvas_xyz.axes[1].relim()
            self.mpl_canvas_xyz.axes[2].relim()
            self.mpl_canvas_xyz.axes[0].autoscale_view()
            self.mpl_canvas_xyz.axes[1].autoscale_view()
            self.mpl_canvas_xyz.axes[2].autoscale_view()
            self.mpl_canvas_xyz.axes[0].set_ylim(-abs_max, abs_max)
            self.mpl_canvas_xyz.axes[1].set_ylim(-abs_max, abs_max)
            self.mpl_canvas_xyz.axes[2].set_ylim(-abs_max, abs_max)
            self.mpl_canvas_xyz.draw()

            self.time_interval = self.acc_data["time"].iloc[1] - self.acc_data["time"].iloc[0]

            self.update_graph_time_data("UpdateTable")
            self.initialize_runnning_spectrum_control()

        else:
            print("Imported file should be csv file")
    
    def initialize_runnning_spectrum_control(self):
        xlim = self.mpl_canvas_xyz.axes[0].get_xlim()
        ylim = self.mpl_canvas_xyz.axes[0].get_ylim()

        self.fft_freq = np.fft.fftfreq(self.section_width, self.acc_data.iloc[1, 0]- self.acc_data.iloc[0, 0])
        fft_freq_only_postive = self.fft_freq[np.where(self.fft_freq >= 0)]

        self.lineedit_min_t_2.setText("{:.3g}".format(xlim[0]))   
        self.lineedit_max_t_2.setText("{:.3g}".format(xlim[1]))  
        self.lineedit_min_freq_2.setText("{:.3g}".format(fft_freq_only_postive.min()))
        self.lineedit_max_freq_2.setText("{:.3g}".format(fft_freq_only_postive.max()))
        self.lineedit_min_amp_2.setText("-1.0")  
        self.lineedit_max_amp_2.setText("2.0")   
        self.lineedit_section_width_2.setText(str(self.section_width))
        self.lineedit_parzen_band_freqency_sectioning_2.setText(str(self.parzen_band_freqency_sectioning))
        self.lineedit_parzen_band_freqency_smoothing_2.setText(str(self.parzen_band_freqency_smoothing))


    def clear_canvas(self):
        self.mpl_canvas_xyz.axes[0].cla()
        self.mpl_canvas_xyz.axes[1].cla()
        self.mpl_canvas_xyz.axes[2].cla()
        self.mpl_canvas_running_spectra.axes[0].cla()

    # Compute running spectrum
    # コメントアウトしている行はParzenフィルタの適用を実装しようとしたもの。計算量が多すぎるために断念
    def compute_update_running_spectra(self, only_update):
        
        result = self.running_spectra_param_check()

        if result[0]:

            time_data = self.acc_data["time"]

            self.fft_freq = np.fft.fftfreq(self.section_width, self.acc_data.iloc[1, 0]- self.acc_data.iloc[0, 0])
            fft_freq_len = len(self.fft_freq)
            
            if not(only_update):


                # max_fft_index = (fft_freq_len + 1) // 2 - 1
                self.running_spectra = np.zeros((len(time_data), fft_freq_len))
                self.running_spectra_parzen = np.zeros((len(time_data), fft_freq_len))

                if self.button_toggle_parzen_sectioning_2.isChecked():
                    parzen_window_sectioning = parzen(self.section_width)
                else:
                    parzen_window_sectioning = 1
                
                # freq_interval = self.fft_freq[1] - self.fft_freq[0]
                # parzen_window_smoothing_weight = self.calcurate_parzen_window_smoothing_weight(freq_interval, self.parzen_band_freqency_smoothing)
                # parzen_window_smoothing_weight_halflength = len(parzen_window_smoothing_weight) // 2 

                max_index = len(time_data) - self.section_width + 1
                progress_percent = 0

                print("Computing Running Spectrum...0...", end="")
                for i in range(max_index):
                    x_data_clipped = self.acc_data.iloc[i:i+self.section_width, 1] * parzen_window_sectioning
                    y_data_clipped = self.acc_data.iloc[i:i+self.section_width, 2] * parzen_window_sectioning
                    z_data_clipped = self.acc_data.iloc[i:i+self.section_width, 3] * parzen_window_sectioning

                    x_data_clipped_fft = np.abs(np.fft.fft(x_data_clipped))
                    y_data_clipped_fft = np.abs(np.fft.fft(y_data_clipped))
                    z_data_clipped_fft = np.abs(np.fft.fft(z_data_clipped))
                    
                    # in order to avoid zero-division
                    z_data_clipped_fft = np.where(z_data_clipped_fft == 0, 0.00000001, z_data_clipped_fft)
                    h_data_clipped_fft = np.sqrt(x_data_clipped_fft ** 2 + y_data_clipped_fft ** 2)


                    hol_var_ratio = h_data_clipped_fft / z_data_clipped_fft
                    # in order to avoid zero-division
                    hol_var_ratio = np.where(hol_var_ratio == 0, 0.00000001, hol_var_ratio)

                    self.running_spectra[i, :] = hol_var_ratio
                    
                    # apply parzen window to smooth H/V spectra
                    # 周波数0の成分に対応するスペクトルの値は補正しない、0以外の成分に対応するスペクトルは正の成分のみ補正
                    # 境界付近では打ち切りを行う。隣り合う5つの振幅への補正係数が[0.1, 0.2, 0.4, 0.2, 0.1]で、最初の0.1が境界よりも外にある場合、
                    # 最初の0.1以外の総和(=0.9)で最初の0.1以外の項を割った配列を補正係数配列とする
                    # 以下のコードは計算量が多いため廃止

                    # tile_hol_var_ratio = np.tile(hol_var_ratio, (fft_freq_len, 1))
                    # weight_parzen = np.zeros_like(tile_hol_var_ratio)

                    # print(tile_hol_var_ratio.shape, weight_parzen.shape)

                    # for j in range(max_fft_index):
                    #     print(j)
                    #     if j == 0:
                    #         weight_parzen[j][0] = 1
                    #     elif j < parzen_window_smoothing_weight_halflength + 1:
                    #         min_index = parzen_window_smoothing_weight_halflength - (j - 1)
                    #         temp_weight = parzen_window_smoothing_weight[min_index:] / parzen_window_smoothing_weight[min_index:].sum()
                    #         print(temp_weight)
                    #         weight_parzen[j][1:j+parzen_window_smoothing_weight_halflength + 1] = temp_weight

                    #     elif j > max_fft_index - parzen_window_smoothing_weight_halflength:
                    #         max_index = parzen_window_smoothing_weight_halflength + max_fft_index - j
                    #         temp_weight = parzen_window_smoothing_weight[:max_index] / parzen_window_smoothing_weight[:max_index].sum()
                    #         weight_parzen[j][j-parzen_window_smoothing_weight_halflength:max_fft_index] = temp_weight
                    #     else:
                    #         weight_parzen[j][j-parzen_window_smoothing_weight_halflength:j+parzen_window_smoothing_weight_halflength+1] = parzen_window_smoothing_weight

                    # tile_hol_var_ratio *= weight_parzen
                    # self.running_spectra_parzen[i, :] = tile_hol_var_ratio.sum(axis=0)

                    temp_percent = (i / max_index) // 0.1 * 10
                    if temp_percent > progress_percent:
                        print(str(int(temp_percent))+"...", end="")
                        progress_percent = temp_percent
                    
                print("Finished!")

        
            tlim, freqlim, amplim = result[1]


            tlim_index = [(time_data >= tlim[0]).idxmax(), (time_data <= tlim[1]).iloc[::-1].idxmax()]
            freq_index = [np.argwhere((self.fft_freq >= freqlim[0]) & (self.fft_freq <= freqlim[1]))[0][0], 
                          np.argwhere((self.fft_freq >= freqlim[0]) & (self.fft_freq <= freqlim[1]))[-1][0]]
            self.running_spectra_display = self.running_spectra[tlim_index[0]:tlim_index[1]+1, freq_index[0]:freq_index[1]+1]

            # Clear running spectrum figure
            self.mpl_canvas_running_spectra.axes[0].cla()
            self.mpl_canvas_running_spectra.axes[1].cla()
            self.mpl_canvas_representative_spectrum.axes.cla()


            temp_aspect_ratio = self.running_spectra_display.shape[0] / self.running_spectra_display.shape[1]

            pos = self.mpl_canvas_running_spectra.axes[0].imshow(
                np.log(self.running_spectra_display.T), 
                vmin = amplim[0],
                vmax = amplim[1],
                aspect = temp_aspect_ratio * 0.45,
                interpolation = 'bilinear'
                )
            
            self.mpl_canvas_running_spectra.axes[0].autoscale(False)
            xmin = tlim[0] / (self.acc_data.iloc[1, 0]- self.acc_data.iloc[0, 0])
            xmax = tlim[1] / (self.acc_data.iloc[1, 0]- self.acc_data.iloc[0, 0])
            self.mpl_canvas_running_spectra.axes[0].set_xlim(xmin=xmin, xmax=xmax)
            self.cbar.update_normal(pos)
            self.mpl_canvas_running_spectra.axes[1].get_xaxis().set_visible(False)
            self.mpl_canvas_running_spectra.axes[1].get_yaxis().set_visible(False)
            self.mpl_canvas_running_spectra.axes[1].set_frame_on(False)

            bbox_running_spectra = self.mpl_canvas_running_spectra.axes[0].get_position()
            corner_xyz = self.mpl_canvas_xyz.axes[0].get_position().get_points()
            temp_position = [[corner_xyz[0][0], 0.1], 
                            [corner_xyz[1][0], 1]]
            bbox_running_spectra.set_points(temp_position)
            self.mpl_canvas_running_spectra.axes[0].set_position(bbox_running_spectra, which="original")

            self.mpl_canvas_running_spectra.axes[0].set_title("Running Spectra", y=1.2)
            self.mpl_canvas_running_spectra.axes[0].set_xlabel("Time (sec)")
            self.mpl_canvas_running_spectra.axes[0].set_ylabel("Freqency (Hz)")

            # Change Tick Label
            xtick_label_location = self.mpl_canvas_running_spectra.axes[0].get_xticks()
            ytick_label_location = self.mpl_canvas_running_spectra.axes[0].get_yticks()

            xtick_label_location = xtick_label_location[(xtick_label_location >= 0) & (xtick_label_location < self.running_spectra_display.shape[0])]
            ytick_label_location = ytick_label_location[(ytick_label_location >= 0) & (ytick_label_location < self.running_spectra_display.shape[1])]

            xtick_label_index = xtick_label_location.astype(int)
            ytick_label_index = ytick_label_location.astype(int)

            xtick_label_new = time_data.iloc[xtick_label_index].to_numpy()
            ytick_label_new = self.fft_freq[freq_index[0]:freq_index[1]+1][ytick_label_index]

            xtick_label_new = ["{:.1f}".format(temp) for temp in xtick_label_new]
            ytick_label_new = ["{:.1f}".format(temp) for temp in ytick_label_new]

            self.mpl_canvas_running_spectra.axes[0].set_xticks(xtick_label_location, xtick_label_new)
            self.mpl_canvas_running_spectra.axes[0].set_yticks(ytick_label_location, ytick_label_new)

            self.mpl_canvas_representative_spectrum.axes.set_xscale("log")
            self.mpl_canvas_representative_spectrum.axes.set_yscale("log")
            self.mpl_canvas_representative_spectrum.axes.set_xlabel("Freqency (Hz)")
            self.mpl_canvas_representative_spectrum.axes.set_ylabel("H/V Ratio")
            self.mpl_canvas_representative_spectrum.axes.set_title("Representative Spectrum")
            self.mpl_canvas_representative_spectrum.axes.legend(ncols=3, loc="upper left", fontsize=8, fancybox=False, frameon=False)

            self.mpl_canvas_running_spectra.draw()
            self.mpl_canvas_representative_spectrum.draw()

            self.running_spectra_arrows = []
            self.representative_spectrum = []


    def calcurate_parzen_window_smoothing_weight(self, interval, band):
        u = 280 / 151 / band
        temp = np.arange(stop=2/u, step=interval)[1:]
        weight = 3 / 4 * u * (np.sin(np.pi * u * temp / 2) / (np.pi * u * temp / 2)) ** 4
        weight = np.hstack([weight[::-1], 3 / 4 * u * np.ones(1, float), weight])
        weight /= weight.sum()
        return weight
        


    def running_spectra_param_check(self):
        try:
            tlim = [float(self.lineedit_min_t_2.text()), float(self.lineedit_max_t_2.text())]
            freqlim = [float(self.lineedit_min_freq_2.text()), float(self.lineedit_max_freq_2.text())]
            amplim = [float(self.lineedit_min_amp_2.text()), float(self.lineedit_max_amp_2.text())]
            self.section_width = int(self.lineedit_section_width_2.text())
            self.parzen_band_freqency_sectioning = float(self.lineedit_parzen_band_freqency_sectioning_2.text())
            self.parzen_band_freqency_smoothing = float(self.lineedit_parzen_band_freqency_smoothing_2.text())

        except ValueError:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Input valuse should be written in Numeric Characters')
            return (False,())

        else:
            return (True, (tlim, freqlim, amplim))



    def onclick_running_spectra(self, event):

        if event.button == 1:

            ylim = self.mpl_canvas_running_spectra.axes[0].get_ylim()
            x_data_real_scale = int(event.xdata) * self.time_interval 
            temp_index = len(self.running_spectra_arrows)
            label = str(temp_index + 1) + "_" + "{:.2f}".format(x_data_real_scale)
            annotated_representative_location = self.mpl_canvas_running_spectra.axes[0].annotate(
                label,
                xy=(event.xdata, ylim[1]),
                xytext=(event.xdata, -ylim[0]*0.075),
                arrowprops=dict(headwidth=6, headlength=6, facecolor="black"),
                fontsize=7)
            annotated_representative_location.set_rotation(90)
            self.running_spectra_arrows.append([annotated_representative_location])
            
            positive_fft_freq_index = self.fft_freq > 0
            x_data = int(event.xdata)
            self.temp_x = self.fft_freq[positive_fft_freq_index]
            temp_y = self.running_spectra[x_data][positive_fft_freq_index]
            self.representative_spectrum.append(self.mpl_canvas_representative_spectrum.axes.plot(self.temp_x, temp_y, linewidth=0.75, label=label))
        
        elif event.button == 3 and len(self.running_spectra_arrows):
            self.running_spectra_arrows.pop(-1)[0].remove()
            self.representative_spectrum.pop(-1)[0].remove()

        self.update_representative_spectrum_geomean()

        self.mpl_canvas_representative_spectrum.axes.legend(ncols=3, loc="upper left", fontsize=8, fancybox=False, frameon=False)
        self.mpl_canvas_running_spectra.draw()
        self.mpl_canvas_representative_spectrum.draw()
    
    def update_representative_spectrum_geomean(self):

        if len(self.representative_spectrum) != 0:

            num_representative_spectrum = len(self.representative_spectrum)
            temp_y_mean = np.ones_like(self.fft_freq[self.fft_freq > 0])
            for i in range(num_representative_spectrum):
                temp_y_mean *= self.representative_spectrum[i][0].get_ydata()
            temp_y_mean = temp_y_mean ** (1/num_representative_spectrum)
            temp_y_mean /= 10

            if len(self.representative_spectrum_geomean) == 0:
                self.representative_spectrum_geomean.append(
                    self.mpl_canvas_representative_spectrum.axes.plot(self.temp_x, 
                                                                      temp_y_mean,
                                                                      color="k", 
                                                                      linewidth=2, 
                                                                      label="Geo Mean"))
            else:
                self.representative_spectrum_geomean[0][0].set_ydata(temp_y_mean)
            
        else:
            self.representative_spectrum_geomean[0][0].set_alpha(0)

    
    def update_graph_time_data(self, event):

        if event == "UpdateTable":
            xlim = self.mpl_canvas_xyz.axes[0].get_xlim()
            ylim = self.mpl_canvas_xyz.axes[0].get_ylim()
            self.lineedit_min_t_1.setText("{:.3g}".format(xlim[0]))
            self.lineedit_max_t_1.setText("{:.3g}".format(xlim[1]))
            self.lineedit_min_y_1.setText("{:.3g}".format(ylim[0]))
            self.lineedit_max_y_1.setText("{:.3g}".format(ylim[1]))

        elif event == "UpdateGraph":
            try:
                xlim = [float(self.lineedit_min_t_1.text()), float(self.lineedit_max_t_1.text())]
                ylim = [float(self.lineedit_min_y_1.text()), float(self.lineedit_max_y_1.text())]
            except ValueError:
                pass
            else:
                self.mpl_canvas_xyz.axes[0].set_xlim(xlim)
                self.mpl_canvas_xyz.axes[1].set_xlim(xlim)
                self.mpl_canvas_xyz.axes[2].set_xlim(xlim)
                self.mpl_canvas_xyz.axes[0].set_ylim(ylim)
                self.mpl_canvas_xyz.axes[1].set_ylim(ylim)
                self.mpl_canvas_xyz.axes[2].set_ylim(ylim)
            
                self.mpl_canvas_xyz.draw()
        
        elif event== "Reset":
            abs_max = self.acc_data.iloc[:, 1:4].max().max() * 1.1

            self.mpl_canvas_xyz.axes[0].relim()
            self.mpl_canvas_xyz.axes[1].relim()
            self.mpl_canvas_xyz.axes[2].relim()
            self.mpl_canvas_xyz.axes[0].autoscale()
            self.mpl_canvas_xyz.axes[1].autoscale()
            self.mpl_canvas_xyz.axes[2].autoscale()
            self.mpl_canvas_xyz.axes[0].set_ylim(-abs_max, abs_max)
            self.mpl_canvas_xyz.axes[1].set_ylim(-abs_max, abs_max)
            self.mpl_canvas_xyz.axes[2].set_ylim(-abs_max, abs_max)
            self.mpl_canvas_xyz.draw()

            self.update_graph_time_data("UpdateTable")

    
    def export_data(self):
        exporting_data = pd.DataFrame({"freqency(Hz)": self.temp_x})
        if len(self.representative_spectrum):
            for line2D in self.representative_spectrum:
                label = line2D[0].get_label()
                ydata = line2D[0].get_ydata()
                exporting_data[label] = ydata
        
        if not os.path.exists("result"):
            os.mkdir("result")
        
        csv_file_name = "result/" + self.analysing_file_path.stem + "_HV_Spectrum_" + datetime.datetime.now().time().strftime("%H%M%S") + ".csv"
        exporting_data.to_csv(csv_file_name, index=False)

    def export_figure(self):
        if not os.path.exists("result"):
            os.mkdir("result")
        
        xyz_file_name = "result/" + self.analysing_file_path.stem + "_xyz_" + datetime.datetime.now().time().strftime("%H%M%S") + ".png"
        running_spectra_file_name = "result/" + self.analysing_file_path.stem + "_running_spectra_" + datetime.datetime.now().time().strftime("%H%M%S") + ".png"
        representative_spectrum_file_name = "result/" + self.analysing_file_path.stem + "_representative_spectrum_" + datetime.datetime.now().time().strftime("%H%M%S") + ".png"

        self.mpl_canvas_xyz.figure.savefig(xyz_file_name, dpi=1200, bbox_inches="tight")
        self.mpl_canvas_running_spectra.figure.savefig(running_spectra_file_name, dpi=1200, bbox_inches="tight")
        self.mpl_canvas_representative_spectrum.figure.savefig(representative_spectrum_file_name, dpi=1200, bbox_inches="tight")

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())