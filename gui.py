#! python3
import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QComboBox,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QCheckBox,
    QHBoxLayout,
    QFileDialog,
    QListWidget,
    QAbstractItemView,
    QFormLayout,
    QDoubleSpinBox,
    QMessageBox,
    QFrame,
    QPlainTextEdit,
    QTabWidget,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

matplotlib.use("QtAgg")

plt.rcParams.update({"font.size": 3})
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

class tabby(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.label = QPushButton("test string")
        self.layout.addWidget(self.label)

    def zoinc(self):
        self.layout.addItem(QPushButton("I want to be last!"))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_data()
        self.p = None
        # for resizeEvent to trigger after the plot objects are created
        self.plot_ready = False

        self.main_layout = QHBoxLayout()
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        self.setWindowTitle("NV dynamics")
        self.showMaximized()

        # self.horizontal_line = QFrame()
        # self.horizontal_line.setFrameShape(QFrame.Shape.HLine)
        # self.horizontal_line.setFrameShadow(QFrame.Shadow.Raised)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.tab_widget = QTabWidget()
        # self.tabtabby = tabby()
        # self.tabtabby.layout.addWidget(QPushButton("hello"))
        # self.tab_widget.addTab(self.tabtabby ,"testy")
        # self.tabtabby.zoinc()

        self.tab_theta = QWidget()
        self.tab_theta_layout = QVBoxLayout()
        self.tab_theta.setLayout(self.tab_theta_layout)
        # self.tab_theta_layout = QVBoxLayout(self.tab_theta)

        # self.tab_13c = QWidget()
        # self.tab_13c_layout = QVBoxLayout(self.tab_13c)

        self.tab_N14 = QWidget()
        self.tab_N14_layout = QVBoxLayout(self.tab_N14)
        
        self.tab_sim = QWidget()
        self.tab_sim_layout = QVBoxLayout()
        self.tab_sim.setLayout(self.tab_sim_layout)

        self.set_tab_theta()
        self.set_tab_N14()
        # self.set_tab_13c()
        self.set_tab_sim()

        self.tab_widget.addTab(self.tab_theta, "N15")
        self.tab_widget.addTab(self.tab_N14, "N14")
        # self.tab_widget.addTab(self.tab_13c, "13C")
        self.tab_widget.addTab(self.tab_sim, "sim")


        self.left_layout.addWidget(self.tab_widget)
        self.set_right_layout()

        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

    def set_tab_theta(self):
        self.ms_label_t = QLabel("m<sub>s</sub> state:")
        self.ms_widget_t = QComboBox()
        self.ms_widget_t.addItems(["-1", "1"])
        self.ms_label_t.setToolTip(
            "Electronic spin state. For magnetic fields near the level anti-crossing (532 G), the state ms=-1 is "
            "close to ms=0 and the system might not present well defined spin manipulations."
        )

        # XY8-N order
        self.N_label_t = QLabel("N:")
        self.N_widget_t = QComboBox()
        self.N_widget_t.addItems([str(val) for val in self.N])
        self.N_label_t.setToolTip(
            "Order of the XY8-N sequence. For high order, the exact duration of the pi pulses becomes important "
            "factor. in this simulations we used pi pulses of around 15 ns."
        )

        # External magnetic field intensity
        self.B0_label_t = QLabel("B\N{SUBSCRIPT ZERO} (G):")
        self.B0_widget_t = QComboBox()
        self.B0_widget_t.setMaxVisibleItems(20)
        self.B0_widget_t.addItems([str(val) for val in self.B0])
        self.B0_label_t.setToolTip(
            "Intensity of the external magnetic field. At low magnetic fields, both electronic spin states are nearly "
            "degenerate and the system presents a complex S=1 dynamics, which may lead to not well defined pulses."
        )

        # Misalignment angle
        self.theta_label_t = QLabel("θ (°):")
        self.theta_widget_t = QComboBox()
        self.theta_widget_t.addItems([str(val) for val in self.theta])
        self.theta_label_t.setToolTip(
            "Angle between the external magnetic field and the quantization axis of the electron spin. At high angles "
            "and field, the nuclear spin is not good quantum number and the Hamiltonian might not represent "
            "faithfully the system."
        )

        left_form_layout = QFormLayout()
        left_form_layout.addRow(self.ms_label_t, self.ms_widget_t)
        left_form_layout.addRow(self.N_label_t, self.N_widget_t)
        left_form_layout.addRow(self.B0_label_t, self.B0_widget_t)
        left_form_layout.addRow(self.theta_label_t, self.theta_widget_t)

        self.tab_theta_layout.addLayout(left_form_layout)
        # self.left_layout.addWidget(self.horizontal_line)

        # Optional experimental data file to compare with
        self.expt_file_label_t = QLabel("No file selected")

        self.expt_file_btn_t = QPushButton("Select File")
        self.expt_file_btn_t.clicked.connect(self.select_file_t)

        expt_HLay = QHBoxLayout()
        self.expt_label_t = QLabel("Compare with Experimental Data")
        self.expt_chkbox_t = QCheckBox()
        expt_HLay.addWidget(self.expt_label_t)
        expt_HLay.addWidget(self.expt_chkbox_t)

        self.expt_label_t.setToolTip(
            "Experimental data must be in a file with two columns: first column is the pulse separation tau in μs and "
            "the second column is the transition probability."
        )
        self.expt_filename_t = ""  # to handle plotting when no file is selected

        self.tab_theta_layout.addWidget(self.expt_file_label_t)
        self.tab_theta_layout.addWidget(self.expt_file_btn_t)

        self.tab_theta_layout.addLayout(expt_HLay)

        self.gyromagnetic_label = QLabel("Compare peaks with:")
        self.gyromagnetic_widget = QListWidget()
        self.gyromagnetic_widget.addItems(
            [key for key in self.gyromagnetic_ratios.keys()]
        )
        self.gyromagnetic_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.gyromagnetic_label.setToolTip(
            "The corresponding tau is 1/(2*γ*B<sub>0</sub>)."
        )

        self.tab_theta_layout.addWidget(self.gyromagnetic_label)
        self.tab_theta_layout.addWidget(self.gyromagnetic_widget)

        self.select_all_button_t = QPushButton("Select all")
        self.clear_button_t = QPushButton("Clear selection")

        self.select_all_button_t.clicked.connect(self.select_all_items_t)
        self.clear_button_t.clicked.connect(self.clear_selection_t)

        gyromagnetic_btn_lay = QFormLayout()
        gyromagnetic_btn_lay.addRow(
            self.select_all_button_t, self.clear_button_t)
        self.tab_theta_layout.addLayout(gyromagnetic_btn_lay)

        # tau range
        self.tau_layout_t = QFormLayout()

        self.tau_min_label_t = QLabel("min τ:")
        self.tau_min_widget_t = QDoubleSpinBox()
        self.tau_min_widget_t.setRange(0.05, 3.0)
        self.tau_min_widget_t.setSingleStep(0.1)
        self.tau_min_widget_t.setValue(0.05)

        self.tau_max_label_t = QLabel("max τ:")
        self.tau_max_widget_t = QDoubleSpinBox()
        self.tau_max_widget_t.setRange(0.05, 3.0)
        self.tau_max_widget_t.setSingleStep(0.1)
        self.tau_max_widget_t.setValue(3)
        self.tau_layout_t.addRow(self.tau_min_label_t, self.tau_min_widget_t)
        self.tau_layout_t.addRow(self.tau_max_label_t, self.tau_max_widget_t)

        self.tab_theta_layout.addLayout(self.tau_layout_t)

        # update button
        self.update_button_t = QPushButton("Update Plot")
        self.update_button_t.clicked.connect(self.update_plot_t)
        self.tab_theta_layout.addWidget(self.update_button_t)
        # self.testwid = tabby()
        # self.tab_theta_layout.addLayout(self.testwid)

    def set_tab_13c(self):
        self.ms_label_c = QLabel("m<sub>s</sub> state:")
        self.ms_widget_c = QComboBox()
        self.ms_widget_c.addItems(["-1", "1"])
        self.ms_label_c.setToolTip(
            "Electronic spin state. For magnetic fields near the level anti-crossing (532 G), the state ms=-1 is "
            "close to ms=0 and the system might not present well defined spin manipulations."
        )

        # XY8-N order
        self.N_label_c = QLabel("N:")
        self.N_widget_c = QComboBox()
        self.N_widget_c.addItems([str(val) for val in self.N])
        self.N_label_c.setToolTip(
            "Order of the XY8-N sequence. For high order, the exact duration of the pi pulses becomes important "
            "factor. in this simulations we used pi pulses of around 15 ns."
        )

        # External magnetic field intensity
        self.B0_label_c = QLabel("B\N{SUBSCRIPT ZERO} (G):")
        self.B0_widget_c = QComboBox()
        self.B0_widget_c.setMaxVisibleItems(20)
        self.B0_widget_c.addItems([str(val) for val in self.B0])
        self.B0_label_c.setToolTip(
            "Intensity of the external magnetic field. At low magnetic fields, both electronic spin states are nearly "
            "degenerate and the system presents a complex S=1 dynamics, which may lead to not well defined pulses."
        )

        left_form_layout = QFormLayout()
        left_form_layout.addRow(self.ms_label_c, self.ms_widget_c)
        left_form_layout.addRow(self.N_label_c, self.N_widget_c)
        left_form_layout.addRow(self.B0_label_c, self.B0_widget_c)

        self.tab_13c_layout.addLayout(left_form_layout)
        # self.left_layout.addWidget(self.horizontal_line)

        # Optional experimental data file to compare with
        self.expt_file_label_c = QLabel("No file selected")

        self.expt_file_btn_c = QPushButton("Select File")
        self.expt_file_btn_c.clicked.connect(self.select_file_c)

        expt_HLay = QHBoxLayout()
        self.expt_label_c = QLabel("Compare with Experimental Data")
        self.expt_chkbox_c = QCheckBox()
        expt_HLay.addWidget(self.expt_label_c)
        expt_HLay.addWidget(self.expt_chkbox_c)

        self.expt_label_c.setToolTip(
            "Experimental data must be in a file with two columns: first column is the pulse separation tau in μs and "
            "the second column is the transition probability."
        )
        self.expt_filename_c = ""  # to handle plotting when no file is selected

        self.tab_13c_layout.addWidget(self.expt_file_label_c)
        self.tab_13c_layout.addWidget(self.expt_file_btn_c)

        self.tab_13c_layout.addLayout(expt_HLay)

        self.families_label = QLabel("13C Families:")
        self.families_widget = QListWidget()
        self.families_widget.addItems([fam_value for fam_value in self.fam])
        self.families_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.families_label.setToolTip(
            "The corresponding tau is 1/(2*γ*B<sub>0</sub>)."
        )

        self.tab_13c_layout.addWidget(self.families_label)
        self.tab_13c_layout.addWidget(self.families_widget)

        self.select_all_button_c = QPushButton("Select all")
        self.clear_button_c = QPushButton("Clear selection")

        self.select_all_button_c.clicked.connect(self.select_all_items_c)
        self.clear_button_c.clicked.connect(self.clear_selection_c)

        fam_btn_layout = QFormLayout()
        fam_btn_layout.addRow(self.select_all_button_c, self.clear_button_c)
        self.tab_13c_layout.addLayout(fam_btn_layout)

        self.gyromagnetic_label_c = QLabel("Compare peaks with:")
        self.gyromagnetic_widget_c = QListWidget()
        self.gyromagnetic_widget_c.addItems(
            [key for key in self.gyromagnetic_ratios.keys()]
        )
        self.gyromagnetic_widget_c.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.gyromagnetic_label_c.setToolTip(
            "The corresponding tau is 1/(2*γ*B<sub>0</sub>)."
        )

        self.tab_13c_layout.addWidget(self.gyromagnetic_label_c)
        self.tab_13c_layout.addWidget(self.gyromagnetic_widget_c)

        self.select_all_btn_gyro_c = QPushButton("Select all")
        self.clear_btn_gyro_c = QPushButton("Clear selection")

        self.select_all_btn_gyro_c.clicked.connect(self.select_all_gyro_c)
        self.clear_btn_gyro_c.clicked.connect(self.clear_gyro_c)

        gyro_btn_layout = QFormLayout()
        gyro_btn_layout.addRow(
            self.select_all_btn_gyro_c, self.clear_btn_gyro_c)
        self.tab_13c_layout.addLayout(gyro_btn_layout)

        # tau range
        self.tau_layout_c = QFormLayout()

        self.tau_min_label_c = QLabel("min τ:")
        self.tau_min_widget_c = QDoubleSpinBox()
        self.tau_min_widget_c.setRange(0.05, 3.0)
        self.tau_min_widget_c.setSingleStep(0.1)
        self.tau_min_widget_c.setValue(0.05)

        self.tau_max_label_c = QLabel("max τ:")
        self.tau_max_widget_c = QDoubleSpinBox()
        self.tau_max_widget_c.setRange(0.05, 3.0)
        self.tau_max_widget_c.setSingleStep(0.1)
        self.tau_max_widget_c.setValue(3)
        self.tau_layout_c.addRow(self.tau_min_label_c, self.tau_min_widget_c)
        self.tau_layout_c.addRow(self.tau_max_label_c, self.tau_max_widget_c)

        self.tab_13c_layout.addLayout(self.tau_layout_c)

        # update button
        self.update_button_c = QPushButton("Update Plot")
        self.update_button_c.clicked.connect(self.update_plot_c)
        self.tab_13c_layout.addWidget(self.update_button_c)

    def load_data(self):
        # Define parameters from simulations
        self.theta = np.arange(0.25, 6, 0.25)
        self.theta = np.append(self.theta, np.arange(6, 32, 2))
        self.B0 = np.arange(10, 800, 10)
        self.N = np.arange(2, 22, 2)
        self.tau = np.linspace(0.05, 3, 1000)

        # Gyromagnetic ratios of most common nuclei
        self.gyromagnetic_ratios = {}

        with open("./data/gyromagnetic_ratios.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.gyromagnetic_ratios[row["substance"]] = float(
                    row["value"])

        self.fam = np.loadtxt(
            "./data/C13_families.csv", skiprows=1, delimiter=",", usecols=(0,), dtype="U20"
        )

        # Load databases from simulations
        # try:
        #     self.XY8_ms_1 = np.load(sys.argv[1], allow_pickle=True)
        #     self.XY8_ms1 = np.load(sys.argv[2], allow_pickle=True)
        # except Exception:
        #     self.XY8_ms_1 = np.load("./database_XY8-N_theta_ms-1.npz", allow_pickle=True)
        #     self.XY8_ms1 = np.load("./database_XY8-N_theta_ms1.npz", allow_pickle=True)

        self.XY8_ms_1_t = np.load(
            "./data/database_XY8-N_theta_ms-1.npz", allow_pickle=True)
        self.XY8_ms1_t = np.load(
            "./data/database_XY8-N_theta_ms1.npz", allow_pickle=True)

        # self.XY8_ms_1_c = np.load(
        #     "./data/database_XY8-N_13C_ms-1.npz", allow_pickle=True)
        # self.XY8_ms1_c = np.load(
        #     "./data/database_XY8-N_13C_ms1.npz", allow_pickle=True)

        self.XY8_ms_1_N14 = np.load("./data/n14.npz", allow_pickle=True)
        self.XY8_ms1_N14 = np.load("./data/n14.npz", allow_pickle=True)

    def resizeEvent(self, event):
        super(MainWindow, self).resizeEvent(event)
        if self.plot_ready:
            if self.tab_widget.currentIndex() == 0:
                self.update_plot_t()
            elif self.tab_widget.currentIndex() == 1:
                self.update_plot_c()

    def select_all_items_t(self):
        for index in range(self.gyromagnetic_widget.count()):
            item = self.gyromagnetic_widget.item(index)
            item.setSelected(True)

    def clear_selection_t(self):
        self.gyromagnetic_widget.clearSelection()

    def select_all_items_c(self):
        for index in range(self.families_widget.count()):
            item = self.families_widget.item(index)
            item.setSelected(True)

    def clear_selection_c(self):
        self.families_widget.clearSelection()

    def select_all_gyro_c(self):
        for index in range(self.gyromagnetic_widget_c.count()):
            item = self.gyromagnetic_widget_c.item(index)
            item.setSelected(True)

    def clear_gyro_c(self):
        self.gyromagnetic_widget_c.clearSelection()

    def set_right_layout(self):
        self.fig = Figure(dpi=400)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel(r"$\tau$ ($\mu$s)")
        self.ax.set_xlim(0.05, 3)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Transition Probability")

        self.canvas = FigureCanvas(self.fig)

        # self.widget_layout.addWidget(self.update_button)

        # Add a navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.right_layout.addWidget(self.toolbar)
        self.right_layout.addWidget(self.canvas)

    def select_file_t(self):
        """
        Select the experimental data file.
        """
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            if selected_file:
                self.expt_filename_t = selected_file
                self.expt_file_label_t.setText(
                    f"File: {os.path.basename(selected_file)}")

    def select_file_c(self):
        """
        Select the experimental data file.
        """
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            if selected_file:
                self.expt_filename_c = selected_file
                self.expt_file_label_c.setText(
                    f"File: {os.path.basename(selected_file)}")

    def update_plot_t(self):
        """
        Update the plot with the selected parameters.
        """
        self.ax.clear()

        selected_ms = int(self.ms_widget_t.currentText())
        selected_N = int(self.N_widget_t.currentText())
        selected_B0 = int(self.B0_widget_t.currentText())
        selected_theta = self.theta_widget_t.currentText()
        selected_tau = [self.tau_min_widget_t.value(),
                        self.tau_max_widget_t.value()]
        selected_gyromagnetic = [
            item.text() for item in self.gyromagnetic_widget.selectedItems()
        ]

        key = f"XY8_{selected_N}_B0{selected_B0}_theta{selected_theta}"

        # Check ms state and loads the corresponding database
        # Check ms state and loads the corresponding database
        if selected_ms == -1:
            try:
                XY8 = self.XY8_ms_1_t[key]
                if XY8.size != 1000:
                    self.ax.text(
                        0.5,
                        0.5,
                        "Not well defined Rabi oscillation for this combination of θ and B\N{SUBSCRIPT ZERO}.",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=self.ax.transAxes,
                        bbox=dict(facecolor="yellow",
                                  alpha=0.5, edgecolor="black"),
                    )
                else:
                    self.ax.plot(self.tau, XY8, linewidth=1,
                                 label=f"Simulated data")
            except KeyError:
                idxerr = QMessageBox(self)
                idxerr.setText(
                    "Simulation results not available for these values")
                idxerr.setWindowTitle("Data doesn't exist")
                idxerr.exec()

        elif selected_ms == 1:
            try:
                XY8 = self.XY8_ms1_t[key]
                if XY8.size != 1000:
                    self.ax.text(
                        0.5,
                        0.5,
                        "Not well defined Rabi oscillation for this combination of θ and B\N{SUBSCRIPT ZERO}.",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=self.ax.transAxes,
                        bbox=dict(facecolor="yellow",
                                  alpha=0.5, edgecolor="black"),
                    )
                else:
                    self.ax.plot(self.tau, XY8, linewidth=1,
                                 label="Simulated Data")
            except KeyError:
                idxerr = QMessageBox(self)
                idxerr.setText(
                    "Simulation results not available for these values")
                idxerr.setWindowTitle("Data doesn't exist")
                idxerr.exec()

        # Plot experimental data if selected
        if self.expt_chkbox_t.isChecked() and self.expt_filename_t != "":
            exp_data = np.loadtxt(self.expt_filename_t, delimiter="\t")

            p = exp_data[:, 1] - exp_data[:, 2] - \
                min(exp_data[:, 1] - exp_data[:, 2])

            self.ax.plot(
                exp_data[:, 0] * 1e6, p, linewidth=1, label="Experimental Data"
            )

        else:
            pass

        # Plot vertical lines for the gyromagnetic ratios chosen
        if selected_B0 != 0:
            for idx, val in enumerate(selected_gyromagnetic):
                pos = 1 / \
                    (2 * self.gyromagnetic_ratios[val] * selected_B0 * 1e-4)
                self.ax.axvline(
                    x=pos,
                    linestyle="--",
                    linewidth=1,
                    label=f"{val} (at {pos:.2f})",
                    color=colors[idx + 1],
                )
        elif selected_B0 == 0 and len(selected_gyromagnetic) != 0:
            B0err = QMessageBox(self)
            B0err.setText("B<sub>0</sub> can not be 0 for comparison with γ")
            B0err.setWindowTitle("Cannot divide by zero")
            B0err.exec()

        self.ax.set_xlabel(r"$\tau$ ($\mu$s)")
        self.ax.set_ylabel("Transition Probability")
        self.ax.set_title(
            rf"XY8-{selected_N} ; $\theta=${selected_theta}$^\circ$ ; $B_0=${selected_B0} G"
        )
        self.ax.set_xlim(selected_tau[0], selected_tau[1])

        self.ax.legend(fancybox=True, framealpha=0.3)
        self.fig.tight_layout()

        self.canvas.draw()
        self.plot_ready = True

    def update_plot_c(self):
        """
        Update the plot with the selected parameters.
        """
        self.ax.clear()

        selected_ms = int(self.ms_widget_c.currentText())
        selected_N = int(self.N_widget_c.currentText())
        selected_B0 = int(self.B0_widget_c.currentText())
        selected_tau = [self.tau_min_widget_c.value(),
                        self.tau_max_widget_c.value()]
        selected_gyromagnetic = [
            item.text() for item in self.gyromagnetic_widget_c.selectedItems()
        ]

        selected_fam = [item.text()
                        for item in self.families_widget.selectedItems()]

        for fam in selected_fam:
            key = f"XY8_{selected_N}_B0{selected_B0}_family{fam}"

            # Check ms state and loads the corresponding database
            if selected_ms == -1:
                try:
                    XY8 = self.XY8_ms_1_c[key]
                    if XY8.size != 1000:
                        self.ax.text(
                            0.5,
                            0.5,
                            "Not well defined Rabi oscillation for this combination of θ and B\N{SUBSCRIPT ZERO}.",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=self.ax.transAxes,
                            bbox=dict(facecolor="yellow",
                                      alpha=0.5, edgecolor="black"),
                        )
                    else:
                        self.ax.plot(self.tau, XY8, linewidth=1,
                                     label=f"{fam}")
                except KeyError:
                    idxerr = QMessageBox(self)
                    idxerr.setText(
                        "Simulation results not available for these values")
                    idxerr.setWindowTitle("Data doesn't exist")
                    idxerr.exec()
                    break

            elif selected_ms == 1:
                try:
                    XY8 = self.XY8_ms1_c[key]
                    if XY8.size != 1000:
                        self.ax.text(
                            0.5,
                            0.5,
                            "Not well defined Rabi oscillation for this combination of θ and B\N{SUBSCRIPT ZERO}.",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=self.ax.transAxes,
                            bbox=dict(facecolor="yellow",
                                      alpha=0.5, edgecolor="black"),
                        )
                    else:
                        self.ax.plot(self.tau, XY8, linewidth=1,
                                     label=f"{fam}")
                except KeyError:
                    idxerr = QMessageBox(self)
                    idxerr.setText(
                        "Simulation results not available for these values")
                    idxerr.setWindowTitle("Data doesn't exist")
                    idxerr.exec()
                    break

            # Check if the simulation has well-defined Rabi oscillation

        # Plot experimental data if selected
        if self.expt_chkbox_c.isChecked() and self.expt_filename_c != "":
            exp_data = np.loadtxt(self.expt_filename_c, delimiter="\t")

            p = exp_data[:, 1] - exp_data[:, 2] - \
                min(exp_data[:, 1] - exp_data[:, 2])

            self.ax.plot(
                exp_data[:, 0] * 1e6, p, linewidth=1, label="Experimental Data"
            )

        else:
            pass

        # Plot vertical lines for the gyromagnetic ratios chosen
        if selected_B0 != 0:
            for idx, val in enumerate(selected_gyromagnetic):
                pos = 1 / \
                    (2 * self.gyromagnetic_ratios[val] * selected_B0 * 1e-4)
                self.ax.axvline(
                    x=pos,
                    linestyle="--",
                    linewidth=1,
                    label=f"{val} (at {pos:.2f})",
                    color=colors[idx + 1],
                )
        elif selected_B0 == 0 and len(selected_gyromagnetic) != 0:
            B0err = QMessageBox(self)
            B0err.setText("B<sub>0</sub> can not be 0 for comparison with γ")
            B0err.setWindowTitle("Cannot divide by zero")
            B0err.exec()

        self.ax.set_xlabel(r"$\tau$ ($\mu$s)")
        self.ax.set_ylabel("Transition Probability")
        self.ax.set_title(rf"XY8-{selected_N} ; $B_0=${selected_B0} G")
        self.ax.set_xlim(selected_tau[0], selected_tau[1])

        self.ax.legend(fancybox=True, framealpha=0.3)
        self.fig.tight_layout()

        self.canvas.draw()
        self.plot_ready = True

    def set_tab_N14(self):
        self.ms_label_N14 = QLabel("m<sub>s</sub> state:")
        self.ms_widget_N14 = QComboBox()
        self.ms_widget_N14.addItems(["-1", "1"])
        self.ms_label_N14.setToolTip(
            "Electronic spin state. For magnetic fields near the level anti-crossing (532 G), the state ms=-1 is "
            "close to ms=0 and the system might not present well defined spin manipulations."
        )

        # XY8-N order
        self.N_label_N14 = QLabel("N:")
        self.N_widget_N14 = QComboBox()
        self.N_widget_N14.addItems([str(val) for val in self.N])
        self.N_label_N14.setToolTip(
            "Order of the XY8-N sequence. For high order, the exact duration of the pi pulses becomes important "
            "factor. in this simulations we used pi pulses of around 15 ns."
        )

        # External magnetic field intensity
        self.B0_label_N14 = QLabel("B\N{SUBSCRIPT ZERO} (G):")
        self.B0_widget_N14 = QComboBox()
        self.B0_widget_N14.setMaxVisibleItems(20)
        self.B0_widget_N14.addItems([str(val) for val in self.B0])
        self.B0_label_N14.setToolTip(
            "Intensity of the external magnetic field. At low magnetic fields, both electronic spin states are nearly "
            "degenerate and the system presents a complex S=1 dynamics, which may lead to not well defined pulses."
        )

        # Misalignment angle
        self.theta_label_N14 = QLabel("θ (°):")
        self.theta_widget_N14 = QComboBox()
        self.theta_widget_N14.addItems([str(val) for val in self.theta])
        self.theta_label_N14.setToolTip(
            "Angle between the external magnetic field and the quantization axis of the electron spin. At high angles "
            "and field, the nuclear spin is not good quantum number and the Hamiltonian might not represent "
            "faithfully the system."
        )

        left_form_layout_N14 = QFormLayout()
        left_form_layout_N14.addRow(self.ms_label_N14, self.ms_widget_N14)
        left_form_layout_N14.addRow(self.N_label_N14, self.N_widget_N14)
        left_form_layout_N14.addRow(self.B0_label_N14, self.B0_widget_N14)
        left_form_layout_N14.addRow(
            self.theta_label_N14, self.theta_widget_N14)

        self.tab_N14_layout.addLayout(left_form_layout_N14)
        # self.left_layout.addWidget(self.horizontal_line)

        # Optional experimental data file to compare with
        self.expt_file_label_N14 = QLabel("No file selected")

        self.expt_file_btn_N14 = QPushButton("Select File")
        self.expt_file_btn_N14.clicked.connect(self.select_file_N14)

        expt_HLay = QHBoxLayout()
        self.expt_label_N14 = QLabel("Compare with Experimental Data")
        self.expt_chkbox_N14 = QCheckBox()
        expt_HLay.addWidget(self.expt_label_N14)
        expt_HLay.addWidget(self.expt_chkbox_N14)

        self.expt_label_N14.setToolTip(
            "Experimental data must be in a file with two columns: first column is the pulse separation tau in μs and "
            "the second column is the transition probability."
        )
        self.expt_filename_N14 = ""  # to handle plotting when no file is selected

        self.tab_N14_layout.addWidget(self.expt_file_label_N14)
        self.tab_N14_layout.addWidget(self.expt_file_btn_N14)

        self.tab_N14_layout.addLayout(expt_HLay)

        self.gyromagnetic_label_N14 = QLabel("Compare peaks with:")
        self.gyromagnetic_widget_N14 = QListWidget()
        self.gyromagnetic_widget_N14.addItems(
            [key for key in self.gyromagnetic_ratios.keys()]
        )
        self.gyromagnetic_widget_N14.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.gyromagnetic_label_N14.setToolTip(
            "The corresponding tau is 1/(2*γ*B<sub>0</sub>)."
        )

        self.tab_N14_layout.addWidget(self.gyromagnetic_label_N14)
        self.tab_N14_layout.addWidget(self.gyromagnetic_widget_N14)

        self.select_all_button_N14 = QPushButton("Select all")
        self.clear_button_N14 = QPushButton("Clear selection")

        self.select_all_button_N14.clicked.connect(self.select_all_items_N14)
        self.clear_button_N14.clicked.connect(self.clear_selection_N14)

        gyromagnetic_btn_lay_N14 = QFormLayout()
        gyromagnetic_btn_lay_N14.addRow(
            self.select_all_button_N14, self.clear_button_N14)
        self.tab_N14_layout.addLayout(gyromagnetic_btn_lay_N14)

        # tau range
        self.tau_layout_N14 = QFormLayout()

        self.tau_min_label_N14 = QLabel("min τ:")
        self.tau_min_widget_N14 = QDoubleSpinBox()
        self.tau_min_widget_N14.setRange(0.05, 3.0)
        self.tau_min_widget_N14.setSingleStep(0.1)
        self.tau_min_widget_N14.setValue(0.05)

        self.tau_max_label_N14 = QLabel("max τ:")
        self.tau_max_widget_N14 = QDoubleSpinBox()
        self.tau_max_widget_N14.setRange(0.05, 3.0)
        self.tau_max_widget_N14.setSingleStep(0.1)
        self.tau_max_widget_N14.setValue(3)
        self.tau_layout_N14.addRow(
            self.tau_min_label_N14, self.tau_min_widget_N14)
        self.tau_layout_N14.addRow(
            self.tau_max_label_N14, self.tau_max_widget_N14)

        self.tab_N14_layout.addLayout(self.tau_layout_N14)

        # update button
        self.update_button_N14 = QPushButton("Update Plot")
        self.update_button_N14.clicked.connect(self.update_plot_N14)
        self.tab_N14_layout.addWidget(self.update_button_N14)

    def select_all_items_N14(self):
        for index in range(self.gyromagnetic_widget_N14.count()):
            item = self.gyromagnetic_widget_N14.item(index)
            item.setSelected(True)

    def clear_selection_N14(self):
        self.gyromagnetic_widget_N14.clearSelection()

    def select_file_N14(self):
        """
        Select the experimental data file.
        """
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            if selected_file:
                self.expt_filename_N14 = selected_file
                self.expt_file_label_N14.setText(
                    f"File: {os.path.basename(selected_file)}")

    def update_plot_N14(self):
        """
        Update the plot with the selected parameters.
        """
        self.ax.clear()

        selected_ms = int(self.ms_widget_N14.currentText())
        selected_N = int(self.N_widget_N14.currentText())
        selected_B0 = int(self.B0_widget_N14.currentText())
        selected_theta = self.theta_widget_N14.currentText()
        selected_tau = [self.tau_min_widget_N14.value(),
                        self.tau_max_widget_N14.value()]
        selected_gyromagnetic = [
            item.text() for item in self.gyromagnetic_widget_N14.selectedItems()
        ]

        key = f"XY8_{selected_N}_B0{selected_B0}_theta{selected_theta}"

        # Check ms state and loads the corresponding database
        if selected_ms == -1:
            try:
                # XY8 = self.XY8_ms_1_N14[key]
                XY8 = self.XY8_ms_1_N14[key+'_f1']
                if XY8.size != 1000:
                    self.ax.text(
                        0.5,
                        0.5,
                        "Not well defined Rabi oscillation for this combination of θ and B\N{SUBSCRIPT ZERO}.",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=self.ax.transAxes,
                        bbox=dict(facecolor="yellow",
                                  alpha=0.5, edgecolor="black"),
                    )
                else:
                    self.ax.plot(self.tau, XY8, linewidth=1,
                                 label=f"Simulated data")
            except KeyError:
                idxerr = QMessageBox(self)
                idxerr.setText(
                    "Simulation results not available for these values")
                idxerr.setWindowTitle("Data doesn't exist")
                idxerr.exec()

        elif selected_ms == 1:
            try:
                # XY8 = self.XY8_ms1_N14[key]
                XY8 = self.XY8_ms1_N14[key+'_f2']
                if XY8.size != 1000:
                    self.ax.text(
                        0.5,
                        0.5,
                        "Not well defined Rabi oscillation for this combination of θ and B\N{SUBSCRIPT ZERO}.",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=self.ax.transAxes,
                        bbox=dict(facecolor="yellow",
                                  alpha=0.5, edgecolor="black"),
                    )
                else:
                    self.ax.plot(self.tau, XY8, linewidth=1,
                                 label="Simulated Data")
            except KeyError:
                idxerr = QMessageBox(self)
                idxerr.setText(
                    "Simulation results not available for these values")
                idxerr.setWindowTitle("Data doesn't exist")
                idxerr.exec()

        # Plot experimental data if selected
        if self.expt_chkbox_N14.isChecked() and self.expt_filename_N14 != "":
            exp_data = np.loadtxt(self.expt_filename_N14, delimiter="\t")

            p = exp_data[:, 1] - exp_data[:, 2] - \
                min(exp_data[:, 1] - exp_data[:, 2])

            self.ax.plot(
                exp_data[:, 0] * 1e6, p, linewidth=1, label="Experimental Data"
            )

        else:
            pass

        # Plot vertical lines for the gyromagnetic ratios chosen
        if selected_B0 != 0:
            for idx, val in enumerate(selected_gyromagnetic):
                pos = 1 / \
                    (2 * self.gyromagnetic_ratios[val] * selected_B0 * 1e-4)
                self.ax.axvline(
                    x=pos,
                    linestyle="--",
                    linewidth=1,
                    label=f"{val} (at {pos:.2f})",
                    color=colors[idx + 1],
                )
        elif selected_B0 == 0 and len(selected_gyromagnetic) != 0:
            B0err = QMessageBox(self)
            B0err.setText("B<sub>0</sub> can not be 0 for comparison with γ")
            B0err.setWindowTitle("Cannot divide by zero")
            B0err.exec()

        self.ax.set_xlabel(r"$\tau$ ($\mu$s)")
        self.ax.set_ylabel("Transition Probability")
        self.ax.set_title(
            rf"XY8-{selected_N} ; $\theta=${selected_theta}$^\circ$ ; $B_0=${selected_B0} G"
        )
        self.ax.set_xlim(selected_tau[0], selected_tau[1])

        self.ax.legend(fancybox=True, framealpha=0.3)
        self.fig.tight_layout()

        self.canvas.draw()
        self.plot_ready = True

    def set_tab_sim(self):
        self.sim_task_label = QLabel("Simulate: ")
        self.sim_task_widget = QComboBox()
        self.sim_task_widget.addItems(["Rabi","Hahn"])

        self.sim_tau_layout = QFormLayout()
        # self.sim_tau_min_label = QLabel("τ inital:")
        # self.sim_tau_min_widget = QDoubleSpinBox()
        # self.sim_tau_min_widget.setRange(0.05, 3.0)
        # self.sim_tau_min_widget.setSingleStep(0.1)
        # self.sim_tau_min_widget.setValue(0.05)
        self.sim_tau_max_label = QLabel("τ final:")
        self.sim_tau_max_widget = QDoubleSpinBox()
        self.sim_tau_max_widget.setRange(0.03, 1)
        self.sim_tau_max_widget.setSingleStep(0.01)
        self.sim_tau_max_widget.setValue(0.06)
        # self.sim_tau_layout.addRow(self.sim_tau_min_label, self.sim_tau_min_widget)
        
        self.sim_w1_label = QLabel("ω1")
        self.sim_w1_widget = QDoubleSpinBox()
        self.sim_w1_widget.setValue(40)
        self.sim_w1_widget.setSingleStep(0.1)

        self.sim_N_label = QLabel("Nitrogen: ")
        self.sim_N_widget = QComboBox()
        self.sim_N_widget.addItems(["14","15"])

        self.sim_ms_label = QLabel("m<sub>s</sub> state:")
        self.sim_ms_widget = QComboBox()
        self.sim_ms_widget.addItems(["-1", "1"])
        self.sim_ms_label.setToolTip(
            "Electronic spin state. For magnetic fields near the level anti-crossing (532 G), the state ms=-1 is "
            "close to ms=0 and the system might not present well defined spin manipulations."
        )

        # External magnetic field intensity
        self.sim_B0_label = QLabel("B\N{SUBSCRIPT ZERO} (G):")
        self.sim_B0_widget = QComboBox()
        self.sim_B0_widget.setMaxVisibleItems(20)
        self.sim_B0_widget.addItems([str(val) for val in self.B0])
        self.sim_B0_label.setToolTip(
            "Intensity of the external magnetic field. At low magnetic fields, both electronic spin states are nearly "
            "degenerate and the system presents a complex S=1 dynamics, which may lead to not well defined pulses."
        )

        self.sim_theta_label = QLabel("θ (°):")
        self.sim_theta_widget = QComboBox()
        self.sim_theta_widget.addItems([str(val) for val in self.theta])
        self.sim_theta_label.setToolTip(
            "Angle between the external magnetic field and the quantization axis of the electron spin. At high angles "
            "and field, the nuclear spin is not good quantum number and the Hamiltonian might not represent "
            "faithfully the system."
        )

        self.simformlayout = QFormLayout()
        self.simformlayout.addRow(self.sim_task_label,self.sim_task_widget)
        self.simformlayout.addRow(self.sim_tau_max_label, self.sim_tau_max_widget)
        self.simformlayout.addRow(self.sim_N_label,self.sim_N_widget)
        self.simformlayout.addRow(self.sim_w1_label,self.sim_w1_widget)
        self.simformlayout.addRow(self.sim_B0_label,self.sim_B0_widget)
        self.simformlayout.addRow(self.sim_theta_label,self.sim_theta_widget)
        self.simformlayout.addRow(self.sim_ms_label,self.sim_ms_widget)
        
        # self.simformlayout.addRow(QLabel("Rabi"),self.horizontal_line)
        self.tab_sim_layout.addLayout(self.simformlayout)

        self.simbtn = QPushButton("Run simulation")
        self.simbtn.pressed.connect(self.start_process)
        self.tab_sim_layout.addWidget(self.simbtn)
        self.simtext = QPlainTextEdit()
        self.simtext.setReadOnly(True)
        self.tab_sim_layout.addWidget(self.simtext)


    def simmessage(self, s):
        self.simtext.appendPlainText(s)

    def start_process(self):
        if self.p is None:  # No process running.
            self.simmessage("Running simulation")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            # self.p.stateChanged.connect(self.handle_state)
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            arguments=[
                "-B", self.sim_B0_widget.currentText(),
                "-N", self.sim_N_widget.currentText(),
                "-m", self.sim_ms_widget.currentText(),
                "--sim", self.sim_task_widget.currentText(),
                "--theta", self.sim_theta_widget.currentText(),
                "--w1", str(self.sim_w1_widget.value()),
                "--trabimax", str(self.sim_tau_max_widget.value())
            ]
            self.p.finished.connect(self.process_finished)
            self.p.start("python", ['sim_script.py']+arguments)

    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.simmessage(stderr)

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.simmessage(stdout)

    # def handle_state(self, state):
    #     states = {
    #         QProcess.ProcessState.NotRunning: 'Not running',
    #         QProcess.ProcessState.Starting: 'Starting',
    #         QProcess.ProcessState.Running: 'Running',
    #     }
    #     state_name = states[state]
    #     self.simmessage(f"State changed: {state_name}")
    
    def process_finished(self):
        self.simmessage("Simulation finished.")
        self.p = None
        if self.sim_task_widget.currentText()=='Rabi':
            simres = np.load('./data/rabi.npz', allow_pickle=True)
            self.ax.clear()
            self.ax.plot(simres['t'],simres['r'],'.',label='Simulation (Rabi)')
            self.ax.plot(simres['t'],simres['fit'][0]*np.cos(2*np.pi*simres['t']/simres['fit'][1])**2 + simres['fit'][2],label='Fit')
            self.ax.legend(fancybox=True, framealpha=0.3)
            self.ax.set_xlabel(r"$\tau$ ($\mu$s)")
            self.ax.set_ylabel("Transition Probability")
            self.ax.set_title(
            rf"{self.sim_task_widget.currentText()}; $\theta=${self.sim_theta_widget.currentText()}$^\circ$ ; $B_0=${self.sim_B0_widget.currentText()}G")
        
        elif self.sim_task_widget.currentText()=='Hahn':
            simres = np.load('./data/hahn.npz', allow_pickle=True)
            self.ax.clear()
            self.ax.plot(simres['t'],simres['r'],linewidth=1,label='Simulation (Hahn)')
            self.ax.legend(fancybox=True, framealpha=0.3)
            self.ax.set_xlabel(r"$\tau$ ($\mu$s)")
            self.ax.set_ylabel("Transition Probability")
            self.ax.set_title(
            rf"{self.sim_task_widget.currentText()}; $\theta=${self.sim_theta_widget.currentText()}$^\circ$ ; $B_0=${self.sim_B0_widget.currentText()}G")

        self.fig.tight_layout()


        self.canvas.draw()
        self.plot_ready = True



if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyle('Breeze')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
