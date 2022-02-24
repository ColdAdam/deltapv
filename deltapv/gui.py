'''
Created on Feb 21, 2022

@author: adamt

Setup and start the GUI window

TODO: Add to Help
n_points (i64): Number of points on a uniform grid
Ls (List[f64]): Thicknesses of each layer
mats (Union[List[Material], Material]): List of materials
Ns (List[f64]): List of doping densities
Snl (f64): Electron recombination velocity at front contact
Snr (f64): Electron recombination velocity at back contact
Spl (f64): Hole recombination velocity at front contact
Spr (f64): Hole recombination velocity at back contact
PhiM0 (f64, optional): Workfunction of front contact. Defaults 
 to -1.
PhiML (f64, optional): Workfunction of back contact. Defaults 
 to -1.
'''

# TODO: change to from deltapv import gui, gui_designer
import gui_designer, plotting
from deltapv import materials
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)

import matplotlib.pyplot as plt
import sys


class MainWindow(object):
    """
    Boot the Qt designer code then connect up as needed

    validator = QtGui.QDoubleValidator()
    self.lineEdit_model_resolution.setValidator(validator)
    """

    def __init__(self, window):
        self.app = gui_designer.Ui_MainWindow()
        self.app.setupUi(window)
        # Add needed connections ..
        self.graphs()
        self.modelLayout()
        self.menuBar()

    def graphs(self):

        band = plt.Figure(figsize=(6, 5), dpi=100)
        model = plt.Figure(figsize=(6, 5), dpi=100)
        charge = plt.Figure(figsize=(6, 5), dpi=100)
        iv = plt.Figure(figsize=(6, 5), dpi=100)

        self.plt_band = plotting.plot_band_diagram(gui=band)
        self.plt_model = plotting.plot_bars(gui=model)
        self.plt_charge = plotting.plot_charge(gui=charge)
        self.plt_iv = plotting.plot_iv_curve(gui=iv)

        self.graphcanvas_iv = FigureCanvasQTAgg(self.plt_iv)
        self.layoutPlot_iv = QtWidgets.QWidget(self.app.graphicsView_iv)
        self.gridPlot_iv = QtWidgets.QGridLayout(self.layoutPlot_iv)
        self.gridPlot_iv.addWidget(self.graphcanvas_iv)
        self.toolbar_iv = NavigationToolbar2QT(self.graphcanvas_iv,
                                               self.app.graphicsView_iv)
        self.gridPlot_iv.addWidget(self.toolbar_iv)

        self.graphcanvas_model = FigureCanvasQTAgg(self.plt_model)
        self.layoutPlot_model = QtWidgets.QWidget(self.app.graphicsView_model)
        self.gridPlot_model = QtWidgets.QGridLayout(self.layoutPlot_model)
        self.gridPlot_model.addWidget(self.graphcanvas_model)
        self.toolbar_model = NavigationToolbar2QT(self.graphcanvas_model,
                                                  self.app.graphicsView_model)
        self.gridPlot_model.addWidget(self.toolbar_model)

        self.graphcanvas_band = FigureCanvasQTAgg(self.plt_band)
        self.layoutPlot_band = QtWidgets.QWidget(self.app.graphicsView_band)
        self.gridPlot_band = QtWidgets.QGridLayout(self.layoutPlot_band)
        self.gridPlot_band.addWidget(self.graphcanvas_band)
        self.toolbar_band = NavigationToolbar2QT(self.graphcanvas_band,
                                                 self.app.graphicsView_band)
        self.gridPlot_band.addWidget(self.toolbar_band)

        self.graphcanvas_charge = FigureCanvasQTAgg(self.plt_charge)
        self.layoutPlot_charge = QtWidgets.QWidget(self.app.graphicsView_charge)  # noqa
        self.gridPlot_charge = QtWidgets.QGridLayout(self.layoutPlot_charge)
        self.gridPlot_charge.addWidget(self.graphcanvas_charge)
        self.toolbar_charge = NavigationToolbar2QT(self.graphcanvas_charge,
                                                   self.app.graphicsView_charge)  # noqa
        self.gridPlot_charge.addWidget(self.toolbar_charge)

    def modelLayout(self):

        dir_material = Path(materials.__file__).joinpath("resources")
        if not dir_material.exists():
            dir_material = ""

        def loadMaterial():
            pth_material, _ = QFileDialog.getOpenFileName(
                caption="Open DeltaPV material file",
                filter="*.yaml",
                directory=dir_material)
            if pth_material:
                pth_material = Path(pth_material)
                rowPosition = self.app.tableWidget_modellayout.rowCount()
                self.app.tableWidget_modellayout.insertRow(rowPosition)
                self.app.tableWidget_modellayout.setItem(
                    rowPosition, 0,
                    QtWidgets.QTableWidgetItem(pth_material.stem))

        def removeMaterial():
            indices = self.app.tableWidget_modellayout.selectionModel(
                ).selectedRows()
            for index in sorted(indices):
                self.app.tableWidget_modellayout.removeRow(index.row())

        def moveUp():
            row = self.app.tableWidget_modellayout.currentRow()
            column = self.app.tableWidget_modellayout.currentColumn()
            if row > 0:
                self.app.tableWidget_modellayout.insertRow(row - 1)
                for i in range(self.app.tableWidget_modellayout.columnCount()):
                    self.app.tableWidget_modellayout.setItem(
                        row - 1, i,
                        self.app.tableWidget_modellayout.takeItem(row + 1, i))
                    self.app.tableWidget_modellayout.setCurrentCell(row - 1,
                                                                    column)
                self.app.tableWidget_modellayout.removeRow(row + 1)

        def moveDown():
            row = self.app.tableWidget_modellayout.currentRow()
            column = self.app.tableWidget_modellayout.currentColumn()
            if row < self.app.tableWidget_modellayout.rowCount() - 1:
                self.app.tableWidget_modellayout.insertRow(row + 2)
                for i in range(self.app.tableWidget_modellayout.columnCount()):
                    self.app.tableWidget_modellayout.setItem(
                        row + 2, i,
                        self.app.tableWidget_modellayout.takeItem(row, i))
                    self.app.tableWidget_modellayout.setCurrentCell(row + 2,
                                                                    column)
                self.app.tableWidget_modellayout.removeRow(row)

        self.app.pushButton_loadmaterial.clicked.connect(loadMaterial)
        self.app.pushButton_remove.clicked.connect(removeMaterial)
        self.app.toolButton_moveup.clicked.connect(moveUp)
        self.app.toolButton_movedown.clicked.connect(moveDown)

    def menuBar(self):

        self.app.actionModel_Setup.triggered.connect(
            lambda: self.app.stackedWidget.setCurrentIndex(0))
        self.app.actionMaterial_Editor.triggered.connect(
            lambda: self.app.stackedWidget.setCurrentIndex(1))
        self.app.actionHelp.triggered.connect(
            lambda: self.app.stackedWidget.setCurrentIndex(2))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    gui = MainWindow(window)
    window.show()
    sys.exit(app.exec_())
