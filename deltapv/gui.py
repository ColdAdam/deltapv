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
import numpy as np


class MainWindow(object):
    """
    Boot the Qt designer code then connect up as needed

    validator = QtGui.QDoubleValidator()
    self.lineEdit_model_resolution.setValidator(validator)
    """

    def __init__(self, window):

        self.app = gui_designer.Ui_MainWindow()
        self.app.setupUi(window)

        self.graphs()
        self.modelLayout()
        self.editMaterial()
        self.menuBar()

    def graphs(self):

        band = plt.Figure(figsize=(6, 5), dpi=100)
        model = plt.Figure(figsize=(6, 5), dpi=100)
        charge = plt.Figure(figsize=(6, 5), dpi=100)
        iv = plt.Figure(figsize=(6, 5), dpi=100)

        def alphaGraph():

            plt_alpha = plt.Figure(figsize=(6, 5), dpi=100)
            ax1 = plt_alpha.add_subplot(111)

            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_ylabel(r"Alpha (cm$^{-1}$)")
            ax1.grid(1)

            return plt_alpha, ax1

        self.plt_band = plotting.plot_band_diagram(gui=band)
        self.plt_model = plotting.plot_bars(gui=model)
        self.plt_charge = plotting.plot_charge(gui=charge)
        self.plt_iv = plotting.plot_iv_curve(gui=iv)
        self.plt_alpha, self.ax_alpha = alphaGraph()

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

        self.graphcanvas_alpha = FigureCanvasQTAgg(self.plt_alpha)
        self.layoutPlot_alpha = QtWidgets.QWidget(self.app.graphicsView_alphaGraph)  # noqa
        self.gridPlot_alpha = QtWidgets.QGridLayout(self.layoutPlot_alpha)
        self.gridPlot_alpha.addWidget(self.graphcanvas_alpha)
        self.toolbar_alpha = NavigationToolbar2QT(self.graphcanvas_alpha,
                                                  self.app.graphicsView_alphaGraph)  # noqa
        self.gridPlot_alpha.addWidget(self.toolbar_alpha)

    def modelLayout(self):

        dir_material = Path(materials.__file__).parent.joinpath("resources")
        if not dir_material.exists():
            dir_material = ""

        def loadMaterial():
            pth_material, _ = QFileDialog.getOpenFileName(
                caption="Open DeltaPV material file",
                filter="*.yaml",
                directory=str(dir_material))
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

    def editMaterial(self):
        dir_material = Path(materials.__file__).parent.joinpath("resources")
        if not dir_material.exists():
            dir_material = ""

        def loadMaterial():
            pth_material, _ = QFileDialog.getOpenFileName(
                caption="Open DeltaPV material file",
                filter="*.yaml",
                directory=str(dir_material))
            if pth_material:
                pth_material = Path(pth_material)
                mat = materials.load_material(pth_material.stem)

                for i in mat.__iter__():
                    try:
                        attr = getattr(self.app, f"lineEdit_{i[0]}")
                        attr.setText(str(float(i[1])))
                    except AttributeError:
                        continue

                self.ax_alpha.plot(materials.lam_interp,
                                   mat.alpha,
                                   label="alpha")
                self.graphcanvas_alpha.draw_idle()

                npdf = np.array([materials.lam_interp, mat.alpha]).transpose()
                data = [[str(x[0]), str(x[1])] for x in npdf]
                alphamodel = self.alphaTable(data)
                self.app.tableView_alphaTable.setModel(alphamodel)

        def saveMaterial():
            pth_material, _ = QFileDialog.getSaveFileName(
                caption="Save DeltaPV material file",
                filter="*.yaml",
                directory=str(dir_material))
            if pth_material:
                pass

        self.app.pushButton_editLoadMaterial.clicked.connect(loadMaterial)
        self.app.pushButton_editSaveMaterial.clicked.connect(saveMaterial)

    def alphaTable(self, data):

        class TableModel(QtCore.QAbstractTableModel):
            def __init__(self, data):
                super(TableModel, self).__init__()
                self._data = data

            def data(self, index, role):
                if role == QtCore.Qt.DisplayRole:
                    # See below for the nested-list data structure.
                    # .row() indexes into the outer list,
                    # .column() indexes into the sub-list
                    return self._data[index.row()][index.column()]

            def rowCount(self, index):
                # The length of the outer list.
                return len(self._data)

            def columnCount(self, index):
                # The following takes the first sub-list, and returns
                # the length (only works if all rows are an equal length)
                return len(self._data[0])
        return TableModel(data)

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
