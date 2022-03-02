'''
Created on Feb 21, 2022

@author: adamt

Setup and start the GUI window
'''

# TODO: change to from deltapv import gui, gui_designer
import gui_designer, plotting
import deltapv as dpv
from deltapv import materials

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import numpy as np
import yaml

import pickle

from jax.lib import xla_bridge

print("Jax config: %s" % xla_bridge.get_backend().platform)


class MainWindow(object):
    """
    Boot the Qt designer code then connect up as needed
    """

    def __init__(self, window):

        self.app = gui_designer.Ui_MainWindow()
        self.app.setupUi(window)

        self.graphs()
        self.modelLayout()
        self.editMaterial()
        self.menuBar()
        self.validators()
        self.dpvSimulator()

        self.des = None

    def graphs(self):

        # band = plt.Figure(figsize=(6, 5), dpi=100)
        # model = plt.Figure(figsize=(6, 5), dpi=100)
        # charge = plt.Figure(figsize=(6, 5), dpi=100)
        # iv = plt.Figure(figsize=(6, 5), dpi=100)

        def alphaGraph():

            plt_alpha = plt.Figure(figsize=(6, 5), dpi=100)
            ax1 = plt_alpha.add_subplot(111)

            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_ylabel(r"Alpha (cm$^{-1}$)")
            ax1.grid(1)

            return plt_alpha, ax1

        self.plt_band, self.ax_band = plotting.plot_band_diagram()
        self.plt_model, self.ax_model = plotting.plot_bars()
        self.plt_charge, self.ax_charge = plotting.plot_charge()
        self.plt_iv, self.ax_iv = plotting.plot_iv_curve()
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

                properties = {"Eg": None, "Et": None, "Chi": None,
                              "eps": None,
                              "Nc": None, "Nv": None,
                              "Ndop": None,
                              "mn": None, "mp": None,
                              "tn": None, "tp": None,
                              "Cn": None, "Cp": None,
                              "Br": None, "A": None}

                for key in properties.keys():
                    attr = getattr(self.app, f"lineEdit_{key}")
                    attr.clear()
                    try:
                        attr.setText(str(float(getattr(mat, key))))
                    except AttributeError:
                        continue

                self.ax_alpha.cla()
                self.ax_alpha.plot(materials.lam_interp,
                                   mat.alpha,
                                   label="alpha")
                self.graphcanvas_alpha.draw_idle()

                npdf = np.array([materials.lam_interp, mat.alpha]).transpose()
                data = [[str(x[0]), str(x[1])] for x in npdf]
                alphamodel = self.alphaTable(data)
                self.app.tableView_alphaTable.setModel(alphamodel)

        def saveMaterial():
            pth_material, ext = QFileDialog.getSaveFileName(
                caption="Save DeltaPV material file",
                filter="*.yaml",
                directory=str(dir_material))
            if pth_material:
                pth_material = Path(pth_material)
                name = pth_material.stem
                if pth_material.suffix != '.yaml':
                    pth_material = pth_material.parent.joinpath(
                        name + ext.lstrip("*"))
                material = {"name": name,
                            "properties": {"Eg": None, "Et": None, "Chi": None,
                                           "eps": None,
                                           "Nc": None, "Nv": None,
                                           "Ndop": None,
                                           "mn": None, "mp": None,
                                           "tn": None, "tp": None,
                                           "Cn": None, "Cp": None,
                                           "Br": None, "A": None}
                            }
                for i in material["properties"].keys():
                    attr = getattr(self.app, f"lineEdit_{i}")
                    if attr.text():
                        material["properties"][i] = float(attr.text())
                with open(pth_material, 'w') as f:
                    yaml.dump(material, f)

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

    def validators(self):
        validator = QtGui.QDoubleValidator()

        self.app.lineEdit_A.setValidator(validator)
        self.app.lineEdit_Br.setValidator(validator)
        self.app.lineEdit_Chi.setValidator(validator)
        self.app.lineEdit_Cn.setValidator(validator)
        self.app.lineEdit_Cp.setValidator(validator)
        self.app.lineEdit_Eg.setValidator(validator)
        self.app.lineEdit_eps.setValidator(validator)
        self.app.lineEdit_Et.setValidator(validator)
        self.app.lineEdit_mn.setValidator(validator)
        self.app.lineEdit_mp.setValidator(validator)
        self.app.lineEdit_Nc.setValidator(validator)
        self.app.lineEdit_Ndop.setValidator(validator)
        self.app.lineEdit_Nv.setValidator(validator)
        self.app.lineEdit_tn.setValidator(validator)
        self.app.lineEdit_tp.setValidator(validator)

        self.app.lineEdit_grid.setValidator(validator)
        self.app.lineEdit_nvelleft.setValidator(validator)
        self.app.lineEdit_nvelright.setValidator(validator)
        self.app.lineEdit_pvelleft.setValidator(validator)
        self.app.lineEdit_pvelright.setValidator(validator)
        self.app.lineEdit_wfback.setValidator(validator)
        self.app.lineEdit_wffront.setValidator(validator)

        class DoubleValidator(QtWidgets.QStyledItemDelegate):
            def createEditor(self, parent, option, index):
                editor = super(DoubleValidator, self).createEditor(
                    parent, option, index)
                if isinstance(editor, QtWidgets.QLineEdit):
                    editor.setValidator(validator)
                return editor
        delegate = DoubleValidator(self.app.tableWidget_modellayout)

        self.app.tableWidget_modellayout.setItemDelegateForColumn(1, delegate)
        self.app.tableWidget_modellayout.setItemDelegateForColumn(2, delegate)

    def dpvSimulator(self):

        def makeDesign():

            try:
                n_points = int(self.app.lineEdit_grid.text())
                Snl = float(self.app.lineEdit_nvelleft.text())
                Snr = float(self.app.lineEdit_nvelright.text())
                Spl = float(self.app.lineEdit_pvelleft.text())
                Spr = float(self.app.lineEdit_pvelright.text())
            except ValueError:
                print("Value error in setup.")
                # TODO: print the tb
                return

            Ls = []
            Ns = []
            mats = []

            rows = self.app.tableWidget_modellayout.rowCount()
            try:
                assert rows > 0
            except AssertionError:
                print("No material defined.")
                return

            for i in range(rows):
                mats.append(
                    self.app.tableWidget_modellayout.item(i, 0).text())
                Ls.append(
                    float(self.app.tableWidget_modellayout.item(i, 1).text()))
                Ns.append(
                    float(self.app.tableWidget_modellayout.item(i, 2).text()))

            mats = [dpv.load_material(x) for x in mats]

            self.des = dpv.make_design(n_points=n_points,
                                       Ls=Ls,
                                       mats=mats,
                                       Ns=Ns,
                                       Snl=Snl, Snr=Snr, Spl=Spl, Spr=Spr)

        def simulate():
            # check design is ready then trigger simulator
            ls = self.app.comboBox_lightsource.currentText()
            if self.des is not None:
                self.results = dpv.simulate(self.des,
                                            ls=dpv.incident_light(ls))

                self.plt_iv, self.ax_iv = plotting.plot_iv_curve(
                    voltages=self.results["iv"][0],
                    currents=self.results["iv"][1],
                    gui=[self.plt_iv, self.ax_iv])

                self.plt_model, self.ax_model = plotting.plot_bars(
                    self.des,
                    gui=[self.plt_model, self.ax_model])

                self.plt_band, self.ax_band = plotting.plot_band_diagram(
                    self.des,
                    self.results["eq"], eq=True,
                    gui=[self.plt_band, self.ax_band])

                self.plt_charge, self.ax_charge = plotting.plot_charge(
                    self.des,
                    self.results["eq"],
                    gui=[self.plt_charge, self.ax_charge])

                self.graphcanvas_iv.draw_idle()
                self.graphcanvas_band.draw_idle()
                self.graphcanvas_charge.draw_idle()
                self.graphcanvas_model.draw_idle()

        self.app.pushButton_makedesign.clicked.connect(makeDesign)
        self.app.pushButton_simulate.clicked.connect(simulate)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    gui = MainWindow(window)
    window.show()
    sys.exit(app.exec_())
