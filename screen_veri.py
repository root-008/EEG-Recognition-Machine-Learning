from PyQt5 import QtCore, QtGui, QtWidgets
import createX_y
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1270, 888)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(470, 0, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.tbl_train_data = QtWidgets.QTableView(self.centralwidget)
        self.tbl_train_data.setGeometry(QtCore.QRect(10, 40, 1251, 141))
        self.tbl_train_data.setObjectName("tbl_train_data")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.tbl_test_data = QtWidgets.QTableView(self.centralwidget)
        self.tbl_test_data.setGeometry(QtCore.QRect(10, 220, 1251, 141))
        self.tbl_test_data.setObjectName("tbl_test_data")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 200, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 540, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(220, 560, 401, 281))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("gorseller/veri_isleme/sinifdagilimi_once.png"))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(620, 560, 401, 281))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap("gorseller/veri_isleme/sinifdagilimi_sonra.png"))
        self.label_6.setObjectName("label_6")
        self.tbl_val_data = QtWidgets.QTableView(self.centralwidget)
        self.tbl_val_data.setGeometry(QtCore.QRect(10, 390, 1251, 141))
        self.tbl_val_data.setObjectName("tbl_val_data")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(10, 370, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1270, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        X_train,X_val,X_test,y_train,y_val,y_test = createX_y.create_train_test_data_for_classif()
        
        self.show_data_in_table(table=self.tbl_train_data,X=X_train,y=y_train.values)
        self.show_data_in_table(table=self.tbl_test_data,X=X_test,y=y_test.values)
        self.show_data_in_table(table=self.tbl_val_data,X=X_val,y=y_val.values)
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VERİ SETİ İŞLEMLER"))
        self.label.setText(_translate("MainWindow", "VERİ SETİ İŞLEMLER"))
        self.label_2.setText(_translate("MainWindow", "Train Data"))
        self.label_3.setText(_translate("MainWindow", "Test Data"))
        self.label_4.setText(_translate("MainWindow", "Undersampling"))
        self.label_7.setText(_translate("MainWindow", "Validation Data"))
        
    def show_data_in_table(self,table,X,y):
        model = QStandardItemModel()
        
        # X ve y'den verileri ekleyin
        for i in range(len(X)):
            x_item = QStandardItem(str(X[i]))
            y_item = QStandardItem(str(y[i]))
            model.appendRow([x_item, y_item])
    
        table.setModel(model)
        table.setColumnWidth(0, 1050)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
