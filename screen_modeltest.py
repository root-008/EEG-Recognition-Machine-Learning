from PyQt5 import QtCore, QtGui, QtWidgets
import createX_y
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import pickle
from tensorflow.keras.models import load_model
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1324, 697)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(550, 0, 211, 61))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.table_data = QtWidgets.QTableView(self.centralwidget)
        self.table_data.setGeometry(QtCore.QRect(10, 100, 1301, 261))
        self.table_data.setObjectName("table_data")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 65, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 400, 281, 31))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.line_veri_num = QtWidgets.QLineEdit(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_veri_num.setFont(font)
        self.line_veri_num.setObjectName("line_veri_num")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.line_veri_num)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 370, 291, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(10, 470, 431, 171))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.formLayoutWidget_2.setFont(font)
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.btn_ann = QtWidgets.QPushButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_ann.setFont(font)
        self.btn_ann.setObjectName("btn_ann")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.btn_ann)
        self.btn_knn = QtWidgets.QPushButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_knn.setFont(font)
        self.btn_knn.setObjectName("btn_knn")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.btn_knn)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.btn_dt = QtWidgets.QPushButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_dt.setFont(font)
        self.btn_dt.setObjectName("btn_dt")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.btn_dt)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.btn_rf = QtWidgets.QPushButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_rf.setFont(font)
        self.btn_rf.setObjectName("btn_rf")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.btn_rf)
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.btn_all = QtWidgets.QPushButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_all.setFont(font)
        self.btn_all.setObjectName("btn_all")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.btn_all)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(10, 440, 601, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(830, 370, 141, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.lbl_sonuc = QtWidgets.QLabel(self.centralwidget)
        self.lbl_sonuc.setGeometry(QtCore.QRect(850, 400, 451, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_sonuc.setFont(font)
        self.lbl_sonuc.setObjectName("lbl_sonuc")
        self.btn_data_ann = QtWidgets.QPushButton(self.centralwidget)
        self.btn_data_ann.setGeometry(QtCore.QRect(130, 67, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_data_ann.setFont(font)
        self.btn_data_ann.setObjectName("btn_data_ann")
        self.btn_data_classf = QtWidgets.QPushButton(self.centralwidget)
        self.btn_data_classf.setGeometry(QtCore.QRect(350, 67, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_data_classf.setFont(font)
        self.btn_data_classf.setObjectName("btn_data_classf")
        
        self.btn_kfold = QtWidgets.QPushButton(self.centralwidget)
        self.btn_kfold.setGeometry(QtCore.QRect(580, 67, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_kfold.setFont(font)
        self.btn_kfold.setObjectName("btn_kfold")
        
        self.btn_holdout = QtWidgets.QPushButton(self.centralwidget)
        self.btn_holdout.setGeometry(QtCore.QRect(810, 67, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_holdout.setFont(font)
        self.btn_holdout.setObjectName("btn_holdout")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1324, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.btn_data_ann.clicked.connect(self.btn_data_ann_click)
        self.btn_data_classf.clicked.connect(self.btn_data_classf_click)
        
        self.btn_kfold.clicked.connect(self.btn_kfold_click)
        self.btn_holdout.clicked.connect(self.btn_holdout_click)
        self.kfold = True
        
        self.btn_ann.clicked.connect(self.btn_ann_click)
        self.btn_knn.clicked.connect(self.btn_knn_click)
        self.btn_dt.clicked.connect(self.btn_dt_click)
        self.btn_rf.clicked.connect(self.btn_rf_click)
        self.btn_all.clicked.connect(self.btn_all_click)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Modeli Test Et"))
        self.label.setText(_translate("MainWindow", "Modeli Test Et"))
        self.label_2.setText(_translate("MainWindow", "Örnek Veriler"))
        self.label_3.setText(_translate("MainWindow", "Veri Numarası :"))
        self.label_4.setText(_translate("MainWindow", "Lütfen Tablodan Veri Numarası Seçiniz."))
        self.label_5.setText(_translate("MainWindow", "Yapay Sinir Ağı :"))
        self.btn_ann.setText(_translate("MainWindow", "Tahmin Et"))
        self.btn_knn.setText(_translate("MainWindow", "Tahmin Et"))
        self.label_6.setText(_translate("MainWindow", "K-En Yakın Komşu :"))
        self.btn_dt.setText(_translate("MainWindow", "Tahmin Et"))
        self.label_7.setText(_translate("MainWindow", "Karar Ağacı :"))
        self.btn_rf.setText(_translate("MainWindow", "Tahmin Et"))
        self.label_8.setText(_translate("MainWindow", "Rastgele Orman :"))
        self.btn_all.setText(_translate("MainWindow", "Tahmin Et"))
        self.label_9.setText(_translate("MainWindow", "Hepsi :"))
        self.label_10.setText(_translate("MainWindow", "Model Seçiniz veya hepsinin ortalamasına göre bir tahmin isterseniz Hepsi\'ni seçiniz."))
        self.label_11.setText(_translate("MainWindow", "Modelin Tahmini : "))
        self.lbl_sonuc.setText(_translate("MainWindow", "-"))
        self.btn_data_ann.setText(_translate("MainWindow", "Yapay Sinir Ağı için Verileri Listele"))
        self.btn_data_classf.setText(_translate("MainWindow", "Sınıflandırıcılar için Verileri Listele"))
        self.btn_kfold.setText(_translate("MainWindow", "K-fold Çapraz Doğrulama"))
        self.btn_holdout.setText(_translate("MainWindow", "Dışarda Tutma Doğrulama"))
        
    def btn_kfold_click(self):
        self.kfold = True
        if self.ann:
            self.label_2.setText('YSA-Kfold')
        else:
            self.label_2.setText('classf-Kfold')
        
    
    def btn_holdout_click(self):
        self.kfold = False
        if self.ann:
            self.label_2.setText('YSA-Holdout')
        else:
            self.label_2.setText('classf-Holdout')
    
    def btn_ann_click(self):
        self.lbl_sonuc.setText('-')
        model = load_model('savedModels/model_ann_holdout.keras')
        veri_num = int(self.line_veri_num.text()) - 1
        print(veri_num)
        data = self.X[veri_num]
        data = data.reshape(1, -1)
        # Modelin tahmini
        tahmin_probs = model.predict(data)
        tahmin_class = np.argmax(tahmin_probs)
        tahmin_probability = tahmin_probs[0, tahmin_class]  # Probability of the predicted class
        
        self.lbl_sonuc.setText(f"Sınıf: {tahmin_class}, Olasılık: {tahmin_probability:.2%}")
    
    def btn_knn_click(self):
        print(self.kfold)
        self.lbl_sonuc.setText('-')
        if self.kfold:
            with open('savedModels/model_knn_kfold.pkl', 'rb') as model_file:
                model, accuracies = pickle.load(model_file)
        else:
            with open('savedModels/model_rf_holdout.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
                
        veri_num = int(self.line_veri_num.text()) - 1
        data = self.X[veri_num]
        data = data.reshape(1, -1)
        tahmin = model.predict(data)
        self.lbl_sonuc.setText(f"Sınıf: {tahmin[0]}")
    
    def btn_dt_click(self):
        self.lbl_sonuc.setText('-')
        if self.kfold:
            with open('savedModels/model_dt_kfold.pkl', 'rb') as model_file:
                model, accuracies = pickle.load(model_file)
        else:
            with open('savedModels/model_dt_holdout.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
        veri_num = int(self.line_veri_num.text()) - 1
        data = self.X[veri_num]
        data = data.reshape(1, -1)
        tahmin = model.predict(data)
        self.lbl_sonuc.setText(f"Sınıf: {tahmin[0]}")
    
    def btn_rf_click(self):
        if self.kfold:
            with open('savedModels/model_rf_kfold.pkl', 'rb') as model_file:
                model, accuracies = pickle.load(model_file)
        else:
            with open('savedModels/model_rf_holdout.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
        veri_num = int(self.line_veri_num.text()) - 1
        data = self.X[veri_num]
        data = data.reshape(1, -1)
        tahmin = model.predict(data)
        self.lbl_sonuc.setText(f"Sınıf: {tahmin[0]}")
    
    def btn_all_click(self):
        self.lbl_sonuc.setText('-')
        
        with open('savedModels/model_rf_holdout.pkl', 'rb') as model_file:
            model_rf = pickle.load(model_file)
            
        with open('savedModels/model_dt_holdout.pkl', 'rb') as model_file:
             model_dt = pickle.load(model_file)
             
        with open('savedModels/model_knn_holdout.pkl', 'rb') as model_file:
             model_knn = pickle.load(model_file)
             
        veri_num = int(self.line_veri_num.text()) - 1
        data = self.X[veri_num]
        data = data.reshape(1, -1)
        
        tahmin_knn = model_knn.predict(data)
        tahmin_dt = model_dt.predict(data)
        tahmin_rf = model_rf.predict(data)
        
        result = np.argmax(np.sum([tahmin_knn, tahmin_dt, tahmin_rf], axis=0))
        self.lbl_sonuc.setText(f"Sınıf : {result}")
        
        
    
    def btn_data_ann_click(self):
        self.ann = True
        if self.kfold:
            self.label_2.setText('YSA-Kfold')
        else:
            self.label_2.setText('YSA-Holdout')
        X_train, X_val, X_test, y_train, y_val, y_test = createX_y.create_train_test_data_for_ann()
        X,y = X_test,y_test
        self.X = X
        self.y = y
        self.show_data_in_table()
    
    
    def btn_data_classf_click(self):
        self.ann = False
        if self.kfold:
            self.label_2.setText('classf-Kfold')
        else:
            self.label_2.setText('classf-Holdout')
        X_train, X_val, X_test, y_train, y_val, y_test = createX_y.create_train_test_data_for_classif()
        X,y = X_test,y_test.values
        self.X = X
        self.y = y
        self.show_data_in_table()
    
    def show_data_in_table(self):
        model = QStandardItemModel()
        
        # X ve y'den verileri ekleyin
        for i in range(len(self.X)):
            x_item = QStandardItem(str(self.X[i]))
            y_item = QStandardItem(str(self.y[i]))
            model.appendRow([x_item, y_item])

        self.table_data.setModel(model)
        self.table_data.setColumnWidth(0, 1146)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
