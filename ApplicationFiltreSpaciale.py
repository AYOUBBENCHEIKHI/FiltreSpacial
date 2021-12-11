import os
import sys
from idlelib.debugger_r import frametable

import wx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from _dlib_pybind11 import range
from scipy import ndimage
import cv2


class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame, self).__init__(parent,title = title,size = (820,500),style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
        self.localpath = os.getcwd()
        self.photoTxt = os.getcwd()+"/image/img.png"
        # pour afficher un image donne licon de frnitre
        self.windowiconfilepath = self.localpath +'/image/lena.png'
        self.SetIcon(wx.Icon(self.windowiconfilepath))
        # l appelle de panel
        self.panel = MyPanel(self)
        self.panel.onView(os.getcwd()+"/image/img.png")
        self.panel.afficherImage("/image/img.png")
        pretext0 = wx.StaticText(self.panel, label="---- Image Origenne ----", pos=(70, 410),
                                 style=wx.ALIGN_CENTER)
        #menu
        menuBar = wx.MenuBar()
    ########################"menu pour ovrire et quiti et Engregistrez#############################
        ovrireFile = wx.Menu()
        open = ovrireFile.Append(wx.ID_OPEN, "Open")
        save= ovrireFile.Append(wx.ID_SAVE, "Save")
        exit = ovrireFile.Append(wx.ID_EXIT,"E&xit\tCtrl-Q", "Exit",)
        menuBar.Append(ovrireFile, "&Open File")
        self.Bind(wx.EVT_MENU, self.exit,exit)
        self.Bind(wx.EVT_MENU, self.openFile,open)
        self.Bind(wx.EVT_MENU, self.save,save)

    ###############################################################################################
    #                            FILTRE PASS BAS SPACIALE                                         #
    ###############################################################################################

        filtreSpacialePB = wx.Menu()

        ##############   filtre Laplacien   ####################
        filtreLaplacien = filtreSpacialePB .Append(wx.ID_ANY, "Filtre Laplacien")
        self.Bind(wx.EVT_MENU, self.filtreLaplacien, filtreLaplacien)
        ##############   filtre Pyramidal   ####################
        filtrePyramidal = filtreSpacialePB .Append(wx.ID_ANY, "Filtre Pyramidal")
        self.Bind(wx.EVT_MENU, self.filtrePyramidal, filtrePyramidal)
        ##############   filtre Conique   ####################
        filtreConique = filtreSpacialePB .Append(wx.ID_ANY, "Filtre Conique")
        self.Bind(wx.EVT_MENU, self.filtreConique, filtreConique)

        ##############   filtre de la moyenne   ####################
        filtreMoyenne = wx.Menu()
        moyenne3 = filtreMoyenne.Append(wx.ID_ANY, "Moyenne 3")
        moyenne5 = filtreMoyenne.Append(wx.ID_ANY, "Moyenne 5")
        self.Bind(wx.EVT_MENU, self.filtreMoyenne3, moyenne3)
        self.Bind(wx.EVT_MENU, self.filtreMoyenne5, moyenne5)
        filtreSpacialePB.AppendSubMenu(filtreMoyenne,"Filtre Moyenn ")

        ##############   filtre de la Gaussien   ####################
        filtreGaussien = wx.Menu()
        gaussien3 = filtreGaussien.Append(wx.ID_ANY, "Gaussien 3")
        gaussien5 = filtreGaussien.Append(wx.ID_ANY, "Gaussien 5")
        self.Bind(wx.EVT_MENU, self.filtreGaussien3, gaussien3)
        self.Bind(wx.EVT_MENU, self.filtreGaussien5, gaussien5)
        filtreSpacialePB.AppendSubMenu(filtreGaussien,"Filtre Gaussien ")
        ##############   filtre de Median   ####################
        filtreMedian = wx.Menu()
        median3 = filtreMedian.Append(wx.ID_ANY, "Median  3")
        median5 = filtreMedian.Append(wx.ID_ANY, "Median  5")
        self.Bind(wx.EVT_MENU, self.filtreMedian3, median3)
        self.Bind(wx.EVT_MENU, self.filtreMedian5, median5)
        filtreSpacialePB.AppendSubMenu(filtreMedian,"Filtre Median ")

        menuBar.Append(filtreSpacialePB, "Filtre Spaciale PB")
    ###############################################################################################
    #                            FILTRE PASS HAUT SPACIALE                                         #
    ###############################################################################################
        filtreSpacialePH = wx.Menu()

        ##############   dtecter le conteur    ####################
        detecterConteur = filtreSpacialePH.Append(wx.ID_ANY, "Detecter Conteur ")
        self.Bind(wx.EVT_MENU, self.detecterConteur, detecterConteur)
        ##############    conteur de Gradient     ####################
        conteurGradient = filtreSpacialePH.Append(wx.ID_ANY, "Conteur de Gradient")
        self.Bind(wx.EVT_MENU, self. conteurGradient,  conteurGradient)
        ##############    conteur de Sobel     ####################
        conteurSobel = filtreSpacialePH.Append(wx.ID_ANY, "Conteur Sobel")
        self.Bind(wx.EVT_MENU, self.conteurSobel, conteurSobel)
        ##############    conteur de Prewitt     ####################
        conteurPrewitt  = filtreSpacialePH.Append(wx.ID_ANY, "Conteur Prewitt ")
        self.Bind(wx.EVT_MENU, self.conteurPrewitt, conteurPrewitt)
        ##############    conteur de Laplacien     ####################
        conteurLaplacien = filtreSpacialePH.Append(wx.ID_ANY, "Conteur Laplacien ")
        self.Bind(wx.EVT_MENU, self.conteurLaplacien, conteurLaplacien)
        ##############   Conteur de la moyenne   ####################
        conteurMoyenne = wx.Menu()
        moyenne3 = conteurMoyenne.Append(wx.ID_ANY, "Moyenne 3")
        moyenne5 = conteurMoyenne.Append(wx.ID_ANY, "Moyenne 5")
        self.Bind(wx.EVT_MENU, self.conteurMoyenne3, moyenne3)
        self.Bind(wx.EVT_MENU, self.conteurMoyenne5, moyenne5)
        filtreSpacialePH.AppendSubMenu(conteurMoyenne,"Conteur de Moyenn ")

        menuBar.Append(filtreSpacialePH, "Filtre Spaciale PH")

        ###############################################################################################
        #                           HISTOGRAMMME                                                     #
        ###############################################################################################
        histogramme = wx.Menu()
        his = histogramme.Append(wx.ID_ANY, "Histogramme")
        self.Bind(wx.EVT_MENU,self.histogramme,his)
        menuBar.Append(histogramme,"histogramme")
        self.SetMenuBar(menuBar)

    def exit(self,event):
        self.Close()
    def openFile(self,event):
        wildcard = "JPEG files (*.jpg)|*.*"
        dialog = wx.FileDialog(None, "Choose a file", wildcard=wildcard, style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt = dialog.GetPath()
        dialog.Destroy()
        self.panel.onView(self.photoTxt)
        print(self.photoTxt)
    def save(self,event):
        print("save image ")

    ######################################################################
    #               Filtre speciale passe bas                            #
    ######################################################################

    def filtreLaplacien(self,event):
        t = self.panel.filtreLaplacien3D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre l'aplacient ----", pos=(590, 410), style=wx.ALIGN_CENTER)
        print("---- Filtre l'aplacient ----")
    def filtrePyramidal(self,event):
        t = self.panel.filtrePyramidal5D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Pyramidal----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Pyramidal ----")
    def filtreConique(self,event):
        t = self.panel.filtreConique5D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Pyramidal----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Conique ----")
    def filtreMoyenne3(self,event):
        t = self.panel.filtreMoyenne3D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Moyenne 3 ----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Moyenne 3 ----")
    def filtreMoyenne5(self,event):
        t = self.panel.filtreMoyenne5D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Moyenne 5----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Moyenne 5 ----")
    def filtreGaussien3(self,event):
        t = self.panel.filtreGaussien3D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Gaussien 3----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Gaussien 3 ----")
    def filtreGaussien5(self,event):
        t = self.panel.filtreGaussien5D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Gaussien 5----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Gaussien 5 ----")
    def filtreMedian3(self,event):
        t = self.panel.filtreMedian3D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Median 3----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Median 3 ----")
    def filtreMedian5(self,event):
        t = self.panel.filtreMedian5D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Filtre Median 5----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Filtre Median 5 ----")
    ######################################################################
    #               Filtre speciale passe bas                            #
    ######################################################################
    def detecterConteur(self,event):
        t = self.panel.DetectionDeContours(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel,label = "---- detecter Conteur ----", pos=(590,410), style = wx.ALIGN_CENTER)
        print("---- detecter Conteur ----")
    def conteurGradient(self,event):
        t = self.panel.conteurGradient(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Conteur Gradient----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Conteur Gradient ----")
    def conteurSobel(self,event):
        t = self.panel.conteurSobelPrewitt(self.photoTxt,2)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Conteur Sobel----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Conteur Sobel ----")
    def conteurPrewitt(self,event):
        t = self.panel.conteurSobelPrewitt(self.photoTxt,1)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Conteur Prewitt----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Conteur Prewitt ----")
    def conteurLaplacien(self,event):
        t = self.panel.conteurLaplacien(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Conteur Laplacien----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Conteur Laplacien ----")
    def conteurMoyenne3(self,event):
        t = self.panel.DetectionConteurMoyenne3D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Conteur Moyenne 3 ----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Conteur Moyenne 3 ----")
    def conteurMoyenne5(self,event):
        t = self.panel.DetectionConteurMoyenne5D(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- Conteur Moyenne 5 ----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- Conteur Moyenne 5 ----")

    #--------------------------HISTOGRAMMME-------------------------
    def histogramme(self,event):
        t = self.panel.histogramme(self.photoTxt)
        self.panel.afficherImage(t[0])
        pretext1 = wx.StaticText(self.panel, label="---- HISTOGRAMMME ----", pos=(590, 410),
                                 style=wx.ALIGN_CENTER)
        print("---- HISTOGRAMMME ----")


class MyPanel(wx.Panel):
    def __init__(self,parent):
        super(MyPanel, self).__init__(parent,pos =(0,0),size=(400,500))


    def onView(self,filepath):

        imageStaticBitmap = wx.StaticBitmap(self)
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
             NewW = 400
             NewH = 400 * H / W
        else:
             NewH = 400
             NewW = 400 * W / H
        img = img.Scale(int(NewW), int(NewH))
        imageStaticBitmap.SetBitmap(wx.Bitmap(img))
        imageStaticBitmap.SetPosition((0, 0))
    def afficherImage(self,name):
        imgGeneral = os.getcwd() + name
        imageStaticBitmap = wx.StaticBitmap(self)

        img = wx.Image(imgGeneral, wx.BITMAP_TYPE_ANY)
        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = 400
            NewH = 400* H / W
        else:
            NewH = 400
            NewW = 400 * W / H

        img = img.Scale(int(NewW), int(NewH))
        imageStaticBitmap.SetBitmap(wx.Bitmap(img))
        imageStaticBitmap.SetPosition((401,0))
    ######################################################################
    #               Filtre speciale passe bas                            #
    ######################################################################


    ##########################Filtre Lplacien ##############################
    def filtreLaplacien3D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1
        gaus = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        g = self.produitDeConvolution3(g,gaus)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    #############################################################################

    ##########################Filtre Pyramidal ################################
    def filtrePyramidal5D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1

        gaus = np.array([[1,2,3,2,1],
                         [2,4,6,4,2],
                         [3,6,8,6,8],
                         [2,4,6,4,2],
                         [1,2,3,2,1]])
        gauss = (1/81)*gaus
        g = self.produitDeConvolution5(g,gauss)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    #############################################################################


    ##########################Filtre Pyramidal ##################################
    def filtreConique5D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1

        gaus = np.array([[0,0,1,0,0],
                         [0,2,2,2,0],
                         [1,2,5,2,1],
                         [0,2,2,2,0],
                         [0,0,1,0,0]])
        gauss = (1/25)*gaus
        g = self.produitDeConvolution5(g,gauss)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    ############################################################################

    ##########################Filtre de la moyenne ##############################

    def filtreMoyenne3D(self,name):
        img = cv2.imread(name)
        # _, _, h = cv2.split(img)#BGR
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # controle de la luminusitez isi P=1
        moy = (1 / 9) * np.ones((3, 3))

        g = self.produitDeConvolution3(g, moy)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png", hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    def filtreMoyenne5D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1
        moy = (1 / 25) * np.ones((5,5))
        g = self.produitDeConvolution5(g,moy)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    ##############################################################################
    ########################## Filtre de la Gaussien ##############################
    def filtreGaussien3D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1

        gaus = np.array([[1,2,1],[2,4,2],[1,2,1]])
        gauss = (1/16)*gaus
        g = self.produitDeConvolution3(g,gauss)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    def filtreGaussien5D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1

        gaus = np.array([[1,4,6,4,1],
                         [4,16,24,16,4],
                         [6,24,36,24,6],
                         [4,16,24,16,4],
                         [1,4,6,4,1]])
        gauss = (1/256)*gaus
        g = self.produitDeConvolution5(g,gauss)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t

    ##############################################################################
    ########################## Filtre de la Median ##############################

    def filtreMedian3D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1
        if len(g.shape) == 3:
            l, c, channels = g.shape[:3]
        else:
            l, c = g.shape[:2]
            channels = 1
        for i in range(1,l-1):
            for j in range(1,c-1):
                f = g[i-1:i+2,j-1:j+2]
                conteur  = np.array([f[0][:],f[1][:],f[2][:]])
                conteurtri = np.sort(conteur)
                median = np.median(conteurtri)
                g[i, j]=median
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    def filtreMedian5D(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        # controle de la luminusitez isi P=1
        if len(g.shape) == 3:
            l, c, channels = g.shape[:3]
        else:
            l, c = g.shape[:2]
            channels = 1
        for i in range(2,l-2):
            for j in range(2,c-2):
                f = g[i-2:i+3,j-2:j+3]
                conteur  = np.array([f[0][:],f[1][:],f[2][:],f[3][:],f[4][:]])
                conteurtri = np.sort(conteur)
                median = np.median(conteurtri)
                g[i, j]=median
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png",hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    ###################################################################################
    ######################################################################
    #               Filtre speciale passe HAUT                            #
    ######################################################################

    def DetectionDeContours(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgFiltre = self.filtreMedian5D(name)
        path = os.getcwd() + imgFiltre[0]
        print(path)
        imgFilrez = cv2.imread(path)
        imgFilrezg = cv2.cvtColor(imgFilrez, cv2.COLOR_BGR2GRAY )
        conteur = g - imgFilrezg
        hr = conteur.astype(np.uint8)
        cv2.imwrite("imgEn/imageConteur.png",hr)
        t = ["/imgEn/imageConteur.png"]
        return t

    def conteurGradient(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m, n = img.shape[:2]
        output = np.zeros((m,n))
        outputhor =  np.zeros((m,n))
        outputver =  np.zeros((m,n))
        maskhor = np.array([[0, 0, 0],
                            [-1, 0, 1],
                            [0, 0, 0]])
        maskver =np.array([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]])
        i=3
        for i in range(m - 3):
            j=4
            for j in range(n - 3):
                k=1
                for k in range(3):
                    l=1
                    for l in range(3):
                         outputhor[i, j] = outputhor[i, j] + g[i - k, j - l] * maskhor[k, l]
                         outputver[i, j] = outputver[i, j] + g[i - k, j - l] * maskver[k, l]

        i = 3
        for i in range(m - 3):
            j = 4
            for j in range(n - 3):
                output[i, j] = np.sqrt(outputhor[i, j] * outputhor[i, j] + outputver[i, j] * outputver[i, j])
        hr = output.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png", hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    def conteurSobelPrewitt(self,name,c):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m, n = img.shape[:2]
        output = np.zeros((m,n))
        outputhor =  np.zeros((m,n))
        outputver =  np.zeros((m,n))
        maskhor = np.array([[1, c, 1],
                            [0, 0, 0],
                            [-1, -c, -1]])
        maskver =np.array([[1, 0, -1],
                           [c, 0, -c],
                           [1, 0, -1]])
        i=3
        for i in range(m - 3):
            j=4
            for j in range(n - 3):
                k=1
                for k in range(3):
                    l=1
                    for l in range(3):
                         outputhor[i, j] = outputhor[i, j] + g[i - k, j - l] * maskhor[k, l]
                         outputver[i, j] = outputver[i, j] + g[i - k, j - l] * maskver[k, l]

        i = 3
        for i in range(m - 3):
            j = 4
            for j in range(n - 3):
                output[i, j] = np.sqrt(outputhor[i, j] * outputhor[i, j] + outputver[i, j] * outputver[i, j])
        hr = output.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png", hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    def conteurLaplacien(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m, n = img.shape[:2]
        output = np.zeros((m,n))
        outputhor =  np.zeros((m,n))
        outputver =  np.zeros((m,n))
        maskhor = np.array([[1, -2, 1],
                            [0, 0, 0],
                            [0, 0, 0]])
        maskver =np.array([[0, 1, 0],
                           [0, -2, 0],
                           [0, 1, 0]])
        i = 3
        for i in range(m - 3):
            j = 4
            for j in range(n - 3):
                k = 1
                for k in range(3):
                    l = 1
                    for l in range(3):
                         outputhor[i, j] = outputhor[i, j] + g[i - k, j - l] * maskhor[k, l]
                         outputver[i, j] = outputver[i, j] + g[i - k, j - l] * maskver[k, l]

        i = 3
        for i in range(m - 3):
            j = 4
            for j in range(n - 3):
                output[i, j] = np.sqrt(outputhor[i, j] * outputhor[i, j] + outputver[i, j] * outputver[i, j])
        hr = output.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png", hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    def DetectionConteurMoyenne3D(self, name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        moy2 = (1 / 9) * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        g = self.produitDeConvolution3(g, moy2)
        hr = g.astype(np.uint8)
        cv2.imwrite("imgEn/imagefiltreMoyenne.png", hr)
        t = ["/imgEn/imagefiltreMoyenne.png"]
        return t
    def DetectionConteurMoyenne5D(self, name):
         img  = cv2.imread(name)
         g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         moy2 = (1 / 25) * np.array([[-1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1],
                                    [-1, -1, 24, -1, -1],
                                    [-1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1]])
         g = ndimage.convolve(g, moy2)
         hr = g.astype(np.uint8)
         cv2.imwrite("imgEn/imagefiltreMoyenne.png", hr)
         t = ["/imgEn/imagefiltreMoyenne.png"]
         return t
    #-------------------histogramme-------------------------
    def histogramme(self,name):
        img = cv2.imread(name)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vals = img.mean(axis=2).flatten()
        patches = plt.hist(vals, 255)
        plt.savefig("imgEn/histogramme.png")
        t = ["/imgEn/histogramme.png"]
        return t
        #plt.show()

    # produite de convolution 3D est 5D
    def produitDeConvolution3(self, H, I):
        if len(H.shape) == 3:
            l, c, channels = H.shape[:3]
        else:
            l, c = H.shape[:2]
            channels = 1
        for i in range(1, l - 1):
            for j in range(1, c - 1):
                f = H[i - 1:i + 2, j - 1:j + 2]
                im = f * I
                som = np.sum(im)
                H[i, j] = som
        return H

    def produitDeConvolution5(self, H, I):
        if len(H.shape) == 3:
            l, c, channels = H.shape[:3]
        else:
            l, c = H.shape[:2]
            channels = 1
        for i in range(2, l - 2):
            for j in range(2, c - 2):
                f = H[i - 2:i + 3, j - 2:j + 3]
                im = f * I
                som = np.sum(im)
                H[i, j] = som
        return H

class MyApp(wx.App):
    def OnInit(self):
        f = MyFrame(None,"les filre Spictrale et Spaciale")
        #frame = wx.Frame(None, style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        f.Show()
        return True


app = MyApp()
app.MainLoop()