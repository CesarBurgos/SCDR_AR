'''
 *** Sistema Clasificador de Daño Radiologico

    Ing. Eduardo Cesar Camargo Burgos
    Estudiante de la MIIDT 9° Generación
    Versión 3.0 - Marzo 2023
''' 
# ________ Para ambas funciones captura y procesamiento de fotografías
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font, PhotoImage, Checkbutton
from PIL import Image, ImageTk
from datetime import datetime
import threading
import os, sys, glob
from sys import platform
import ctypes 
import subprocess
import imutils
import functools
#import math # ---- X
import cv2 # --- Manipulación de imágenes con Python (PDI)

# ________ Para la captura de fotografías
import serial, time
import serial.tools.list_ports

# ________ Para el procesamiento de las fotografías
#import mediapipe as mp
#import yolov5

#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt 
#import matplotlib.animation as animation
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from scipy.spatial import distance

#from reportlab.pdfgen import canvas
#from reportlab.lib.pagesizes import A4

# _____________________________________________ VARIABLES

# Lista de información a utilizar en la opción seleccionada
list_opcs = []

# _______ Variables globales 
# Variable usa en la identificación de Subventas (ventanas emergentes de tkinter)
global SubVent, SubVent2, SubVent3, detener_hilo
SubVent, SubVent2, SubVent3 = None, None, None
detener_hilo = False


# PARA DATASET
num_img_dataset = [0,0]

paletaColores = ['#355c7d', '#A9D4F1', '#BFDFF5', '#D4EAF8', '#EAF4FC', '#FFFFFF']

# ************************************* Objeto que administra la conexión con la tarjeta Arduino
class arduino:
    # Información del arduino
    inf_arduino = None

    def __init__(self):
        self.inf_arduino = []

    # Función para detección y conexión con tarjeta Arduino
    def inicializar(self):
        ports = list(serial.tools.list_ports.comports())

        self.inf_arduino.clear()
        for p in ports:
            if('Arduino' in p[1]) or ('CH340' in p[1]):
                self.inf_arduino.append(p[0]) # Puerto COM
                self.inf_arduino.append(p[1]) # Descripción tarjeta Arduino
                self.inf_arduino.append(9600) # Frecuencia de transmisión
    
                arduino = serial.Serial(self.inf_arduino[0], self.inf_arduino[2])
                time.sleep(2)                                                      # Tiempo de espera debido al reseteo de la tarjeta Arduino
                self.inf_arduino.append(arduino)                                   # Conexión con Arduino
                break

    # Función que permite validar si existe una conexión con la tarjeta Arduino conectada
    def deteccion(self):
        # Validando si el arduino NO se encuentra conectado...
        if self.inf_arduino:
            if not self.inf_arduino[3]:
                self.inf_arduino.clear()
                return False
            else:
                return True
        else:
            return False

    # Función que envia la señal para accionar el relevador el cual haga encender las luces led o laser, depende del valor enviado por opc
    def envio_Arduino(self, opc):
        self.inf_arduino[3].write(str.encode(opc))



# ************************************* Objeto que administra, controla la captura de las imagenes obtenidas por medio de la cámara
class capturaImgs:
    # Información de la cámara
    inf_cam = None
    # Control de botones
    btns = None
    # Bandera que indicara si ha sido capturada una imagen
    band_captura = None
    # Bandera que indicara si ha sido seleccionada la ruta de almacenamiento de las imágenes
    band_selecc_ubicacion = None
    # Tarjeta arduino
    tarjeta = None
    # Detección de manos - Mediapipe
    mano = None

    def __init__(self):
        self.inf_cam = []
        self.btns = []
        self.band_captura = False
        self.band_selecc_ubicacion = False
        self.tarjeta = arduino()

    # Comprobación de dispostivos (cámara y Arduino) conectados correctamente para realizar la captura de fotgrafías
    def comprobacionDispositivos(self):
        global Continuar

        # Inhabilitando botón continuar
        Continuar.grid_forget()

        try:
            # BUSCAR COMO SELECCIONAR LA CAMARA QUE ESTA CONECTADA POR USB # ---- **

            # Objeto de OpenCV para la captura de fotos por medio de la cámara
            camara = cv2.VideoCapture(0, cv2.CAP_DSHOW) # ---------------------- **

            #________________ Configurando paramétros de la cámara
            camara.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # No realizar un autoenfoque
            camara.set(cv2.CAP_PROP_FPS, 30)               # Captura de imagenes cada 30 Frames

            #________________ Información de la cámara (Lista)
            # Posición 0 --- Objeto de OpenCV que captura imagen por medio de la cámara
            self.inf_cam.append(camara)
            # Posición 1 --- Hilo que muestra la imagen obtenida por la cámara en tiempo real
            self.inf_cam.append(threading.Thread(target = lambda:self.imagenCamara(), name = 'Camara'))
            # Posición 2 --- 
            #self.inf_cam.append(0)

            # Modificando el ancho de la imagen
            camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # Modificando el largo de la imagen
            camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # Posición 3 --- Objeto de la imagen obtenida por CV2 - Modificación de frames capturados
            self.inf_cam.append(camara)
            
            # Validando si la cámara conectada esta dando imagen
            if self.inf_cam[0].isOpened():
                #Iniciando tarjeta Arduino conectada
                self.tarjeta.inicializar()

                #Detección de Arduino conectada
                if self.tarjeta.deteccion():
                    messagebox.showinfo(' ◢ Conexión con dispositivos: ', 'Cámara ---- Conectada.\n\n Tarjeta Arduino ---- Conectada. \n\n A continuación seleccione la ubicación del almacenamiento de las fotografías.')
                else:
                    messagebox.showwarning(' ◢ Conexión con dispositivos: ', 'Cámara ---- Conectada.\n\n *Tarjeta Arduino ---- Sin Conexión. \n\n A continuación seleccione la ubicación del almacenamiento de las fotografías.')

                #Selección de la ubicación donde serán almacenadas las imágenes capturadas.
                self.band_selecc_ubicacion = self.ubicacion_almacenamiento_fotos()

                if self.band_selecc_ubicacion:
                    # Iniciando el hilo que muestra la imagen obtenida por la cámara
                    self.inf_cam[1].start()

                    # Mostrar ventanas del control de captura
                    self.mostrar_control_captura()
                else:
                    self.inf_cam[0].release()
                    self.inf_cam.clear()
            else:
                self.inf_cam[0].release()
                self.inf_cam.clear()
                messagebox.showinfo(' ◢ Conexión con dispositivos: ', '*Cámara: ---- Sin Conexión.\n\nVerifique la conexión o estado de su Camara e intentelo nuevamente.\n\n')
                Continuar.grid(column = 2, row = 10, padx = (10, 10), pady = (10, 10), sticky = 'sew', columnspan = 2)
        except Exception as err:
            if self.inf_cam and self.inf_cam[0]:
 

                self.inf_cam[0].release()

            self.inf_cam.clear()
            messagebox.showinfo(' ◢ Conexión con dispositivos: ', '*Cámara: ---- Sin Conexión. \n\nVerifique la conexión o estado de su Camara e intentelo nuevamente.\n\n'+str(err))
            Continuar.grid(column = 2, row = 10, padx = (10, 10), pady = (10, 10), sticky = 'sew', columnspan = 2)

    # Función - Selección de la ubicación de almacenamiento de imágenes (jpg) y resultados (archivo CSV) obtenidos.
    def ubicacion_almacenamiento_fotos(self):
        try:
            # Creando carpeta
            os.mkdir(list_opcs[0])
            return True
        except:
            return False

    #______________________________________________________________ Funciones de ventana Captura de fotografias
    # Función que indicará que serán capturadas fotografías a la mano Derecha o Izquierda del sujeto
    def indicado_mano(self, event, indicador):
        seleccion = indicador.get()

        if seleccion[0] == 'I':
            list_opcs[4] = 'I'
        else:
            list_opcs[4] = 'D'

        Etiqueta6.config(text = ' * ¿La mano '+indicador.get()+' es DOMINANTE?: ')

    # Función que indicará cual mano es la dominate del paciente
    def indicando_mano_dom(self, ind_dom):
        global Etiqueta6, Dominante, NoDominante, MiniMenu3

        if ind_dom == 'Dom':
            list_opcs[5] = (list_opcs[4] + '_' + ind_dom)
        else:
            if list_opcs[4] == 'D':
                list_opcs[5] = ('I_Dom')
            else:
                list_opcs[5] = ('D_Dom')

        Etiqueta6.grid_forget()
        Dominante.grid_forget()
        NoDominante.grid_forget()

        # Mostrando botones de captura
        MiniMenu3.grid(column = 0, row = 6, padx = (0, 0), pady = (8, 8), sticky = 'nsew', columnspan = 4)

    # Función que indicará que luz desea activar el sistema
    def indicado_luz(self, event, indicador):
        seleccion = indicador.get()

        try:
            if seleccion[-1] == 'D': #led
                global SubVent3
                self.tarjeta.envio_Arduino('O')
                time.sleep(1)

                if SubVent3 is not None:
                    SubVent3.destroy()
                    SubVent3 = None
            else:
                self.tarjeta.envio_Arduino('I')
                time.sleep(1)
                self.ventana_ctrl_laser()
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

 
    # Función que solo ocultará a la ventana que controla a la luz laser, ya que esto permite mantener el estado de las luces encendidas/apagadas
    def ocultar_panel_laser(self):
        global SubVent3, SubVent2
        SubVent3.withdraw()
        SubVent2.grab_set()

    # Función que crea la ventana que tendrá el control del encendido/apagado de la luz laser
    def ventana_ctrl_laser(self):
        global SubVent3
        
        if SubVent3 is None:
            SubVent3 = tk.Toplevel(raiz)
            SubVent3.focus_force()
            SubVent3.grab_set()
            SubVent3.grab_release()
            SubVent3.resizable(0, 0)
            SubVent3.title('SCDR - Control de capturas: CONTROL DE LUZ LASER [3]')
            SubVent3.resizable(False, False)
            #SubVent3.geometry('{}x{}+{}+{}'.format(int(DatosEjec[0]/3), int(DatosEjec[0]/4)+40, int((DatosEjec[0]/2)-int(DatosEjec[0]/4)), int((DatosEjec[1]/2) - int(DatosEjec[0]/4))))
            #SubVent3.geometry('{}x{}+{}+{}'.format(int(DatosEjec[0]/3), int(DatosEjec[0]/2), int((DatosEjec[0]/2)-int(DatosEjec[0]/4)), int((DatosEjec[1]/2) - int(DatosEjec[0]/4))))

            SubVent3.geometry('{}x{}+{}+{}'.format(ventana[14], ventana[15], ventana[16], ventana[17]))
            SubVent3.config(bg = 'white', bd = 9, relief = 'groove')
            SubVent3.protocol('WM_DELETE_WINDOW', lambda:self.ocultar_panel_laser())

            # --------------------- Panel de botones - Control de luz laser/led 
            botones = tk.Frame(SubVent3)
            #Configuración de columnas
            botones.grid_columnconfigure(0, weight = 1, uniform = 'fig')
            botones.grid_rowconfigure(0, weight = 1)
            botones.grid_rowconfigure(1, weight = 1, uniform = 'fig')
            botones.grid_rowconfigure(2, weight = 1)
            # --------------------- 

            Etiqueta = tk.Label(botones, 
                                text = ' Sistema Clasificador de Daño Radiológico por AR \n Control de luces lasers', 
                                bg = '#1A5276', fg = 'white', font = ('Microsoft YaHei UI', 13, 'bold'))

            # _________________________________________________________________________________________ Plantilla de botones (luces led)

            ctrlluces = tk.LabelFrame(botones,
                                     text = '',
                                     bd = 1, bg = 'white', fg = 'black',
                                     font = ('Microsoft YaHei UI',DatosEjec[4],'bold'))

            #Configuración de columnas
            for i in range(0, 9):
                ctrlluces.grid_columnconfigure(i, weight = 1, uniform = 'fig')

            indicador0 = tk.Label(ctrlluces,
                                 text=' Seleccione y haga clic sobre el numero del laser que desea accionar.',
                                 bg = 'white', fg='black', 
                                 font = ('Microsoft YaHei UI', DatosEjec[4]))

            # 1 FILA DE LUCES LED - ROJAS
            btn1 = tk.Button(ctrlluces, text = '1', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('11', btn1))
            btn2 = tk.Button(ctrlluces, text = '2', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('12',btn2))
            btn3 = tk.Button(ctrlluces, text = '3', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('13',btn3))

            # 2 FILA DE LUCES LED - ROJAS
            btn4 = tk.Button(ctrlluces, text = '4', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('18',btn4))
            btn5 = tk.Button(ctrlluces, text = '5', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('6',btn5))
            btn6 = tk.Button(ctrlluces, text = '6', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('7',btn6))
            btn7 = tk.Button(ctrlluces, text = '7', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('8',btn7))
            btn8 = tk.Button(ctrlluces, text = '8', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('9',btn8))
            btn9 = tk.Button(ctrlluces, text = '9', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('10',btn9))
            btn10 = tk.Button(ctrlluces, text = '10', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('5',btn10))
            
            # 3 FILA DE LUCES LED - ROJAS
            btn11 = tk.Button(ctrlluces, text = '11', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('25',btn11))
            btn12 = tk.Button(ctrlluces, text = '12', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('17',btn12))
            btn13 = tk.Button(ctrlluces, text = '13', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('16',btn13))
            btn14 = tk.Button(ctrlluces, text = '14', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('15',btn14))
            btn15 = tk.Button(ctrlluces, text = '15', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('14',btn15))
            btn16 = tk.Button(ctrlluces, text = '16', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('4',btn16))
            btn17 = tk.Button(ctrlluces, text = '17', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('19',btn17))
            
            # 4 FILA DE LUCES LED - ROJAS
            btn18 = tk.Button(ctrlluces, text = '18', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('32',btn18))
            btn19 = tk.Button(ctrlluces, text = '19', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('24',btn19))
            btn20 = tk.Button(ctrlluces, text = '20', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('23',btn20))
            btn21 = tk.Button(ctrlluces, text = '21', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('22',btn21))
            btn22 = tk.Button(ctrlluces, text = '22', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('21',btn22))
            btn23 = tk.Button(ctrlluces, text = '23', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('20',btn23))
            btn24 = tk.Button(ctrlluces, text = '24', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('26',btn24))
            
            # 5 FILA DE LUCES LED - ROJAS 
            btn25 = tk.Button(ctrlluces, text = '25', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('34',btn25))
            btn26 = tk.Button(ctrlluces, text = '26', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('31',btn26))
            btn27 = tk.Button(ctrlluces, text = '27', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('30',btn27))
            btn28 = tk.Button(ctrlluces, text = '28', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('29',btn28))
            btn29 = tk.Button(ctrlluces, text = '29', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('28',btn29))
            btn30 = tk.Button(ctrlluces, text = '30', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('27',btn30))
            btn31 = tk.Button(ctrlluces, text = '31', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('33',btn31))

            # 6 FILA DE LUCES LED - ROJAS
            btn32 = tk.Button(ctrlluces, text = '32', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('36',btn32))
            btn33 = tk.Button(ctrlluces, text = '33', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('38',btn33))
            btn34 = tk.Button(ctrlluces, text = '34', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('37',btn34))
            btn35 = tk.Button(ctrlluces, text = '35', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('35',btn35))
            
            # 7 FILA DE LUCES LED - ROJAS
            btn36 = tk.Button(ctrlluces, text = '36', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('42',btn36))
            btn37 = tk.Button(ctrlluces, text = '37', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('41',btn37))
            btn38 = tk.Button(ctrlluces, text = '38', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('40',btn38))
            btn39 = tk.Button(ctrlluces, text = '39', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('39',btn39))
            
            # 8 FILA DE LUCES LED - ROJAS
            btn40 = tk.Button(ctrlluces, text = '40', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('44',btn40))
            btn41 = tk.Button(ctrlluces, text = '41', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('43',btn41))
            
            btn42 = tk.Button(ctrlluces, text = '42', cursor = 'hand2', relief = tk.RIDGE, borderwidth = 1, fg='white', bg='gray50', font = ('Microsoft YaHei UI', DatosEjec[4],'bold'), command = lambda:self.acccion_luz('3',btn42))
           
            # _________________________________________________________________________________________ Indicador de encendido/apagado

            indic_luces = tk.LabelFrame(ctrlluces, text = ' ', bd = 0, bg = 'white')

            #Configuración de columnas
            for i in range(0, 4):
                indic_luces.grid_columnconfigure(i, weight = 1, uniform = 'fig')

            btn_1 = tk.Button(indic_luces, text = ' ', relief = tk.RIDGE, fg = 'white',
                                  borderwidth = 1, activebackground='#C70039', bg='#C70039', 
                                  font = ('Microsoft YaHei UI', DatosEjec[4]), command = None)

            indc1 = tk.Label(indic_luces, text = ' ENCENDIDO ', fg = 'black',
                                  bg='white', font = ('Microsoft YaHei UI', DatosEjec[4]))

            btn_2 = tk.Button(indic_luces, text = ' ', relief = tk.RIDGE, fg = 'black',
                                  borderwidth = 1, activebackground='gray70', bg='gray70', 
                                  font = ('Microsoft YaHei UI', DatosEjec[4]), command = None)

            indc2 = tk.Label(indic_luces, text = ' APAGADO ', fg = 'black',
                                  bg='white', font = ('Microsoft YaHei UI', DatosEjec[4]))

            # ___________________________________________________________________________________________

            ctrlluces2 = tk.LabelFrame(botones,
                                     text = ' Botones de encendido rápido ',
                                     bd = 1, bg = 'white', fg = 'black',
                                     font = ('Microsoft YaHei UI',DatosEjec[4],'bold'))

            #Configuración de columnas
            for i in range(0, 4):
                ctrlluces2.grid_columnconfigure(i, weight = 1, uniform = 'fig')

            indc_plt1 = tk.Label(ctrlluces2, text = ' * Encender todos los lasers ', fg = 'black',
                                  bg='white', font = ('Microsoft YaHei UI', DatosEjec[4]))

            plt_1 = tk.Button(ctrlluces2, text = ' Realizar acción ', relief = tk.RIDGE, fg = 'white',
                              borderwidth = 1, bg='#1A5276', cursor = 'hand2', 
                              font = ('Microsoft YaHei UI', DatosEjec[4]), command = lambda:self.acc_plt1())

            indc_plt2 = tk.Label(ctrlluces2, text = ' * Apagar todos los lasers ', fg = 'black',
                                bg='white', font = ('Microsoft YaHei UI', DatosEjec[4]))

            plt_2 = tk.Button(ctrlluces2, text = ' Realizar acción ', relief = tk.RIDGE, fg = 'white',
                              borderwidth = 1, bg='#1A5276', cursor = 'hand2',
                              font = ('Microsoft YaHei UI', DatosEjec[4]), command = lambda:self.acc_plt2())

            indc_plt3 = tk.Label(ctrlluces2, text = ' Encender área de los dedos ', fg = 'black',
                                bg='white', font = ('Microsoft YaHei UI', DatosEjec[4]))

            plt_3 = tk.Button(ctrlluces2, text = ' Mano Der. ', relief = tk.RIDGE, fg = 'white',
                              borderwidth = 1, bg='#1A5276', cursor = 'hand2',
                              font = ('Microsoft YaHei UI', DatosEjec[4]), command = lambda:self.acc_plt3())
            
            plt_4 = tk.Button(ctrlluces2, text = ' Mano Izq. ', relief = tk.RIDGE, fg = 'white',
                              borderwidth = 1, bg='#1A5276', cursor = 'hand2',
                              font = ('Microsoft YaHei UI', DatosEjec[4]), command = lambda:self.acc_plt4())

            indc_plt4 = tk.Label(ctrlluces2, text = ' Encender área de artículaciones MCP-PIP ', fg = 'black',
                                bg='white', font = ('Microsoft YaHei UI', DatosEjec[4]))

            plt_5 = tk.Button(ctrlluces2, text = ' Realizar acción ', relief = tk.RIDGE, fg = 'white',
                              borderwidth = 1, bg='#1A5276', cursor = 'hand2',
                              font = ('Microsoft YaHei UI', DatosEjec[4]), command = lambda:self.acc_plt5())

            # Colocando elementos en ventana
            Etiqueta.grid(column=0, row=0, padx=(10, 10), pady=(2, 4), columnspan=1, sticky='nswe')
            ctrlluces.grid(column=0, row=1, padx=(10, 10), pady=(4, 4), columnspan=1, sticky='nswe')
            ctrlluces2.grid(column=0, row=2, padx=(10, 10), pady=(2, 10), columnspan=1, sticky='nswe')

            # Plantilla de botones
            indicador0.grid(column=0, row=0, padx=(10, 10), pady=(5, 5), columnspan = 9, sticky='nsw')

            mov = 1
            # 1° Fila
            btn1.grid(column=mov+2, row=2, padx=(3, 3), pady=(10,3), columnspan=1, sticky='nsew')
            btn2.grid(column=mov+3, row=2, padx=(3, 3), pady=(10,3), columnspan=1, sticky='nsew')
            btn3.grid(column=mov+4, row=2, padx=(3, 3), pady=(10,3), columnspan=1, sticky='nsew')

            # 2° Fila
            btn4.grid(column=mov+0, row=3, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')
            btn5.grid(column=mov+1, row=3, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn6.grid(column=mov+2, row=3, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn7.grid(column=mov+3, row=3, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn8.grid(column=mov+4, row=3, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn9.grid(column=mov+5, row=3, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn10.grid(column=mov+6, row=3, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')
            
            # 3° Fila
            btn11.grid(column=mov+0, row=4, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')
            btn12.grid(column=mov+1, row=4, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn13.grid(column=mov+2, row=4, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn14.grid(column=mov+3, row=4, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn15.grid(column=mov+4, row=4, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn16.grid(column=mov+5, row=4, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn17.grid(column=mov+6, row=4, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')
            

            # 4° Fila
            btn18.grid(column=mov+0, row=5, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')
            btn19.grid(column=mov+1, row=5, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn20.grid(column=mov+2, row=5, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn21.grid(column=mov+3, row=5, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn22.grid(column=mov+4, row=5, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn23.grid(column=mov+5, row=5, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn24.grid(column=mov+6, row=5, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')
            

            # 5° Fila
            btn25.grid(column=mov+0, row=6, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')
            btn26.grid(column=mov+1, row=6, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn27.grid(column=mov+2, row=6, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn28.grid(column=mov+3, row=6, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn29.grid(column=mov+4, row=6, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn30.grid(column=mov+5, row=6, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn31.grid(column=mov+6, row=6, padx=(3, 3), pady=(15,3), columnspan=1, sticky='nsew')

            # 6° Fila
            btn32.grid(column=mov+0, row=7, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn33.grid(column=mov+1, row=7, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn42.grid(column=mov+2, row=7, padx=(3, 3), pady=(3,0), columnspan=3, sticky='nsew')
            btn34.grid(column=mov+5, row=7, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn35.grid(column=mov+6, row=7, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            
            # 7° Fila
            btn36.grid(column=mov+0, row=8, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn37.grid(column=mov+1, row=8, padx=(3, 3), pady=(3,3), columnspan=1, sticky='nsew')
            btn38.grid(column=mov+5, row=8, padx=(3, 3), pady=(3,0), columnspan=1, sticky='nsew')
            btn39.grid(column=mov+6, row=8, padx=(3, 3), pady=(3,0), columnspan=1, sticky='nsew')

            # 8° Fila
            btn40.grid(column=mov+1, row=9, padx=(3, 3), pady=(3,0), columnspan=1, sticky='nsew')
            btn41.grid(column=mov+5, row=9, padx=(3, 3), pady=(3,0), columnspan=1, sticky='nsew')
            

            # Sección indicador de Encendido/Apagado
            indic_luces.grid(column=0, row=10, padx=(10, 10), pady=(0, 0), columnspan = 14, sticky='nswe')
            indc1.grid(column = 0, row=0, padx=(5, 8), pady=(3, 3), columnspan=1, sticky='nswe')
            btn_1.grid(column = 1, row=0, padx=(8, 5), pady=(3, 3), columnspan=1, sticky='nswe')
            indc2.grid(column = 2, row=0, padx=(5, 8), pady=(3, 3), columnspan=1, sticky='nswe')
            btn_2.grid(column = 3, row=0, padx=(8, 5), pady=(3, 3), columnspan=1, sticky='nswe')


            # Botones de encendido rápido
            indc_plt1.grid(column = 0, row=0, padx=(15, 5), pady=(5, 5), columnspan=2, sticky='nsw')
            plt_1.grid(column = 3, row=0, padx=(5, 15), pady=(5, 5), columnspan=2, sticky='nswe')
            indc_plt2.grid(column = 0, row=1, padx=(15, 5), pady=(5,5), columnspan=2, sticky='nsw')
            plt_2.grid(column = 3, row=1, padx=(5, 15), pady=(5, 5), columnspan=2, sticky='nswe')
            indc_plt3.grid(column = 0, row=2, padx=(15, 5), pady=(5, 5), columnspan=2, sticky='nsw')
            plt_3.grid(column = 2, row=2, padx=(5, 15), pady=(5,5), columnspan=1, sticky='nswe')
            plt_4.grid(column = 3, row=2, padx=(5, 15), pady=(5,5), columnspan=1, sticky='nswe')
            
            indc_plt4.grid(column = 0, row=3, padx=(15, 5), pady=(5, 5), columnspan=2, sticky='nsw')
            plt_5.grid(column = 3, row=3, padx=(5, 15), pady=(5, 5), columnspan=2, sticky='nswe')

            # Añadiendo paneles a la ventana
            botones.pack(fill = 'both', side='top', expand=1)
            botones.config(bg = 'white', bd = 3, relief = 'raised')
            botones.propagate(0)

            self.btns = [btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9, btn10,
                btn11, btn12, btn13, btn14, btn15, btn16, btn17, btn18, btn19, btn20,
                btn21, btn22, btn23, btn24, btn25, btn26, btn27, btn28, btn29, btn30,
                btn31, btn32, btn33, btn34, btn35, btn36, btn37, btn38, btn39, btn40,
                btn41, btn42]
        else:
            SubVent3.deiconify()
            SubVent3.grab_set()

    # Función que acciona el encendido de las luces led rojas
    def acccion_luz(self, pin_arduino, btn = None):
        try:
            if(btn.cget("bg") == '#C70039'):
                btn.configure(bg = 'gray50') #●
                self.tarjeta.envio_Arduino(pin_arduino+'_0')
                time.sleep(0.5)
            else:
                btn.configure(bg = '#C70039')
                self.tarjeta.envio_Arduino(pin_arduino+'_1')
                time.sleep(0.5)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

    # Función que acciona el encendido de las luces led rojas
    def acccion_luz2(self, pin_arduino, btn = None, btn2 = None):
        try:
            if(btn.cget("bg") == '#C70039'):
                btn.configure(bg = 'gray50') #●
                btn2.configure(bg = 'gray50') #●
                self.tarjeta.envio_Arduino(pin_arduino+'_0')
                #time.sleep(1)
            else:
                btn.configure(bg = '#C70039') #●
                btn2.configure(bg = '#C70039') #●
                self.tarjeta.envio_Arduino(pin_arduino+'_1')
                #time.sleep(1)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

    # Función plantilla luces 1: Encender todas las luces
    def acc_plt1(self):
        try:
            self.tarjeta.envio_Arduino('P1')
            for btn in self.btns:
                btn.configure(bg = '#C70039')
            time.sleep(1)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

    # Función plantilla luces 2: Apagar todas las luces
    def acc_plt2(self):
        try:
            self.tarjeta.envio_Arduino('P2')
            for btn in self.btns:
                btn.configure(bg = 'gray50')
            time.sleep(1)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

    # Función plantilla luces 3: Encender luces en la sección de dedos
    def acc_plt3(self):
        # Mano Derecha
        try:
            btns2 = [self.btns[3], self.btns[9], self.btns[10], self.btns[16], self.btns[17],
                     self.btns[23], self.btns[24], self.btns[30], self.btns[33], self.btns[34],
                     self.btns[37], self.btns[38], self.btns[40], self.btns[41]]

            self.tarjeta.envio_Arduino('P3')

            j = 0
            for btn in self.btns:
                if btn == btns2[j]:
                    btn.configure(bg = 'gray50')
                    if j < len(btns2)-1: j+=1
                else:
                    btn.configure(bg = '#C70039')
            time.sleep(1)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

    # Función plantilla luces 4: Encender luces en la sección de dedos
    def acc_plt4(self):
        # Mano Izquierda
        try:
            btns2 = [self.btns[3], self.btns[9], self.btns[10], self.btns[16], self.btns[17],
                     self.btns[23], self.btns[24], self.btns[30], self.btns[31], self.btns[32],
                     self.btns[35], self.btns[36], self.btns[39], self.btns[41]]

            self.tarjeta.envio_Arduino('P4')

            j = 0
            for btn in self.btns:
                if btn == btns2[j]:
                    btn.configure(bg = 'gray50')
                    if j < len(btns2)-1: j+=1
                else:
                    btn.configure(bg = '#C70039')
            time.sleep(1)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

    # Función plantilla luces 5: Encender luces en la sección de MCP a PIP
    def acc_plt5(self):
        try:
            btns2 = [self.btns[18], self.btns[19], self.btns[20], self.btns[21], self.btns[22],
                     self.btns[25], self.btns[26], self.btns[27], self.btns[28], self.btns[29],
                     self.btns[41]]
                    
            self.tarjeta.envio_Arduino('P5')

            j = 0
            for btn in self.btns:
                if btn == btns2[j]:
                    btn.configure(bg = '#C70039')
                    if j < len(btns2)-1: j+=1
                else:
                    btn.configure(bg = 'gray50')
            time.sleep(1)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

    # Función que incrementa el contador de siguiente persona a la cual será capturada una fotografía
    def siguiente_persona_capturar(self, indicador_mano, indicador_luz):
        global Etiqueta3, Etiqueta6, Dominante, NoDominante, SubVent3, MiniMenu3
        
        list_opcs[2] += 1
        Etiqueta3.config(text = ' ■ Fotografiando mano del sujeto no. '+str(list_opcs[2]))


        try:
            self.tarjeta.envio_Arduino('O')
            #self.tarjeta.envio_Arduino('x')
            time.sleep(1)
        except:
            messagebox.showwarning(' Error de conexión con la tarjeta ', ' Ha ocurrido un error de conexión con la tarjeta Arduino ')

        if SubVent3 is not None:
            SubVent3.destroy()
            SubVent3 = None

        indicador_mano.current(0)
        indicador_luz.current(0)
        
        Etiqueta6.grid(column = 0, row = 1, padx = (10, 10), pady = (2, 5), sticky = 'nswe', columnspan = 2)
        Dominante.grid(column = 2, row = 1, padx = (10, 10), pady = (2, 15), sticky = 'nswe', columnspan = 1)
        NoDominante.grid(column = 3, row = 1, padx = (10, 10), pady = (2, 15), sticky = 'nswe', columnspan = 1)
        MiniMenu3.grid_forget()

    # Función que realiza el reinicio de los laser en la tarjeta arduino así como también en detener la captura de fotogramas y destruir las posibles ventanas emergentes
    def cerrarSubventana2(self):
        global SubVent2, SubVent3, detener_hilo

        try:
            self.tarjeta.envio_Arduino('O')
            #self.tarjeta.envio_Arduino('x')
            time.sleep(1)
        except Exception as err:
            print(err)

        if SubVent3 is not None:
            SubVent3.destroy()
            SubVent3 = None

        SubVent2.destroy()
        SubVent2 = None

        # Cerrando ventana de cámara
        detener_hilo = True

        # Mostrando raíz
        raiz.deiconify()

    # Creación de la ventana que realiza el control de la captura de fotografías
    def mostrar_control_captura(self):
        global SubVent, SubVent2, Manoizq, Manoder, luz_led, luz_laser, Etiqueta3, Etiqueta6, Dominante, NoDominante, MiniMenu3

        # Eliminando la ventana que solicita la información de las fotografías obtenidas
        SubVent.destroy()

        # Variable auxiliar que ayuda a reducir un poco el tamaño de la letra
        aux = 10

        # Variable auxiliar que amplia el valor de altura de la ventana si se encuentra conectada una tarjeta Aruino para mostrar los controles de luces de la caja
        aux2 = 50

        # Validando si existe una tarjeta Arduino conectada
        if self.tarjeta.deteccion(): #-------------------------------------------***************************************//////////////////////
            aux2 = 70
     
        SubVent2 = tk.Toplevel(raiz)
        SubVent2.focus_force()
        SubVent2.grab_set()
        SubVent2.grab_release()
        SubVent2.title('SCDR-AR Controles para captura de fotos: CAPTURA DE FOTOGRAFÍAS [2]')
        SubVent2.resizable(False, False)
        SubVent2.geometry('{}x{}+{}+{}'.format(ventana[10], ventana[11], ventana[12], ventana[13]))

        SubVent2.config(bg = paletaColores[-1], bd = 9, relief = 'groove')
        SubVent2.protocol('WM_DELETE_WINDOW', lambda:self.cerrarSubventana2())
        SubVent2.grid_columnconfigure(0, weight = 1, uniform = 'fig')
        SubVent2.grid_columnconfigure(1, weight = 1, uniform = 'fig')

        
        SubVent2.grid_rowconfigure(2, weight = 1)

        # ___________________________________________________ Posición 2 - Contador de sujetos
        list_opcs.append(1)
        # ___________________________________________________ Posición 3 - Contador de fotografías capturadas
        list_opcs.append(1)
        # ___________________________________________________ Posición 4 - Bandera que indica cual mano (izq - I / der - D) es capturada la fotografía 
        list_opcs.append('D')
        # ___________________________________________________ Posición 5 - Bandera que indica si la mano capturada en la fotografía es la dominante (Dom - Dominante / NDom - No Dominate)
        list_opcs.append(list_opcs[4] + 'Dom')

        
        Etiqueta = tk.Label(SubVent2, 
                        text = '\n Sistema Clasificador de Daño Radiológico por AR \n\n Captura de fotografías\n', 
                        bg = '#1A5276', fg = 'white', font = ('Microsoft YaHei UI', 13, 'bold'))

        MiniMenu = tk.LabelFrame(SubVent2,
                                 text = ' Información de las fotografías a capturar ',
                                 bd = 3, bg = 'white', fg = '#092B3C',
                                 font = ('Microsoft YaHei UI', 13, 'bold'))

        MiniMenu.grid_columnconfigure(0, weight = 1, uniform = 'fig')
        #MiniMenu.grid_rowconfigure(0, weight = 1, uniform = 'fig')
        #MiniMenu.grid_rowconfigure(1, weight = 1, uniform = 'fig')

        Etiqueta3 = tk.Label(MiniMenu,
                             text = ' ■ Fotografiando mano del sujeto no. '+str(list_opcs[2]),
                             bg = 'white', fg = 'black', justify = 'left',
                             font = ('Microsoft YaHei UI', 13, 'bold'))

        Etiqueta1 = tk.Label(MiniMenu,
                             text = ' ■ Nombre identificador: '+list_opcs[1],
                             bg = 'white', fg = 'black', justify = 'left',
                             font = ('Microsoft YaHei UI', 13, 'bold'))


       

        MiniMenu2 = tk.LabelFrame(SubVent2,
                                text = ' Menú de opciones ',
                                bd = 3, bg = 'white', fg = '#092B3C',
                                font = ('Microsoft YaHei UI', 13,'bold'))

        MiniMenu2.grid_columnconfigure(0, weight = 1, uniform = 'fig')
        MiniMenu2.grid_columnconfigure(1, weight = 1, uniform = 'fig')
        MiniMenu2.grid_columnconfigure(2, weight = 1, uniform = 'fig')
        MiniMenu2.grid_columnconfigure(3, weight = 1, uniform = 'fig')

        Etiqueta4 = tk.Label(MiniMenu2,
                            text = ' ■ Fotografiando mano: ',
                            bg = 'white', fg = 'black', justify = 'left',
                            font = ('Microsoft YaHei UI', 13, 'bold'))

        style = ttk.Style()
        style.theme_create('combostyle', parent='alt',
                            settings = {'TCombobox':
                                        {'configure':
                                          {'selectbackground': paletaColores[-1],
                                           'fieldbackground': paletaColores[-1],
                                           'background': paletaColores[1],
                                           'foreground': '#092B3C', 
                                           'selectforeground': '#092B3C',
                                           'padding': [2, 2, 2, 2]}}})
        style.theme_use('combostyle')

        indicador_mano = ttk.Combobox(MiniMenu2, state = 'readonly',
                                      values = ['DERECHA', 'IZQUIERDA'],
                                      font = ('Microsoft YaHei UI', 13, 'bold'))
        indicador_mano.current(0)
        indicador_mano.bind("<<ComboboxSelected>>", lambda _ :self.indicado_mano(_, indicador_mano))

        Etiqueta6 = tk.Label(MiniMenu2,
                             text = ' * ¿La mano DERECHA es la DOMINANTE?: ',
                             bg = 'white', fg = 'black', justify = 'left',
                             font = ('Microsoft YaHei UI', 13, 'bold'))

        Dominante = tk.Button(MiniMenu2,
                              text = 'Si',
                              cursor = 'hand2', bg = '#AED6F1', fg = 'black',
                              font = ('Microsoft YaHei UI', 13, 'bold'),
                              command = lambda:self.indicando_mano_dom('Dom'))

        NoDominante = tk.Button(MiniMenu2,
                              text = 'No',
                              cursor = 'hand2', bg = 'white', fg = 'black',
                              font = ('Microsoft YaHei UI', 13, 'bold'),
                              command = lambda:self.indicando_mano_dom('NoDom'))





        
        MiniMenu3 = tk.Frame(MiniMenu2, bd = 0, bg = 'white')
        MiniMenu3.grid_columnconfigure(0, weight = 1, uniform = 'fig')
        MiniMenu3.grid_columnconfigure(1, weight = 1, uniform = 'fig')
        MiniMenu3.grid_columnconfigure(2, weight = 1, uniform = 'fig')
        MiniMenu3.grid_columnconfigure(3, weight = 1, uniform = 'fig')

        Etiqueta5 = tk.Label(MiniMenu3,
                            text = ' ■ Luz de la caja activa: ',
                            justify = 'left', bg = 'white', fg = 'black',
                            font = ('Microsoft YaHei UI', 13, 'bold'))

        indicador_luz = ttk.Combobox(MiniMenu3, state = 'readonly',
                                    values = ['LUZ LED', 'LUCES LASER'],
                                    font = ('Microsoft YaHei UI', 13, 'bold'))
        indicador_luz.current(0)
        indicador_luz.bind("<<ComboboxSelected>>", lambda _ :self.indicado_luz(_, indicador_luz))

        
        Nueva_persona = tk.Button(MiniMenu3,
                        text = ' Fotografíar nuevo sujeto ',
                        cursor = 'hand2', bg = paletaColores[4], fg = 'black',
                        font = ('Microsoft YaHei UI', 13, 'bold'),
                        command = lambda:self.siguiente_persona_capturar(indicador_mano, indicador_luz))

        Capturar = tk.Button(MiniMenu3,
                        text = ' Capturar fotografía ',
                        cursor = 'hand2', bg = '#1A5276', fg = 'white',
                        font = ('Microsoft YaHei UI', 13, 'bold'),
                        command = lambda:self.captura_fotografia())

        Etiqueta.grid(column = 0, row = 0, padx = (10, 10), pady = (8, 5), sticky = 'nswe', columnspan = 2)
        
        MiniMenu.grid(column = 0, row = 1, padx = (10, 10), pady = (5, 5), sticky = 'nsew', columnspan = 2)
        Etiqueta3.grid(column = 0, row = 1, padx = (10, 10), pady = (5, 10), sticky = 'w', columnspan = 2)
        Etiqueta1.grid(column = 0, row = 0, padx = (10, 10), pady = (10, 5), sticky = 'w', columnspan = 2)
        

        
        MiniMenu2.grid(column = 0, row = 2, padx = (10, 10), pady = (5, 10), sticky = 'nsew', columnspan = 2)
        Etiqueta4.grid(column = 0, row = 0, padx = (10, 10), pady = (2, 5), sticky = 'nsw', columnspan = 2)
        indicador_mano.grid(column = 2, row = 0, padx = (10, 10), pady = (2, 15), sticky = 'nswe', columnspan = 2)

        Etiqueta6.grid(column = 0, row = 1, padx = (10, 10), pady = (2, 5), sticky = 'nsw', columnspan = 2)
        Dominante.grid(column = 2, row = 1, padx = (10, 10), pady = (2, 15), sticky = 'nswe', columnspan = 1)
        NoDominante.grid(column = 3, row = 1, padx = (10, 10), pady = (2, 15), sticky = 'nswe', columnspan = 1)



        #Minimenu3
        Etiqueta5.grid(column = 0, row = 0, padx = (10, 10), pady = (5, 10), sticky = 'nsw', columnspan = 2)
        indicador_luz.grid(column = 2, row = 0, padx = (10, 10), pady = (5, 10), sticky = 'nswe', columnspan = 2)

        Nueva_persona.grid(column = 0, row = 1, padx = (10, 10), pady = (5, 15), sticky = 'nswe', columnspan = 2)
        Capturar.grid(column = 2, row = 1, padx = (10, 10), pady = (5, 15), sticky = 'nswe', columnspan = 2)
        

    #_____________ Hilos ________________
    # Imagen obtenida de la camara
    def imagenCamara(self):
        global detener_hilo

        # No detener la lectura de la cámara
        detener_hilo = False

        #self.inf_cam[2] = 0
        
        cv2.namedWindow('Imagen_caja', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Imagen_caja', int(ventana[0] - (ventana[0]*.70)), ventana[15])
        cv2.moveWindow('Imagen_caja', int((ventana[0]/2) + 10), int((ventana[1]/2) - (ventana[15]/2)))
        cv2.getWindowProperty('Imagen_caja', cv2.WND_PROP_VISIBLE)

        
        while self.inf_cam[0].isOpened() and not detener_hilo:
            try:
                #Captura de Imagen
                _, frame = self.inf_cam[0].read()

                # Medidas de la imagen
                ancho, alto, _ = frame.shape

                # Rotando la imagen
                frame = cv2.rotate(frame.copy(), cv2.ROTATE_90_CLOCKWISE)  
            
                # Obteniendo las coordenadas de la posición del centro
                x = int(ancho/2)
                y = int(alto/2)
                
                # Dibujado de centro en la imagen obtenida por la cámara
                cv2.circle(frame, (x, y), 15, [0, 0, 255], -1)
        
                # Mostrando imagen
                cv2.imshow('Imagen_caja', frame)
        
                # Cerrando ventana de video capturado
                if ((cv2.waitKey(1) & 0xFF) == ord('q')) or ((cv2.waitKey(1) & 0xFF) == ord('Q')) or cv2.getWindowProperty('Imagen_caja', cv2.WND_PROP_VISIBLE) <= 0:
                    if self.tarjeta.deteccion():
                        self.tarjeta.envio_Arduino('O')
                        time.sleep(1)
                    break
            except:
                if self.tarjeta.deteccion():
                    self.tarjeta.envio_Arduino('O')
                    time.sleep(1)
                break
                messagebox.showwarning(' Error en cámara ', ' Ha ocurrido un error en la conexión de la cámara ')

        # Cerrado de ventanas
        global SubVent2, SubVent3

        if SubVent3 is not None:
            SubVent3.destroy()
            SubVent3 = None

        if SubVent2 is not None:
            SubVent2.destroy()
            SubVent2 = None

        list_opcs.clear()

        self.inf_cam[0].release()
        self.inf_cam.clear()
        cv2.destroyAllWindows()

        # Mostrando raíz
        raiz.deiconify()


    #Efecto - Indicador de fotografia capturada
    def captura_fotografia(self):
        #Captura de Imagen
        _, captura = self.inf_cam[2].read()#3
        #cv2.setIndex("Imagen_caja", cv2. WND_PROP_FULLSCREEN, cv2. WINDOW_FULLSCREEN)

        # Medidas de la imagen
        ancho, alto, _ = captura.shape

        #captura = cv2.resize(captura, (1920, 1080), interpolation = cv2.INTER_AREA)

        # Rotando la imagen
        #captura = cv2.rotate(captura, cv2.ROTATE_90_CLOCKWISE) # OJO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Redimensionamiento de la imagen
        #captura = cv2.resize(captura, (round(ancho*2), round(alto*2)), interpolation = cv2.INTER_AREA) # OJO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Guardando imagen capturada
        #list_opcs[4] = indicador de mano
        #list_opcs[2] += 1 #persona
        #list_opcs[3] = numeor foto
        #list_opcs[5] =  indicador de Dominante
        if list_opcs[4] == 'D':
            aux = ("_D_NoDom", "_D_Dom")[list_opcs[5] == 'D_Dom']
        else:
            #'I'
            aux = ("_I_NoDom", "_I_Dom")[list_opcs[5] == 'I_Dom']

        #cv2.imwrite(list_opcs[0]+'/'+list_opcs[1]+str(list_opcs[3])+'_'+str(list_opcs[2])+list_opcs[4]+'.jpg', captura)
        try:
            cv2.imwrite(list_opcs[0]+'/'+list_opcs[1]+str(list_opcs[3])+'_'+str(list_opcs[2])+aux+'.jpg', captura)
            list_opcs[3]+=1
            print('Fotografía capturada correctamente...')
        except Exception as err:
            print('Error ocurrido en la captura fotografía.\n\n', err)

   




# ************************************* Objeto que realizará el procesamiento de las imagenes obtenidas por la cámara
class procesamientoImgs:
    modeloYolo = None          # Objeto de YOLO
    bandProcesamiento = None   # Bandera de control en procesamiento

    # Constructor
    def __init__(self):
        # Detección de articulaciones con modeloYolo en imágenes
        self.modeloYolo = yolov5.load('PESOS_YOLO/YOLO_RGB.pt')
        
        # Configuración de YOLOv5
        self.modeloYolo.conf = 0.70
        self.modeloYolo.iou = 0.45
        self.modeloYolo.agnostic = False      # NMS class-agnostic
        self.modeloYolo.multi_label = False   # NMS multiple labels per box
        self.modeloYolo.max_det = 1000        # maximum number of detections per image
        
        self.bandProcesamiento = 0



    # -- ALMACENAMIENTO DE LOS RESULTADOS *******************************************************************************************
    # Función que solicita la ruta en donde serán almacenados los resultados obtenidos
    def seleccionRutaAlmacenamientoResultados(self):
        global SubVent      # ventana emergente

        rutaAlmacenamiento = None
        
        # ventana emergente en la cual solicitará la dirección donde serán almacenados los resultados obtenidos
        rutaAlmacenamiento = filedialog.askdirectory(parent = SubVent)

        if not rutaAlmacenamiento:
            messagebox.showwarning(' ◢ Extracción de fotografías', 'No ha seleccionado la ubicación donde serán almacenados los resultados obtenidos', parent = SubVent)
        else:
            fecha = datetime.now()
            d = (str(fecha.day), '0' + str(fecha.day))[fecha.day < 10]
            m = (str(fecha.month), '0' + str(fecha.month))[fecha.month < 10]
            a = (str(fecha.year), '0' + str(fecha.year))[fecha.year < 10]
            h = (str(fecha.hour), '0' + str(fecha.hour))[fecha.hour < 10]
            mi = (str(fecha.minute), '0' + str(fecha.minute))[fecha.minute < 10]
            s = (str(fecha.second), '0' + str(fecha.second))[fecha.second < 10]

            nombCarpeta = 'Resultados_'+list_opcs[1]+'_f{}_{}_{}h{}_{}_{}'.format(d, m, a, h, mi, s)
            rutaAlmacenamiento += '/'+nombCarpeta

        return rutaAlmacenamiento

    # Creado la carpeta que almacenará los resultados obtenidos, dentro de la dirección indicada previamente 
    def almacenamientoResultados(self):
        global SubVent     # ventana emergente

        if not list_opcs[2]:
            messagebox.showwarning(' ◢ Almacenamiento de resultados', 'No ha seleccionado la ubicación donde serán almacenados los resultados obtenidos', parent = SubVent)
            list_opcs[2] = None

            # Solicitando ruta para almacenar los resultados obtenidos
            try:
                list_opcs[2] = self.seleccionRutaAlmacenamientoResultados()
                os.mkdir(list_opcs[2])    # Creando carpeta donde serán almacenados los resultados
            except Exception as err:
                list_opcs[2] = None
                messagebox.showwarning(' ◢ Almacenamiento de resultados', 'Ha ocurrido un error al seleccionar la ubicación en donde serán almacenados los resultados obtenidos. \n\n'+err, parent = SubVent)
        else:
            try:
                os.mkdir(list_opcs[2])    # Creando carpeta donde serán almacenados los resultados
            except:
                list_opcs[2] = None

        return list_opcs[2]




    # Almacenamiento de los resultados obtenidos (detección de puntos)
    def resultados_puntos(self, carpetaResultados, resultados, img, noImg, tituloImg):
        if(carpetaResultados is not None) and (os.path.isdir(carpetaResultados)):
            carpetaResultados_img = carpetaResultados+'/'+tituloImg #img'+str(noImg)

            #______________ Guardando images y resultados
            #carpetaResultados = carpetaResultados_p+'\\Puntos'
            #carpetaResultados1 = carpetaResultados_p+'/'+nomb_carpeta+'/puntos_identificados'
            #print(carpetaResultados)
            #os.mkdir(carpetaResultados1)

            cv2.imwrite(carpetaResultados_img+'/puntos_MediaPipe.jpg', img)
            
            #______________ Guardando CSV
            df = []
            num_dedo = 1;
            for dedo in resultados[0]:
                ### VER LO QUE LA DRA ME ESCRIBIO EN LA FOTO DE LA HOJA
                dfaux = pd.DataFrame(resultados[num_dedo], columns = ['id_falanges', 'Distancia', 'Pendiente', 'Angulo'])
                df.append(dfaux)
                dfaux.to_csv(carpetaResultados_img+'/rslts_'+dedo+'_'+str(num_dedo)+'.csv', index = False)
                num_dedo+=1

    # Almacenamiento del resultado obtenido (dirección de la palma)
    def resultados_direccion_palma(self, carpetaResultados, img, img2, noImg, tituloImg):
        if(carpetaResultados is not None) and (os.path.isdir(carpetaResultados)):
            carpetaResultados_img = carpetaResultados+'/'+tituloImg #img'+str(noImg)

            #______________ Guardando images de resultados
            cv2.imwrite(carpetaResultados_img+'/direccion_palma1.jpg', img)
            cv2.imwrite(carpetaResultados_img+'/direccion_palma2.jpg', img2)




    # -- PREPARACIÓN DE LA(S) IMÁGENE(S) ********************************************************************************************
    # Contabiliza las imágene(s) a procesar dentro de la carpeta indicada, extrayendo y almacenando en una lista, la ruta donde se encuentra cada imagen
    def contadorImgs(self):
        imgsProcesar = []     # Lista con las rutas de las imágenes a procesar
        for img in glob.glob(list_opcs[0]+'/'+list_opcs[1]+'*.jpg'):
            imgsProcesar.append(img)

        return imgsProcesar
    
    # Realiza la carga de las imágenes a procesar y posteriormente iniciar su procesamiento
    def cargaImgs(self):
        imgs = self.contadorImgs()

        if imgs:
            if len(imgs[0]) != 0:
                self.procesamientoGrpImgs(imgs)
            else:
                messagebox.showerror(' ◢ Extracción de imágenes', 'Dentro de: \n'+list_opcs[0]+'\n\n ** No hay imágenes con el ID: "'+list_opcs[1]+'"', parent = raiz)
                list_opcs.clear()  # Vaciando lista
        else:
            messagebox.showerror(' ◢ Extracción de imágenes', 'Dentro de: \n'+list_opcs[0]+'\n\n ** No hay imágenes con el ID: "'+list_opcs[1]+'"', parent = raiz)
            list_opcs.clear()      # Vaciando lista
    

            



    # -- FUNCIONES AUXILIARES EN LA EJECUCIÓN DEL PROCESAMIENTO ********************************************************************   
    # Función que preguntará al usuario si desea detener el procesamiento de las imágenes
    def detener_procesamiento(self):
        self.bandProcesamiento = -1
        self.bandProcesamiento = messagebox.askyesno('Procesamiento de imágenes', '¿Desea deneter el procesamiento de las imágenes?')
    




    # Visualizando los resultados obtenidos en el procesamiento
    def visualizar_rslts(self, fig, noImg, imagen, imagen_modeloYolo, img_p, img_cuadricula):

        ventana_x = ventana[0]
        ventana_y = ventana[1]

        imgs1 = cv2.hconcat([imagen, imagen_modeloYolo])
        imgs2 = cv2.hconcat([img_p, img_cuadricula])
        imgs_result = cv2.vconcat([imgs1, imgs2])
        
        cv2.namedWindow('Resultados', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Resultados', int(ventana_x/4), int(ventana_y/4))
        cv2.moveWindow('Resultados', int(ventana_x/2), int(ventana_y/2))
        cv2.getWindowProperty('Resultados', cv2.WND_PROP_VISIBLE)
        cv2.imshow('Resultados', imgs_result)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    # -- PROCESAMIENTO DE IMÁGENES *************************************************************************************************
    def procesamientoGrpImgs(self, imgs):
        global SubVent               # Nueva ventana emergente
        #carpetaResultados = None    # Bandera de resultados

        # Creando la carpeta donde serán almacenados los resultados
        carpetaResultados = self.almacenamientoResultados()


        # Validando si fue creada correctamente la carpeta para almacenar los resultados obtenidos
        if carpetaResultados is not None:
            # _________________________________________________________________________ Procesamiento de las imágenes
            # Validando si fuerón encontrados imágenes para procesar
            if len(imgs) != 0:
                #for noImg in range(0, len(imgs)):
                noImg = 0

                ## DESTRUI ventana Y MOSTAR UNA NUEVA DE EJECUCIÓN

                ventana_x = ventana[0]
                ventana_y = ventana[1]

                fig = plt.figure()

                
                while True:
                    # Creando la carpeta por imagen procesada
                    try:
                        tituloImg = imgs[noImg].split('\\')
                        tituloImg = tituloImg[1][:-4]
                        print('\n ********* Procesamiento de IMG: ', noImg+1, '\n -- ['+tituloImg+']\n')

                        # Creación de la carpeta de los resultados obtenidos de la imagen en cuestión
                        carpetaResultadosImg = carpetaResultados+'/'+tituloImg
                        os.mkdir(carpetaResultadosImg)

                        # Lectura de la imagen a color
                        imagen = cv2.imread(imgs[noImg], 1)

                        # Rotando la imagen 90° en sentido horario (Para mi caja)
                        imagen = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
                        imagen2 = imagen.copy() # Copiendo imagen para realizar la identificación de las articulaciones con YOLO

                        

                        # _____ PREPROCESAMIENTO DE LA IMAGEN
                        imagen = cv2.blur(imagen, (7,7), cv2.BORDER_DEFAULT)
                        #s = 0.5
                        #imagen = cv2.GaussianBlur(imagen, ksize=(5,5), sigmaX=s, sigmaY=s)
                        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
                        contrast = 0.95 #0.75
                        brightness = 10
                        imagen[:,:,2] = np.clip(contrast * imagen[:,:,2] + brightness, 0, 255)
                        imagen = cv2.cvtColor(imagen, cv2.COLOR_HSV2BGR)




                        # -- Identificación de mano dentro de la imagen
                        print('----- Identificando una mano dentro de la imagen ', end='.'*5)
                        imagenMano_SinFondo = imagen.copy() #self.identificadorMano(imagen)
                        print(' > OK')

                        # -- Identificación y mediciones apartir de las articulaciones por el modelo YOLOv5
                        print('----- Identificación de articulaciones por YOLOv5 en la imagen ', end='.'*5)
                        puntos_modeloYolo = self.deteccionArticulaciones(carpetaResultados, imagen.copy(), imagenMano_SinFondo.copy())
                        

                        if True: #if img is not None:
                            print(' > OK')

                            '''# -- Identificación y recortes de las articulaciones
                            #recorte_imgs_articulaciones(carpetaResultados, puntos_modeloYolo, img.copy(), (noImg+1))
                            print('----- Recortando secciones de articulaciones de la modeloYolo en la imagen ', end='.'*5)
                            self.recorte_imgs_articulaciones(carpetaResultados, puntos_modeloYolo, imagen.copy(), img.copy(), (noImg+1), tituloImg)
                            print(' > OK')

                            # -- Realizando el trazado de la cuadrícula deteccion_cuadricula
                            print('----- Determinando dirección de la modeloYolo en la imagen ', end='.'*5)
                            img_cuadricula = self.determinacion_direccion_palma(carpetaResultados, img.copy(), articulaciones_p, (noImg+1), tituloImg)
                            print(' > OK')
                            
                            #self.visualizar_rslts(noImg, imagen, imagenMano_SinFondo, img_p, img_cuadricula)
                            #self.visualizar_rslts(fig, noImg, imagen, imagenMano_SinFondo, img_p, img_cuadricula)

                            # Esperando respuesta del usuario
                            while self.bandProcesamiento == -1:
                                pass
                            
                            imgs1 = cv2.hconcat([imagen, img])
                            imgs2 = cv2.hconcat([img_p, img_cuadricula])
                            imgs_result = cv2.vconcat([imgs1, imgs2])
                            
                            print('\n----- Mostrando Resultados')
                            cv2.namedWindow('Resultados', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Resultados', int(ventana_x/2), int(ventana_y-100))
                            cv2.moveWindow('Resultados', int(ventana_x/4), 5)
                            cv2.getWindowProperty('Resultados', cv2.WND_PROP_VISIBLE)
                            cv2.imshow('Resultados', imgs_result)
                            cv2.waitKey(3000)
                            

                            print('\n\n\n')


                            noImg+=1
                            if(self.bandProcesamiento == 1) or (noImg == len(imgs)):'''
                            cv2.destroyAllWindows()
                            break
                            #else:
                            #    cv2.destroyAllWindows()
                        else:
                            print('\t > NO DETECTADOS \n')
                            #Eliminando la carpeta de la imagen no procesada
                            #os.rmdir(carpetaResultados_img)
                            noImg+=1
                    except Exception as err:
                        print(err)
                        #noImg+=1
                        #Eliminando la carpeta de la imagen no procesada
                        #os.rmdir(carpetaResultados_img)

                        messagebox.showerror('Error en procesamiento', 'Ha ocurrido un error al procesar a la imagen: '+tituloImg)
                        
                messagebox.showinfo('Procesamiento finalizado', 'El procesamiento del conjunto de imágenes ha concluido')
                list_opcs.clear()
            else:
                messagebox.showinfo('Procesamiento finalizado', 'No hay un conjunto de imágenes para ser procesadas')
                list_opcs.clear()
    






    # _____________________________________________ FUNCIONES

    # _________________________________________________ 
    # PROCESO NO.1 ----- Elimina el fondo de la imagen para solo mostrar la modeloYolo identificada
    def identificadorMano(self, img):
        # Conversión de la imagen a escala de grises
        identificacionMano = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detección de bordes con algoritmo Canny
        canny = cv2.Canny(identificacionMano, 65, 65)

        # Encuentra los contornos de la imagen
        contornos, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        identificacionMano = cv2.drawContours(identificacionMano, contornos, -1, (0, 0, 0), 1)

        _, identificacionMano = cv2.threshold(identificacionMano, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        identificacionMano = cv2.dilate(identificacionMano, kernel, iterations = 5)
        identificacionMano = cv2.morphologyEx(identificacionMano, cv2.MORPH_OPEN, kernel, iterations = 5)
        identificacionMano = cv2.morphologyEx(identificacionMano, cv2.MORPH_CLOSE, kernel, iterations = 6)
    
        identificacionMano = self.depuradorObjetos(identificacionMano, 0)
        identificacionMano = self.depuradorObjetos((255 - identificacionMano), 1) 
        identificacionMano = self.depuradorObjetos(identificacionMano, 0)
        identificacionMano = 255 - identificacionMano

        # Aplicar la máscara a la imagen para quitar el fondo
        img = cv2.bitwise_and(img, img, mask = identificacionMano)

        return img

    # Depurador de objetos pequeños (eliminando objetos cuya área en píxeles es menor)
    def depuradorObjetos(self, Imagen, opcion):
        obj, output, stats, _ = cv2.connectedComponentsWithStats(Imagen, connectivity=4)
        sizes = stats[1:, -1]  # Áreas (en píxeles) de los objetos identificados
        obj -= 1               # Ignorando el fondo en la imagen
        
        tamanioObjs_Ordenados = np.sort(sizes, None) # Ordenando áreas de menor a mayor
        if(len(tamanioObjs_Ordenados) != 1):
            if(opcion == 0):
                min_size = tamanioObjs_Ordenados[obj-1]   # Área minima: El 2°do objeto de mayor área
            else:
                min_size = tamanioObjs_Ordenados[obj-2]   # Área minima: El 3°er objeto de mayor área

            imgObj = np.zeros((Imagen.shape), dtype = 'uint8')
            for j in range(0, obj):
                if sizes[j] >= min_size:                  # Solo dejar a los objetos cuya área en píxeles sea mayor la indicada como mínima
                    imgObj[output == j + 1] = 255
        
            return imgObj
        else:
            return Imagen
    
    
    
    # _________________________________________________ 
    # PROCESO NO.2 ----- Detección de puntos en la modeloYolo
    def deteccionArticulaciones(self, carpetaResultados, imgOriginal, img_sinFondo):
        #Imagenes de puntos de articulaciones ----- img

        # Dimensiones de la imagen a procesar
        alto, ancho, _ = imgOriginal.shape

        # Cambiando canales de BGR a RGB
        imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)

        # Aplicando modelo YOLOv5
        resultadoModelo = self.modeloYolo(imgOriginal, augment=True)

        # Ejecutando la predicción de YOLO
        predictions = resultadoModelo.pred[0]

        # === RESULTADOS DE YOLO
        # Coordenadas de cada objeto identificado
        boxes = predictions[:, :4] # x1, y1, x2, y2

        # Probabilidades de los objetos identificados
        scores = predictions[:, 4]

        # Clase a la que pertenece el objeto identificado
        categories = predictions[:, 5]

        # Organizando en una lista cada objeto:
        # [n][0] - CLASE
        # [n][1] - PROBABILIDAD 
        # [n][2] - COORDENADAS


        #CMC, MCP, IP, PIP
        objts = [[],[],[],[]]

        for i in range(0, categories.numel()):
            clase = int(categories[i].item())
            objt = [
                round(scores[i].item(), 3),  # Probabilidad
                boxes[i].tolist()
            ]

            # Clase CMC
            #if int(categories[i].item()) == 0:
            objts[clase].append(objt)

        #objts[:, 0].argsort()
        print('\n\n Objetos identificados: \n')
        i = 0
        for objt in objts:
            print(' - CLASE ' + str(i))
            for obj in objt:
                print(obj,'\n')
            i+=1


        # Mostrando resultados
        resultadoModelo.show()

        # Guardando los resultados dentro una carpeta (Debe crearse primeramente)
        resultadoModelo.save(save_dir=carpetaResultados)
        os.rename(carpetaResultados+'/image0.jpg', carpetaResultados+'/Articulaciones_Identificadas(YOLOv5).jpg')

        
        puntos_modeloYolo = None
        
        # Bandera que dentra la acción de rotar la imagen
        if True:
            print('HOLA')
        else:
            puntos_modeloYolo = None

        #Retornando la posición de las articulaciones principales
        return puntos_modeloYolo, articulaciones_principales, img_original, img2, img




    # _________________________________________________ 
    # PROCESO NO.3 ----- Detección del sentido en que se encuentre la mano
    def determinacion_direccion_palma(self, carpeta_resultados, img, puntos_articulaciones, num_img, nomb_img):
        # Dimensiones de la imagen a procesar
        alto, ancho, _ = img.shape

        #Imagen de la cuadricula
        img_cuadricula = np.zeros((img.shape), dtype = 'uint8')

        '''
        articulaciones_principales = [dedo_pulgar[2], dedo_pulgar[1], # P2, P3,  ----- MediaPipe
                                        dedo_indice[3], dedo_indice[2],   # P5, P6,
                                        dedo_medio[3], dedo_medio[2],     # P9, P10,
                                        dedo_anular[3], dedo_anular[2],   # P13, P14,
                                        dedo_menique[3], dedo_menique[2], # P17, P18,
                                        munieca]       # P0 (Muñeca) ----------
        

        #Puntos de referencia para trazar la cuadrícula
        puntos_referencia = [puntos_articulaciones[4][3],  # Meñique P17
                        puntos_articulaciones[1][3],  # Indice P5
                        puntos_articulaciones[0][3],  # Pulgar P1
                        puntos_articulaciones[-1]]  # Muñeca P0
        '''
        
        color_p = [0, 255, 0]

        #linea de referencia
        puntos_referencia_central = [puntos_articulaciones[10], # Muñeca
                                    puntos_articulaciones[4]]  # MCP dedo Medio

        # Calculando el punto medio en la palma de la mano
        x1_c = int(puntos_referencia_central[0].x * ancho)
        y1_c = int(puntos_referencia_central[0].y * alto)
        
        x2_c = int(puntos_referencia_central[1].x * ancho)
        y2_c = int(puntos_referencia_central[1].y * alto)
        
        centroide_x = int((x1_c + x2_c)/2)
        centroide_y = int((y1_c + y2_c)/2)

        # Dibujado de puntos de referencia centrales
        cv2.circle(img, (x2_c, y2_c), 10, [0, 0, 255], -1)
        cv2.circle(img, (centroide_x, centroide_y), 15, [0, 0, 255], -1)
        cv2.circle(img, (x1_c, y1_c), 10, [0, 0, 255], -1)
        cv2.line(img, (int((centroide_x + x2_c)/2), int((centroide_y + y2_c)/2)), (int((centroide_x + x1_c)/2), int((centroide_y + y1_c)/2)), (0, 255, 0), 5)

        # Calcula la pendiente de la línea original (Nudillo central y centro de la mano)
        if x2_c - centroide_x == 0:
            pendiente = float('inf')
            # es una línea vertical. 
        else:
            pendiente = (y2_c - centroide_y) / (x2_c - centroide_x)

        # Calcula la pendiente de la línea perpendicular (Para calcular el trazado de las líneas en el eje horizontal - Proyecciones en "X")
        pendiente_perpendicular = -1 / pendiente

        # Una medida grande para alargar la distancia entre 2 puntos
        distancia = 100000000
        x3_aux, y3_aux, x4_aux, y4_aux = 0, 0, 0, 0

        # Calcula los extremos de la línea perpendicular
        if pendiente_perpendicular == float('inf'):
            # es una línea vertical.
            x3 = centroide_x + distancia
            x3_aux = centroide_x -100

            y3 = centroide_y
            y3_aux = centroide_y

            x4 = x3
            x4_aux = x3_aux

            y4 = centroide_y - distancia
            y4_aux = centroide_y - 100
        else:
            # es una línea horizontal.
            if(pendiente_perpendicular >= 0):
                # se inclina hacia arriba
                x3 = centroide_x + distancia
                x3_aux = centroide_x + 100
                x4 = centroide_x - distancia 
                x4_aux = centroide_x - 100
            else:
                # se inclina hacia abajo 
                x3 = centroide_x - distancia
                x3_aux = centroide_x - 100
                x4 = centroide_x + distancia
                x4_aux = centroide_x + 100

            y3 = int(centroide_y + pendiente_perpendicular * (x3 - centroide_x))
            y3_aux = int(centroide_y + pendiente_perpendicular * (x3_aux - centroide_x))
            y4 = int(centroide_y + pendiente_perpendicular * (x4 - centroide_x))
            y4_aux = int(centroide_y + pendiente_perpendicular * (x4_aux - centroide_x))

        #Lineas del eje X
        direction = (y4_aux - y3_aux, x3_aux - x4_aux)
        #for i in range(-10, 40, 1):
        for i in range(-3, 4, 1):
            # Definir la distancia y la dirección para la línea paralela
            distance = 0.5 * i

            # Calcular la posición de la línea paralela
            punto_paralelo1 = (int(x3 + distance * direction[0]), int(y3 + distance * direction[1]))
            punto_paralelo2 = (int(x4 + distance * direction[0]), int(y4 + distance * direction[1]))
            
            # Dibujando línea paralela "X" (trazo de las líneas representativas de X)
            cv2.line(img, punto_paralelo1, punto_paralelo2, (255, 255, 255), 2)
            cv2.line(img_cuadricula, punto_paralelo1, punto_paralelo2, (255, 255, 255), 5)

        #Guía de X (Trazado de la línea que representa a x - COLOR VERDE)
        cv2.line(img, (x3_aux, y3_aux), (x4_aux, y4_aux), (0, 255, 0), 5)

        #Lineas del eje Y
        direction = (y2_c - y1_c, x1_c - x2_c)
        direction2 = (x2_c - x1_c, y2_c - y1_c)
        
        for i in range(-1, 2, 1):
            # Definir la distancia y la dirección para la línea paralela
            distance = 0.5 * i

            # Calcular la posición de la línea paralela
            x1 = int(x1_c + distance * direction[0])
            x1 = int(x1 + 10 * direction2[0])

            y1 = int(y1_c + distance * direction[1])
            y1 = int(y1 + 10 * direction2[1])
            punto_paralelo1 = (x1, y1)

            x2 = int(x2_c + distance * direction[0])
            x2 = int(x2 - 10 * direction2[0])

            y2 = int(y2_c + distance * direction[1])
            y2 = int(y2 - 10 * direction2[1])
            punto_paralelo2 = (x2, y2)
            
            # Dibujar la línea paralela y
            cv2.line(img, punto_paralelo1, punto_paralelo2, (255, 255, 255), 2)
            cv2.line(img_cuadricula, punto_paralelo1, punto_paralelo2, (255, 255, 255), 5)
        
        
        # Almacenado los resultados obtenidos
        self. resultados_direccion_palma(carpeta_resultados, img_cuadricula, img, num_img, nomb_img)

        return img



    # _________________________________________________ 
    # PROCESO NO.4 ----- Recorte de las articulaciones identificadas
    def recorte_imgs_articulaciones(self, carpeta_resultados, puntos_articulaciones, imagen, img, num_img, nomb_img):
        # Dimensiones de la imagen a procesar
        alto, ancho, _ = img.shape

        # Bandera para indicar si pueden ser almacenados los recortes
        band_carpeta_recortes = False

        # Creando carpeta donde serán almacenados los recortes
        if(carpeta_resultados is not None) and (os.path.isdir(carpeta_resultados)):
            carpeta_resultados_img = carpeta_resultados+'/'+nomb_img #img'+str(num_img)

            try:
                carpeta_resultados_imgs = carpeta_resultados_img+'/recortes_articulaciones'
                os.mkdir(carpeta_resultados_imgs)
                band_carpeta_recortes = True

                # Recortes para el Dataset
                carpeta_resultados_imgs2 = carpeta_resultados_img+'/recortes_articulaciones_DATASET'
                os.mkdir(carpeta_resultados_imgs2)
            except Exception as err:
                print(err)
                
        
        
        
        
        
        # DETERMINAR POR MEDIO DE RECTAS PARALELAS LAS ARTICULACIONES A RECORTAR.... PROBAR SI SE RECORTAN EN EL SENTIDO DESEADO





        #print(len(puntos))
        '''
        dedo_pulgar,    #0
        dedo_indice,    #1
        dedo_medio,     #2
        dedo_anular,    #3
        dedo_menique
        '''

        i = 0
        n_articulacion = 0
        salto = 1
        img2 = img.copy()
        img3 = img.copy()
        img4 = img.copy()
        
        for dedos in puntos_articulaciones:
            for puntos in dedos:
                if (i+1) == salto:
                    salto += 4
                    pass
                else:
                    if (i+1) != 4:
                        n_articulacion += 1
                        # Mi caja
                        #x, y, x2, y2 = int(puntos.x * ancho) - 60, int(puntos.y * alto) - 60,  int(puntos.x * ancho) + 60 , int(puntos.y * alto) + 60
                        
                        # Dataset  - Diana
                        x, y, x2, y2 = int(puntos.x * ancho) - 100, int(puntos.y * alto) - 100,  int(puntos.x * ancho) + 100 , int(puntos.y * alto) + 100
                        
                        #print(x, y, x2, y2)

                        image = imagen[y: y2, x: x2]
                        
                        #img = cv2.rectangle(img, (x, y), (x2, y2), (0,255,0), 3)
                        img2 = cv2.putText(img2, str(n_articulacion), ( int(puntos.x * ancho) - 80, int(puntos.y * alto) - 80), cv2.FONT_ITALIC, 0.8, (140, 255, 255), 5, cv2.LINE_AA)
                        
                        # ----- Imagen con mensaje de las articulaciones identificadas
                        img4 = cv2.putText(img4, str(n_articulacion), ( int(puntos.x * ancho) - 80, int(puntos.y * alto) - 80), cv2.FONT_ITALIC, 0.8, (140, 255, 255), 5, cv2.LINE_AA)
                        

                        img3 = cv2.rectangle(img3, ( int(puntos.x * ancho) - 35, int(puntos.y * alto) - 35),  ( int(puntos.x * ancho) + 35 , int(puntos.y * alto) + 35), (0,255,0), 2)
                        img3 = cv2.circle(img3, (int(puntos.x * ancho), int(puntos.y * alto)), 20, (255,0,0),-1)

                        # ----- Imagen con mensaje de las articulaciones identificadas
                        img4 = cv2.rectangle(img4, ( int(puntos.x * ancho) - 35, int(puntos.y * alto) - 35),  ( int(puntos.x * ancho) + 35 , int(puntos.y * alto) + 35), (0,255,0), 2)
                        

                        if band_carpeta_recortes:
                            cv2.imwrite(carpeta_resultados_imgs+'/articulacion_'+str(n_articulacion)+'.jpg', image)

                            if n_articulacion == 5 or n_articulacion == 8 or n_articulacion == 11 or n_articulacion == 14:
                                cv2.imwrite(carpeta_resultados_imgs2+'/nudillos_'+str(num_img_dataset[0])+'.jpg', image)
                                num_img_dataset[0] += 1
                            else:
                                cv2.imwrite(carpeta_resultados_imgs2+'/articulacion_'+str(num_img_dataset[1])+'.jpg', image)
                                num_img_dataset[1] += 1


                        
                        '''
                        cv2.namedWindow('Mano', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Mano', 500, 600)
                        cv2.imshow('Mano', imagen_m)
                        cv2.imshow('Articulacion detectada', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        '''
                i+=1

        if band_carpeta_recortes:
            cv2.imwrite(carpeta_resultados_img+'/articulaciones1.jpg', img2)
            cv2.imwrite(carpeta_resultados_img+'/articulaciones2.jpg', img3)
            cv2.imwrite(carpeta_resultados_img+'/articulaciones3.jpg', img4)





#______________________________________________________________ Funciones de ventana principal
# ********************************************************************************  Función que realiza el diseño de las subventanas
def creacionSubventa(colorSeccion, imgLogo):
    global SubVent, seccion4
    # __________________________________________________________ Panel de icono
    seccion3 = tk.Frame(SubVent)

    #Configuración de columnas
    seccion3.grid_columnconfigure(0, weight = 1,uniform = 'fig')
    seccion3.grid_columnconfigure(1, weight = 1,uniform = 'fig')
    seccion3.grid_columnconfigure(2, weight = 1,uniform = 'fig')
    #Configuración de filas
    seccion3.grid_rowconfigure(0, weight = 1,uniform = 'fig')
    seccion3.grid_rowconfigure(1, weight = 1)
    seccion3.grid_rowconfigure(2, weight = 1,uniform = 'fig')
    seccion3.grid_rowconfigure(3, weight = 1,uniform = 'fig')
    seccion3.grid_rowconfigure(4, weight = 1,uniform = 'fig')


    #img = Image.open('iconos/Logo.png')            # Load the image
    #img = img.resize((500, 500), Image.ANTIALIAS)  # Resize the image in the given (width, height)
    #img = ImageTk.PhotoImage(img, master = seccion1) 
    #img = tk.PhotoImage(file = 'iconos/Logo_1.png', master = seccion3)
    logo = tk.Label(seccion3, image = imgLogo, bg = colorSeccion)

    label1 = tk.Label(seccion3, text=' Sistema Clasificador de Daño Radiológico por AR', 
                        bg = colorSeccion, fg = 'white', font = ('Microsoft YaHei UI', 15, 'bold'))

    label2 = tk.Label(seccion3, text=' Versión 1.4 - Agosto 2023 ', 
                        bg = colorSeccion, fg = 'white', font = ('Microsoft YaHei UI', 12, 'bold'))

    logo.grid(column = 0, row = 1, padx = (10, 10), pady = (1, 1), columnspan = 3, sticky = 'nswe')
    label1.grid(column = 0, row = 2, padx = (10, 10), pady = (1, 1), columnspan = 3, sticky = 'swe')
    label2.grid(column = 0, row = 3, padx = (10, 10), pady = (1, 10), columnspan = 3, sticky = 'nwe')

    # ___________________________________________________________________ Panel de contenido
    seccion4 = tk.Frame(SubVent)

    #Configuración de columnas
    seccion4.grid_columnconfigure(0, weight = 1,uniform = 'fig')
    #Configuración de filas
    seccion4.grid_rowconfigure(0, weight = 1,uniform = 'fig')

    # Añadiendo paneles a la subventana
    seccion3.config(bg = colorSeccion, bd = 0, relief = 'raised')
    seccion3.grid(column = 0, row = 0, padx = (0, 0), pady = (0, 0), columnspan = 1, sticky = 'nswe')
    
    seccion4.config(bg = paletaColores[4], bd = 0, relief = 'raised')
    seccion4.grid(column = 1, row = 0, padx = (0, 0), pady = (0, 0), columnspan = 1, sticky = 'nswe')
    


# ******************************************************************************** PRIMERA OPCIÓN  "Capturar fotografías"
def capturaFotografias(imgLogo):
    # Llamada a variable global
    global SubVent, seccion4, rutaAlmacenamiento, idImgs, Continuar

    # Variable auxiliar que ayuda a reducir un poco el tamaño de la letra
    aux = 0

    # Ocultando raíz
    raiz.withdraw()

    SubVent = tk.Toplevel(raiz)
    SubVent.focus_force()
    SubVent.grab_set()
    SubVent.grab_release()
    SubVent.title(' SCDR-AR: Captura de fotografías [Ventana 1]')
    SubVent.resizable(False, False)

    #if (DatosEjec[0] == 1800 and DatosEjec[1] == 970):
    #    SubVent.geometry('{}x{}+{}+{}'.format(int(DatosEjec[0]/3), int(DatosEjec[1]/3)+35, int((ventana[0]/2) - (DatosEjec[0]/6)), int((ventana[1]/2) - (((ventana[1]/3)+35)/2))))
    #    aux = 0
    #else:
    #    #SubVent.geometry('{}x{}+{}+{}'.format(int(DatosEjec[0]/2), int(DatosEjec[1]/2), int((ventana[0]/2) - (DatosEjec[0]/4)), int((ventana[1]/2) - (DatosEjec[1]/4))))
    SubVent.geometry('{}x{}+{}+{}'.format(ventana[6], ventana[7], ventana[8], ventana[9]))
    aux = 2

    SubVent.deiconify()
    SubVent.config(bg = 'white', bd = 8, relief = 'groove')
    SubVent.grid_columnconfigure(0, weight = 1, uniform = 'fig')
    SubVent.grid_columnconfigure(1, weight = 1, uniform = 'fig')
    SubVent.grid_rowconfigure(0, weight = 1, uniform = 'fig')
    SubVent.protocol('WM_DELETE_WINDOW', cerrarSubventana)

    # _________________________________________ Varibles de la subventana
    # Cadena donde serán almacenadas las fotografías capturadas
    rutaAlmacenamiento = tk.StringVar(SubVent)
    rutaAlmacenamiento.set('')

    # Nombre que identificará al grupo de imágenes a capturar  
    idImgs = tk.StringVar(SubVent)
    idImgs.set('')

    # Creando el diseño de la subventana
    creacionSubventa('#1A5276',imgLogo)
        
    MiniMenu = tk.LabelFrame(seccion4, bd = 3, bg = 'white')
    # Configurando Columnas
    MiniMenu.grid_columnconfigure(0, weight = 1, uniform = 'fig')
    MiniMenu.grid_columnconfigure(1, weight = 1, uniform = 'fig')
    MiniMenu.grid_columnconfigure(2, weight = 1, uniform = 'fig')
    MiniMenu.grid_columnconfigure(3, weight = 1, uniform = 'fig')
    # Configurando Filas
    MiniMenu.grid_rowconfigure(0, weight = 1, uniform = 'fig')
    MiniMenu.grid_rowconfigure(1, weight = 1)
    MiniMenu.grid_rowconfigure(2, weight = 1)
    MiniMenu.grid_rowconfigure(3, weight = 1)
    MiniMenu.grid_rowconfigure(4, weight = 1)
    MiniMenu.grid_rowconfigure(5, weight = 1)
    MiniMenu.grid_rowconfigure(6, weight = 1)
    MiniMenu.grid_rowconfigure(7, weight = 1, uniform = 'fig')

    Etiqueta0 = tk.Label(MiniMenu,
                         text = '  ■ Captura de fotografías  ',
                         bg = '#1A5276', fg = 'white', justify = 'left',
                         font = ('Microsoft YaHei UI', DatosEjec[4]+aux, 'bold'))
    
    Etiqueta1 = tk.Label(MiniMenu,
                         text = ' Información para el almacenamiento de las fotografías.',
                         bg = 'white', fg = 'black', justify = 'left',
                         font = ('Microsoft YaHei UI',DatosEjec[4]+aux, 'bold'))
                         
    Etiqueta2 = tk.Label(MiniMenu,
                         text = ' 1) Indique la ubicación donde serán almacenadas:',
                         bg = 'white', fg = 'black', justify = 'left',
                         font = ('Microsoft YaHei UI',DatosEjec[4]+aux, 'bold')) 

    Entrada1 = tk.Entry(MiniMenu,
                       textvariable = rutaAlmacenamiento, relief = tk.GROOVE,
                       bg = 'white', fg = 'midnight blue', state = 'readonly',
                       font = ('Microsoft YaHei UI', DatosEjec[4]+aux), justify = 'center')

    Busqueda_ruta = tk.Button(MiniMenu,
                          text = ' Buscar ubicación ', bd = 3,
                          relief = tk.GROOVE, bg = paletaColores[3], fg = 'black',
                          font = ('Microsoft YaHei UI', DatosEjec[4] + aux),
                          cursor = 'hand2', command = seleccionRutaAlmacenamiento)

    Etiqueta3 = tk.Label(MiniMenu,
                         text = ' 2) Ingrese un nombre que identifique a las fotografías:',
                         bg = 'white', fg = 'black', justify = 'left',
                         font = ('Microsoft YaHei UI', DatosEjec[4] + aux, 'bold'))

    Entrada2 = tk.Entry(MiniMenu,
                       textvariable = idImgs,
                       bg = '#F0F0F0', fg = 'black', relief = tk.GROOVE,
                       border = 1, justify = 'center',
                       font = ('Microsoft YaHei UI', DatosEjec[4] + aux))
    
    Continuar = tk.Button(MiniMenu,
                          relief = tk.RAISED, text = ' Continuar ',
                          bg = '#1A5276', fg = 'white', bd = 4,
                          font = ('Microsoft YaHei UI', DatosEjec[4] + aux),
                          cursor = 'hand2', command = capturaFotografias2)
    
    MiniMenu.grid(column = 0, row = 0, padx = (10, 10), pady = (5, 5), sticky = 'nsew', columnspan = 1)
    
    Etiqueta0.grid(column = 0, row = 0, padx = (10, 10), pady = (15, 8), sticky = 'sw', columnspan = 4)
    Etiqueta1.grid(column = 0, row = 1, padx = (10, 10), pady = (0, 0), sticky = 'nw', columnspan = 4)

    Etiqueta2.grid(column = 0, row = 2, padx = (10, 10), pady = (0, 2), sticky = 'sw', columnspan = 4)
    Entrada1.grid(column = 0, row = 3, padx = (10, 10), pady = (2, 2), sticky = 'nsew', columnspan = 4)
    Busqueda_ruta.grid(column = 0, row = 4, padx = (10, 10), pady = (2, 5), sticky = 'nw', columnspan = 4)
    
    Etiqueta3.grid(column = 0, row = 5, padx = (10, 10), pady = (5, 0), sticky = 'sw', columnspan = 4)
    Entrada2.grid(column = 0, row = 6, padx = (10, 10), pady = (0, 20), sticky = 'nsew', columnspan = 4)

    Continuar.grid(column = 2, row = 7, padx = (10, 10), pady = (10, 10), sticky = 'sew', columnspan = 2)

# Función que determina la ruta donde serán almacenadas las fotografías capturadas
def seleccionRutaAlmacenamiento():
    global SubVent
    
    rutaAlmacenamiento.set('')
    
    # ventana emergente en la cual solicitará la dirección de almacenamiento
    ruta = filedialog.askdirectory()

    if not ruta:
        messagebox.showwarning(' ◢ Almacenamiento de fotografías', 'No ha seleccionado la ubicación donde serán almacenadas las fotografías a capturar', parent = SubVent)
    else:
        try:
            rutaAlmacenamiento.set(ruta)
        except Exception as err:
            messagebox.showerror(' ◢ Almacenamiento de fotografías', 'Ha ocurrido el siguiente error:\n '+str(err), parent = SubVent)

# Función que controla la acción del botón x en las subventanas
def cerrarSubventana():
    global SubVent

    # Vaciando lista
    list_opcs.clear()
    SubVent.destroy()

    # Mostrando raíz
    raiz.deiconify()

# Función que inicia la captura de fotografías
def capturaFotografias2():
    # Validando los datos ingresados
    ruta = rutaAlmacenamiento.get()
    ids = 'IAAR0'+idImgs.get() #idImgs.get()
    
    # Si ha ingresado la ruta donde almacenar las fotografías
    if ruta:
        if ids:
            # Eliminado espacios en blanco
            ids = ids.replace(" ", "")

            # Eliminando caracteres especiales
            caracteres_invalidos = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]
            for caracter in caracteres_invalidos:
                ids = ids.replace(caracter, "")
            
            # Creando carpeta de archivos
            fecha = datetime.now()
            d = (str(fecha.day), '0' + str(fecha.day))[fecha.day < 10]
            m = (str(fecha.month), '0' + str(fecha.month))[fecha.month < 10]
            h = (str(fecha.hour), '0' + str(fecha.hour))[fecha.hour < 10]
            mi = (str(fecha.minute), '0' + str(fecha.minute))[fecha.minute < 10]
            s = (str(fecha.second), '0' + str(fecha.second))[fecha.second < 10]
            carpetaAlmacenamiento = ids+'_f{}_{}_{}h{}_{}_{}'.format(d, m, str(fecha.year), h, mi, s)
            ruta += '/'+carpetaAlmacenamiento

            # ___________________________________________________ Posición 0 - Ruta de almacenamiento de las fotografías a capturar
            list_opcs.append(ruta)
            
            # ___________________________________________________ Posición 1 - Nombre identificador del grupo de imgs
            list_opcs.append(ids)
            
            # Creando objeto de la cámara
            imgsMano = capturaImgs()
            imgsMano.comprobacionDispositivos()
        else:
            messagebox.showwarning(' ◢ Almacenamiento de fotografías', 'No ha ingresado el nombre Identificador de las fotografías a capturar', parent = SubVent)
    else:
        messagebox.showwarning(' ◢ Almacenamiento de fotografías', 'No ha seleccionado la ubicación donde serán almacenadas las fotografías a capturar', parent = SubVent)



# ******************************************************************************** SEGUNDA OPCIÓN "Procesamiento de fotografías"
def procesamientoFotografias(imgLogo):
    # Llamada a variable global
    global SubVent, seccion4, ruta_extraccion, id_imgs, ruta_almacenamiento_resultados, Continuar

    # Variable auxiliar que ayuda a reducir un poco el tamaño de la letra
    aux = 0

    # Ocultando raíz
    raiz.withdraw()

    SubVent = tk.Toplevel(raiz)
    SubVent.focus_force()
    SubVent.grab_set()
    SubVent.grab_release()
    SubVent.title('SCDR-AR: Procesamiento de fotografías [Ventana 1]')
    SubVent.resizable(False, False)

    #if (DatosEjec[0] == 1800 and DatosEjec[1] == 970):
    #    SubVent.geometry('{}x{}+{}+{}'.format(int(DatosEjec[0]/3), int(DatosEjec[1]/2), int((ventana[0]/2) - (DatosEjec[0]/6)), int((ventana[1]/2) - (ventana[1]/4))))
    #    aux = 0
    #else:
    #    SubVent.geometry('{}x{}+{}+{}'.format(int(DatosEjec[0]/2), int(DatosEjec[1]/2), int((ventana[0]/2) - (DatosEjec[0]/4)), int((ventana[1]/2) - (DatosEjec[1]/4))))
    #    aux = 1
    SubVent.geometry('{}x{}+{}+{}'.format(ventana[6], ventana[7], ventana[8], ventana[9]))
    aux = 2

    SubVent.deiconify()
    SubVent.config(bg = 'white', bd = 9, relief = 'groove')
    SubVent.grid_columnconfigure(0, weight = 1, uniform = 'fig')
    SubVent.grid_columnconfigure(1, weight = 1, uniform = 'fig')
    SubVent.grid_rowconfigure(0, weight = 1, uniform = 'fig')
    SubVent.protocol('WM_DELETE_WINDOW', cerrarSubventana2)

    # _________________________________________ Varibles de la subventana
    # Cadena donde serán extraídas las fotografías capturadas
    ruta_extraccion = tk.StringVar(SubVent)
    ruta_extraccion.set('')

    # Nombre que identificará al grupo de imágenes a procesar  
    id_imgs = tk.StringVar(SubVent)
    id_imgs.set('')

    # Cadena donde serán almacenados los resultado obtenidos
    ruta_almacenamiento_resultados = tk.StringVar(SubVent)
    ruta_almacenamiento_resultados.set('')

    # Creando el diseño de la subventana
    creacionSubventa('#196F3D', imgLogo)

    MiniMenu = tk.LabelFrame(seccion4, bd = 3, bg = 'white')

    # Configurando Columnas
    MiniMenu.grid_columnconfigure(0, weight = 1, uniform = 'fig')
    MiniMenu.grid_columnconfigure(1, weight = 1, uniform = 'fig')
    MiniMenu.grid_columnconfigure(2, weight = 1, uniform = 'fig')
    MiniMenu.grid_columnconfigure(3, weight = 1, uniform = 'fig')
    # Configurando Filas
    MiniMenu.grid_rowconfigure(0, weight = 1, uniform = 'fig')
    MiniMenu.grid_rowconfigure(1, weight = 1)
    MiniMenu.grid_rowconfigure(2, weight = 1)
    MiniMenu.grid_rowconfigure(3, weight = 1)
    MiniMenu.grid_rowconfigure(4, weight = 1)
    MiniMenu.grid_rowconfigure(5, weight = 1)
    MiniMenu.grid_rowconfigure(6, weight = 1)
    MiniMenu.grid_rowconfigure(7, weight = 1)
    MiniMenu.grid_rowconfigure(8, weight = 1)
    MiniMenu.grid_rowconfigure(9, weight = 1, uniform = 'fig')

    Etiqueta0 = tk.Label(MiniMenu,
                         text = ' ■ Procesamiento de fotografías ',
                         bg = '#196F3D', fg = 'white', justify = 'left',
                         font = ('Microsoft YaHei UI', DatosEjec[4]+aux, 'bold'))

    Etiqueta1 = tk.Label(MiniMenu,
                         text = ' Información para el procesamiento de las fotografías.',
                         bg = 'white', fg = 'black', justify = 'left',
                         font = ('Microsoft YaHei UI',DatosEjec[4]+aux, 'bold'))

    Etiqueta2 = tk.Label(MiniMenu,
                         text = ' 1) Indique la ubicación de donde serán extraíadas:',
                         bg = 'white', fg = 'black', justify = 'left',
                         font = ('Microsoft YaHei UI',DatosEjec[4]+aux, 'bold')) 

    Entrada1 = tk.Entry(MiniMenu,
                       textvariable = ruta_extraccion,
                       bg = 'white', fg = 'midnight blue', state = 'readonly',
                       font = ('Microsoft YaHei UI', DatosEjec[4]+aux), justify = 'center')

    Busqueda_ruta1 = tk.Button(MiniMenu,
                          text = ' Buscar ubicación ', bd = 3,
                          relief = tk.GROOVE, bg = paletaColores[3], fg = 'black', #bg = '#F7DC6F'
                          font = ('Microsoft YaHei UI', DatosEjec[4] + aux),
                          cursor = 'hand2', command = ruta_extraccion_imgs)

    Etiqueta3 = tk.Label(MiniMenu,
                         text = ' 2) Indique el nombre que identifica a las fotografías:',
                         justify = 'left', bg = 'white', fg = 'black',
                         font = ('Microsoft YaHei UI', DatosEjec[4] + aux, 'bold'))

    Entrada2 = tk.Entry(MiniMenu,
                       textvariable = id_imgs, bg = '#F0F0F0', fg = 'black', 
                       justify = 'center', relief = tk.GROOVE, border = 1,
                       font = ('Microsoft YaHei UI', DatosEjec[4] + aux))

    Etiqueta4 = tk.Label(MiniMenu,
                         text = ' 3) Ubicación donde serán almacenados los resultados:',
                         justify = 'left', bg = 'white', fg = 'black',
                         font = ('Microsoft YaHei UI',DatosEjec[4]+aux, 'bold')) 

    Entrada3 = tk.Entry(MiniMenu,
                       textvariable = ruta_almacenamiento_resultados,
                       bg = 'white', fg = 'midnight blue', state = 'readonly',
                       font = ('Microsoft YaHei UI', DatosEjec[4]+aux), justify = 'center')

    Busqueda_ruta3 = tk.Button(MiniMenu,
                          text = ' Buscar ubicación ', bd = 3,
                          relief = tk.GROOVE, bg = paletaColores[3], fg = 'black',
                          font = ('Microsoft YaHei UI', DatosEjec[4] + aux),
                          cursor = 'hand2', command = ruta_almacenamiento_rltds)

    Continuar = tk.Button(MiniMenu,
                          relief = tk.RAISED, text = ' Continuar ', 
                          bg = '#1A5276', fg = 'white',
                          font = ('Microsoft YaHei UI', DatosEjec[4] + aux),
                          cursor = 'hand2', command = procesar_fotografias2)
    
    MiniMenu.grid(column = 0, row = 0, padx = (10, 10), pady = (1, 5), sticky = 'nsew', columnspan = 2)

    Etiqueta0.grid(column = 0, row = 0, padx = (10, 10), pady = (15, 8), sticky = 'sw', columnspan = 4)
    Etiqueta1.grid(column = 0, row = 1, padx = (10, 10), pady = (0, 0), sticky = 'nw', columnspan = 4)

    Etiqueta2.grid(column = 0, row = 2, padx = (10, 10), pady = (15, 0), sticky = 'sw', columnspan = 4)
    Entrada1.grid(column = 0, row = 3, padx = (10, 10), pady = (0, 5), sticky = 'nsew', columnspan = 4)
    Busqueda_ruta1.grid(column = 0, row = 4, padx = (10, 10), pady = (0, 10), sticky = 'nw', columnspan = 4)
    
    Etiqueta3.grid(column = 0, row = 5, padx = (10, 10), pady = (10, 0), sticky = 'sw', columnspan = 4)
    Entrada2.grid(column = 0, row = 6, padx = (10, 10), pady = (0, 20), sticky = 'nsew', columnspan = 4)

    Etiqueta4.grid(column = 0, row = 7, padx = (10, 10), pady = (15, 0), sticky = 'sw', columnspan = 4)
    Entrada3.grid(column = 0, row = 8, padx = (10, 10), pady = (0, 5), sticky = 'nsew', columnspan = 4)
    Busqueda_ruta3.grid(column = 0, row = 9, padx = (10, 10), pady = (0, 10), sticky = 'nw', columnspan = 4)

    Continuar.grid(column = 2, row = 10, padx = (10, 10), pady = (10, 10), sticky = 'sew', columnspan = 2)

# Función que determina la ruta donde serán extraídas las fotografías capturadas
def ruta_extraccion_imgs():
    global SubVent

    # ventana emergente en la cual solicitará la dirección de almacenamiento
    carpeta_fotos = filedialog.askdirectory()

    if not carpeta_fotos:
        try:
            messagebox.showwarning(' ◢ Extracción de fotografías', 'No ha seleccionado la ubicación donde serán extraídas las fotografías capturadas', parent = SubVent)
            ruta_extraccion.set('')
        except Exception as err:
            print(err)
    else:
        ruta_extraccion.set(carpeta_fotos)

# Función que determina la ruta donde serán almacenadas los resultados obtenidos
def ruta_almacenamiento_rltds():
    global SubVent

    # ventana emergente en la cual solicitará la dirección de almacenamiento
    carpeta_fotos = filedialog.askdirectory()

    if not carpeta_fotos:
        try:
            messagebox.showwarning(' ◢ Extracción de fotografías', 'No ha seleccionado la ubicación donde serán almacenados los resultados obtenidos', parent = SubVent)
            ruta_almacenamiento_resultados.set('')
        except Exception as err:
            print(err)
    else:
        ruta_almacenamiento_resultados.set(carpeta_fotos)


# Función que controla la acción del botón x en las subventanas
def cerrarSubventana2():
    global SubVent

    # Vaciando lista
    list_opcs.clear()
    SubVent.destroy()

    # Mostrando raíz
    raiz.deiconify()

# Función que inicia el procesamiento de las fotografías
def procesar_fotografias2():
    # Validando los datos ingresados
    ruta_extraer = ruta_extraccion.get()
    ruta_almacenar = ruta_almacenamiento_resultados.get()
    id_grp_imgs = id_imgs.get()
    
    # Si ha ingresado la ruta donde serán extraídas las fotografías
    if ruta_extraer:
        # Si ha ingresado la ruta donde serán almacenados los resultados
        if ruta_almacenar:
            if id_grp_imgs:
                # Creando carpeta de archivos
                
                #try:
                fecha = datetime.now()
                d = (str(fecha.day), '0' + str(fecha.day))[fecha.day < 10]
                m = (str(fecha.month), '0' + str(fecha.month))[fecha.month < 10]
                a = (str(fecha.year), '0' + str(fecha.year))[fecha.year < 10]
                h = (str(fecha.hour), '0' + str(fecha.hour))[fecha.hour < 10]
                mi = (str(fecha.minute), '0' + str(fecha.minute))[fecha.minute < 10]
                s = (str(fecha.second), '0' + str(fecha.second))[fecha.second < 10]
                nomb_carpeta = 'Resultados_'+id_grp_imgs+'_f{}_{}_{}h{}_{}_{}'.format(d, m, a, h, mi, s)
                ruta_almacenar += '/'+nomb_carpeta

                # ___________________________________________________ Posición 0 - Ruta de extracción de las fotografías capturadas
                list_opcs.append(ruta_extraer)
                
                # ___________________________________________________ Posición 1 - Nombre identificador del grupo de imgs
                list_opcs.append(id_grp_imgs)

                # ___________________________________________________ Posición 2 - Ruta de almacenamiento de los resultados obtenidos
                list_opcs.append(ruta_almacenar)



                # Creando objeto de procesamiento...
                processimgs = procesamientoImgs()
                processimgs.cargaImgs()                     
                #except Exception as err:
                #    print(err)
                #    messagebox.showerror(' ◢ Extración de fotografías', 'Ha ocurrido un error dentro de la ruta del almacenamiento de las fotografías', parent = SubVent)
            else:
                messagebox.showerror(' ◢ Extración de fotografías', 'No ha ingresado el nombre Identificador de las fotografías a imgs_procesar', parent = SubVent)
        else:
            messagebox.showerror(' ◢ Extración de fotografías', 'No ha seleccionado la ubicación donde serán almacenados los resultados obtenidos', parent = SubVent)
    else:
        messagebox.showerror(' ◢ Extración de fotografías', 'No ha seleccionado la ubicación donde serán extraídas las fotografías capturadas', parent = SubVent)



# Función que controla la acción del botón x en la ventana raíz
def salir_principal():
    # Vaciando lista
    list_opcs.clear()

    #Si esta la cámara activada no podrá cerrar
    sys.exit() 



# _________________________________________________________________________________________________________________________________________
DatosEjec = []
ventana = []

# Determinando las dimensiones de las pantalla _____________________
# Identificando sistema operativo...
if platform == "linux" or platform == "linux2":
    # Sistema: Linux...
    # Dimensión de la pantalla
    args = ["xrandr", "-q", "-d", ":0"]
    proc = subprocess.Popen(args,stdout=subprocess.PIPE)
    for line in proc.stdout:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
            # Dimensión de la pantalla
            if "Screen" in line:
                DatosEjec.append(int(line.split()[7]))
                DatosEjec.append(int(line.split()[9][:-1]))

                ventana.append(int(line.split()[7]))
                ventana.append(int(line.split()[9][:-1]))
elif platform == "darwin":
    # Sistema: OS X...
    pass
elif platform == "win32":
    # Sistema: Windows...
    # Dimensión de la pantalla
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    DatosEjec.append(user32.GetSystemMetrics(0)) #Ancho
    DatosEjec.append(user32.GetSystemMetrics(1)) #Alto

    ventana.append(user32.GetSystemMetrics(0)) #Ancho
    ventana.append(user32.GetSystemMetrics(1)) #Alto

# Resoluciones de la ventana en donde esta siendo ejecutado el software
if(DatosEjec[0] >= 1920 and DatosEjec[1] >= 1080):
    #1920*1080
    #____ Medidas de la ventana principal
    #raiz.geometry('1800x970+50+25')
    DatosEjec[0] = 1800    # - 0
    DatosEjec[1] = 970     # - 1
    #Posición
    DatosEjec.append(50)   # X - 2
    DatosEjec.append(25)   # Y - 3
    #Fuente
    DatosEjec.append(12)   # - 4
    #tamaño (labels - Info)
    DatosEjec.append(50)   # - 5
    #Posición (labels - Info)
    DatosEjec.append(1200) # X - 6  Antes: 1200
    DatosEjec.append(400)  # Y - 7  Antes: 440

    #____ Medidas de la Subventana
    #SubVent.geometry('700x300+590+300')
    DatosEjec.append(700)  # 8
    DatosEjec.append(300)  # 9
    #Posición
    DatosEjec.append(590)  # X - 10
    DatosEjec.append(300)  # Y - 11
    #Fuente
    DatosEjec.append(12)  # - 12
    #Saltos
    DatosEjec.append(40)  # - 13

    #elif(Datosventana[0] >= 1280 and Datosventana[1] >= 720):
elif(DatosEjec[0] >= 1200 and DatosEjec[1] >= 800):
    #1280*720
    #____ Medidas de la ventana principal
    #raiz.geometry('1800x970+50+25')
    DatosEjec[0] = 1220    # - 0
    DatosEjec[1] = 620     # - 1
    #Posición
    DatosEjec.append(23)   # X - 2
    DatosEjec.append(25)   # Y - 3
    #Fuente
    DatosEjec.append(10)   # - 4
    #tamaño (labels - Info)
    DatosEjec.append(35)   # - 5
    #Posición (labels - Info)
    DatosEjec.append(830) # X - 6   Antes: 830
    DatosEjec.append(180)  # Y - 7  Antes: 220

    #____ Medidas de la Subventana
    #SubVent.geometry('700x300+590+300')
    DatosEjec.append(700)  # 8
    DatosEjec.append(300)  # 9
    #Posición
    DatosEjec.append(310)  # X - 10
    DatosEjec.append(170)  # Y - 11
    #Fuente
    DatosEjec.append(12)  # - 12
    #Salto
    DatosEjec.append(33)  # - 13
else:
    #1024*768
    #____ Medidas de la ventana principal
    #raiz.geometry('1800x970+50+25')
    DatosEjec[0] = 1000    # - 0
    DatosEjec[1] = 670     # - 1
    #Posición
    DatosEjec.append(5)   # X - 2
    DatosEjec.append(10)  # Y - 3
    #Fuente
    DatosEjec.append(10)   # - 4
    #tamaño (labels - Info)
    DatosEjec.append(10)   # - 5
    #Posición (labels - Info)
    DatosEjec.append(530) # X - 6   Antes: 830
    DatosEjec.append(180)  # Y - 7  Antes: 220 180

    #____ Medidas de la Subventana
    #SubVent.geometry('700x300+590+300')
    DatosEjec.append(700)  # 8
    DatosEjec.append(280)  # 9
    #Posición
    DatosEjec.append(160)  # X - 10
    DatosEjec.append(170)  # Y - 11
    #Fuente
    DatosEjec.append(12)  # - 12
    #Salto
    DatosEjec.append(33)  # - 13

# Restando un 5% de la longitud en ancho y 10% de alto de la resolución total de la pantalla
ventana.append(int(ventana[0]-(ventana[0]*.05)))      # 2 -- Ancho - Ventana - GRANDE
ventana.append(int(ventana[1]-(ventana[1]*.10)))      # 3 -- Alto - Ventana - GRANDE
ventana.append(int((ventana[0]/2) - (ventana[2]/2)))  # 4 -- Posición X - Ventana - GRANDE 
ventana.append(int(ventana[1]*.01))                   # 5 -- Posición Y - Ventana - GRANDE

ventana.append(int(ventana[0]-(ventana[0]*.35))) #25   # 6 -- Ancho - Ventana - MEDIANA
ventana.append(int(ventana[1]-(ventana[1]*.45))) #40   # 7 -- Alto - Ventana - MEDIANA
ventana.append(int((ventana[0]/2) - (ventana[6]/2)))   # 8 -- Posición X - Ventana - MEDIANA
ventana.append(int((ventana[1]/2) - (ventana[7]/2)))   # 9 -- Posición Y - Ventana - MEDIANA


ventana.append(int(ventana[0]-(ventana[0]*.65))) #25   # 10 -- Ancho - Ventana - MEDIANA-VERTICAL
ventana.append(int(ventana[1]-(ventana[1]*.40))) #40   # 11 -- Alto - Ventana - MEDIANA-VERTICAL
ventana.append(int((ventana[0]/2) - (ventana[10]+10))) # 12 -- Posición X - Ventana - MEDIANA-VERTICAL
ventana.append(int((ventana[1]/2) - (ventana[11]/2)))  # 13 -- Posición Y - Ventana - MEDIANA-VERTICAL



ventana.append(int(ventana[0]-(ventana[0]*.65))) #25   # 14 -- Ancho - Ventana - MEDIANA-VERTICAL
ventana.append(int(ventana[1]-(ventana[1]*.20))) #40   # 15 -- Alto - Ventana - MEDIANA-VERTICAL
ventana.append(int((ventana[0]/2) - (ventana[14]/2)))  # 16 -- Posición X - Ventana - MEDIANA-VERTICAL
ventana.append(int((ventana[1]/2) - (ventana[15]/2)))  # 17 -- Posición Y - Ventana - MEDIANA-VERTICAL



#ventana.append(int(ventana[0]-(ventana[0]*.55)))      # 10 -- Ancho - Ventana - MEDIANA (Vertical 1 - Solicitand datos)
#ventana.append(int(ventana[1]-(ventana[1]*.20)))      # 11 -- Alto - Ventana - MEDIANA (Vertical 1 - Solicitand datos)
#ventana.append(int((ventana[0]/2) - (ventana[10]/2))) # 12 -- Posición X - Ventana - MEDIANA (Vertical 1 - Solicitand datos)
#ventana.append(int((ventana[1]/2) - (ventana[11]/2))) # 13 -- Posición Y - Ventana - MEDIANA (Vertical 1 - Solicitand datos)




# ____________________________ CREACIÓN DE ventana PRINCIPAL/INICIAL ______________________________
raiz = tk.Tk()
raiz.resizable(0, 0)
raiz.title(' SCDR-AR (Version 4.0) ')
raiz.geometry('{}x{}+{}+{}'.format(ventana[6], ventana[7], ventana[8], ventana[9]))
raiz.protocol('WM_DELETE_WINDOW', salir_principal)
raiz.iconphoto(True, tk.PhotoImage(file='iconos/Logo.png'))
raiz.config(bg = 'white', bd = 8, relief = 'groove')

raiz.grid_columnconfigure(0, weight = 1, uniform = 'fig')
raiz.grid_columnconfigure(1, weight = 1, uniform = 'fig')
raiz.grid_rowconfigure(0, weight = 1, uniform = 'fig')

aux = 3 #1
if(DatosEjec[0] == 1800 and DatosEjec[1] == 970):
    aux = 0

# --------------------- Agregando panel de botones (Logotipo)
seccion1 = tk.Frame(raiz)
#Configuración de columnas
seccion1.grid_columnconfigure(0, weight = 1,uniform = 'fig')
seccion1.grid_columnconfigure(1, weight = 1,uniform = 'fig')
seccion1.grid_columnconfigure(2, weight = 1,uniform = 'fig')
#Configuración de filas
seccion1.grid_rowconfigure(0, weight = 1,uniform = 'fig')
seccion1.grid_rowconfigure(1, weight = 1)
seccion1.grid_rowconfigure(2, weight = 1,uniform = 'fig')
seccion1.grid_rowconfigure(3, weight = 1,uniform = 'fig')
seccion1.grid_rowconfigure(4, weight = 1,uniform = 'fig')



#imgLogo = Image.open('iconos/Logo.png')                # Load the image
#imgLogo = imgLogo.resize((500, 500), Image.ANTIALIAS)  # Resize the image in the given (width, height)
#imgLogo = ImageTk.PhotoImage(imgLogo, master = seccion1) 
imgLogo = tk.PhotoImage(file = 'iconos/Logo_1.png', master = seccion1)
logo = tk.Label(seccion1, image = imgLogo, bg = paletaColores[0])

label1 = tk.Label(seccion1, text=' Sistema Clasificador de Daño Radiológico por AR', 
                    bg = paletaColores[0], fg = 'white', font = ('Microsoft YaHei UI', 15, 'bold'))

label2 = tk.Label(seccion1, text=' Versión 1.4 - Agosto 2023 ', 
                    bg = paletaColores[0], fg = 'white', font = ('Microsoft YaHei UI', 12, 'bold'))

logo.grid(column = 0, row = 1, padx = (10, 10), pady = (1, 1), columnspan = 3, sticky = 'nswe')
label1.grid(column = 0, row = 2, padx = (10, 10), pady = (1, 1), columnspan = 3, sticky = 'swe')
label2.grid(column = 0, row = 3, padx = (10, 10), pady = (1, 10), columnspan = 3, sticky = 'nwe')

# --------------------- Agregando panel de botones (Procesamiento)
seccion2 = tk.Frame(raiz)
#Configuración de columnas
seccion2.grid_columnconfigure(0, weight = 1,uniform = 'fig')
#Configuración de filas
seccion2.grid_rowconfigure(0, weight = 1,uniform = 'fig')
seccion2.grid_rowconfigure(1, weight = 1)
seccion2.grid_rowconfigure(2, weight = 1,uniform = 'fig')


seccion3 = tk.Frame(seccion2)
#Configuración de columnas
seccion3.grid_columnconfigure(0, weight = 1, uniform = 'fig')
seccion3.grid_columnconfigure(1, weight = 1, uniform = 'fig')
#Configuración de filas
seccion3.grid_rowconfigure(0, weight = 1)
seccion3.grid_rowconfigure(1, weight = 1)
seccion3.grid_rowconfigure(2, weight = 1)
seccion3.grid_rowconfigure(3, weight = 1)

seccion3.config(bg = paletaColores[4], bd = 0, relief = 'raised')
seccion3.grid(column = 0, row = 1, padx = (20, 20), pady = (0, 0), columnspan = 1, rowspan = 1, sticky = 'nswe')



label1 = tk.Label(seccion3, text = ' Seleccione la acción que desea realizar: ', justify = 'left', 
                    bg = paletaColores[4], fg = 'black', font = ('Microsoft YaHei UI', 16))

btn1 = tk.Button(seccion3, text = ' Capturar fotografías ', cursor = 'hand2', relief = tk.RAISED, 
                borderwidth = 3, bg = '#1A5276', fg='white', font = ('Microsoft YaHei UI', 16), command = lambda:capturaFotografias(imgLogo))

btn2 = tk.Button(seccion3, text = ' Procesar fotografías', cursor = 'hand2', relief = tk.RAISED, 
                borderwidth = 3, bg = '#196F3D', fg='white', font = ('Microsoft YaHei UI', 16), command = lambda:procesamientoFotografias(imgLogo))

label1.grid(column = 0, row = 1, padx = (10, 10), pady = (0, 0), columnspan = 2, sticky = 'sew')
btn1.grid(column = 0, row = 2, padx = (10, 10), pady = (10, 0), columnspan = 1, sticky = 'new')
btn2.grid(column = 1, row = 2, padx = (10, 10), pady = (10, 0), columnspan = 1, sticky = 'new')


# Añadiendo paneles a la ventana
seccion1.config(bg = paletaColores[0], bd = 0, relief = 'raised')
seccion1.grid(column = 0, row = 0, padx = (0, 0), pady = (0,0), columnspan = 1, sticky = 'nsew')

seccion2.config(bg = paletaColores[4], bd = 0, relief = 'raised')
seccion2.grid(column = 1, row = 0, padx = (0, 0), pady = (0,0), columnspan = 1, sticky = 'nsew')

raiz.mainloop() # Visualizando ventana inicial