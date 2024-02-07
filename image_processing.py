print('\n','===='*10,end='\n')
import cv2
import numpy as np
import os, json, csv, time, shutil
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
print('\n - Preparando modelo YOLOv5: ', end = ' ')
import ModelYolov5.detect as detectYolov5
print('  Ok ', end='\n')	


# ____________________ Declaración de funciones:
# Extracción de la ruta de las images
def imagePathExtraction(dir_path):
	# Lista que almacenará la ruta completa de la ubicación de las imágenes
	images = []
 
	# Recorrer los archivos en la carpeta
	for file in os.listdir(dir_path):
		# Verificar si el archivo es una imagen (puedes agregar más extensiones según sea necesario)
		if file.lower().endswith(('.png', '.jpg', '.jpeg')):
			pathComplete =  os.path.normpath(os.path.join(dir_path, file))
			images.append(pathComplete)

	return images



# Creador de archivo CSV en el cual serán almacenados los resultados obtenidos a partir del procesamiento de las imágenes por el modelo MediaPipe
def dataresults(bandDataset, idCSV, path_save):
	# Creando CSV para almacenar los resulados de todas las manos procesadas
	dataResults = pd.DataFrame()

	dataResults['nombImg'] = None
	for finger in ['Index', 'Middle', 'Ring', 'Little', 'Thumb']:
		if finger != 'Thumb':
			keys = ['CMC-MCP', 'MCP-PIP', 'PIP-DIP', 'DIP-TIP']
			for key in keys:
				dataResults['dist_'+key+'_'+finger] = None
	
			keys = ['MCP', 'PIP', 'DIP', 'TIP']
			for key in keys:
				dataResults['angl_'+key+'_'+finger] = None
		else:
			keys = ['CMC-MCP', 'MCP-DIP', 'DIP-TIP']
			for key in keys:
				dataResults['dist_'+key+'_'+finger] = None
	
			keys = ['MCP', 'DIP', 'TIP']
			for key in keys:
				dataResults['angl_'+key+'_'+finger] = None
	 	
	for finger in ['Index', 'Middle', 'Ring', 'Little', 'Thumb']:
		dataResults['sumFinger-'+finger+'-dists'] = 0
		dataResults['meanFinger-'+finger+'-dists'] = 0
		dataResults['sumFinger-'+finger+'-angls'] = 0
		dataResults['meanFinger-'+finger+'-angls'] = 0
		
  
	dataResults['sumFingers-dists'] = 0
	dataResults['meanFingers-dists'] = 0
	dataResults['sumFingers-angls'] = 0
	dataResults['meanFingers-angls'] = 0
	dataResults['output'] = None
	
	'''# __ Mediciones en distancias
	dataResults['sumFinger-index-dists'] = None
	dataResults['sumFinger-middle-dists'] = None
	dataResults['sumFinger-ring-dists'] = None
	dataResults['sumFinger-little-dists'] = None
	dataResults['sumFinger-thumb-dists'] = None
	dataResults['sumFingers-dists'] = None

	dataResults['meanFinger-index-dists'] = None
	dataResults['meanFinger-middle-dists'] = None
	dataResults['meanFinger-ring-dists'] = None
	dataResults['meanFinger-little-dists'] = None
	dataResults['meanFinger-thumb-dists'] = None
	dataResults['meanFingers-dists'] = None
	
 	# __ Mediciones en distancias
	dataResults['sumFinger-index-angls'] = None
	dataResults['sumFinger-middle-angls'] = None
	dataResults['sumFinger-ring-angls'] = None
	dataResults['sumFinger-little-angls'] = None
	dataResults['sumFinger-thumb-angls'] = None
	dataResults['sumFingers-angls'] = None

	dataResults['meanFinger-index-angls'] = None
	dataResults['meanFinger-middle-angls'] = None
	dataResults['meanFinger-ring-angls'] = None
	dataResults['meanFinger-little-angls'] = None
	dataResults['meanFinger-thumb-angls'] = None
	dataResults['meanFingers-angls'] = None'''

  				
	if bandDataset == 0:
		nombfile = path_save+'/'+idCSV+'_data_results_hand_CMC.csv'
		print('\n - Creando archivo CSV que almacenará las métricas calculadas de la imagen de la mano: \n  '+nombfile)
	else:
		nombfile = path_save+'/'+idCSV+'_data_results_hand_MCP.csv'
		print('\n - Creando archivo CSV que almacenará las métricas calculadas de la imagen de la mano (cada dedo): \n  '+nombfile)

	try:
		dataResults.to_csv(nombfile, index=False, index_label=None)
	except Exception as err:
		dataResults.clear()
		print(' * Error en la creación del archivo CSV para el almacenamiento de los resultados obtenidos: \n ', err)
		return None

	return dataResults.to_dict(), nombfile

# Determina la coordenadas en X, Y de cada articulación identificada por los puntos de MediaPipe
def pointIdentification(pointsMediaPipe, w, h):
	points = []
 
	# Validando si fue identificada la mano en cualquiera de los 2 tipos de imágenes
	if pointsMediaPipe.multi_hand_landmarks is not None:
		hand_landmarks = pointsMediaPipe.multi_hand_landmarks[0]
		
		for point in range(0, 21):
			# * Coordenadas de las articulaciones de los dedos
			# Muñeca: 0 | Pulgar: 1-4 | Indice: 5-8 | Medio: 9-12 | Anular: 13-16 | meñique: 17-20
   
			pointsJoint = hand_landmarks.landmark[point]
			pointsJoint = [int(pointsJoint.x * w), int(pointsJoint.y * h)]
			points.append(pointsJoint)
	
	return points

# Realizado el recorte de la sección de la mano a partir de los puntos calculados por MediaPipe
def cut_outHandSection(image, pointsJoints_MdPp, pixelsForCropping):
	points = pointsJoints_MdPp.copy()#[: -1]

	# Alto y Ancho de la imagen
	h, w, _ = image.shape
 
	# Coordenas en X de los puntos detectados por MediaPipe
	coordX = [x for x, _ in points]
	coordXSeccHand = [np.min(coordX), np.max(coordX)]
	coordXSeccHand[0] = np.clip(coordXSeccHand[0]-pixelsForCropping, 0, w)
	coordXSeccHand[1] = np.clip(coordXSeccHand[1]+pixelsForCropping, 0, w)

 	# Coordenas en Y de los puntos detectados por MediaPipe
	coordY = [y for _, y in points]
	coordYSeccHand = [np.min(coordY), np.max(coordY)]
	coordYSeccHand[0] = np.clip(coordYSeccHand[0]-pixelsForCropping, 0, h)
	coordYSeccHand[1] = np.clip(coordYSeccHand[1]+pixelsForCropping, 0, h)
	
	# Recortando la sección que corresponde a la mano
	imgHand = image[coordYSeccHand[0]:coordYSeccHand[1], 
				  	coordXSeccHand[0]:coordXSeccHand[1]]
 
	#Redimensionamiento de la imagen
	#imgHand = cv2.resize(imgHand, (w_r, h_r), interpolation = cv2.INTER_AREA)
 
	return imgHand

# Función que rota las imágenes de manera vertical para finalmente retornarla junto con los puntos identificados por MediaPipe
def hand_landmarksMediaPipe(file):
	# Aplicando modelo MediaPipe
	img = cv2.imread(file)

	# Alto y Ancho de la imagen
	h, w, _ = img.shape
	
	# Verficando que imagen es procesada (A partir de su resolución)
	if h == 1080 and w == 1920:
		# Imagen WebCam
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # Rotando Imagen 90°
		cv2.imwrite(file, img)

		# Alto y Ancho de la imagen
		h, w, _ = img.shape

		# Girando a espejo
		img = cv2.flip(img, 1)

		# Invirtiendo canales de la imagen para procesar Img por MediaPipe
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Procesando la imagen con MediaPipe (Identificando puntos)
		results = handMediaPipe.process(cv2.blur(img.copy(), (7,7), cv2.BORDER_DEFAULT))
		
		# MediaPipe no pudó identificar a la mano en la imagen
		if results.multi_hand_landmarks is None:
			print('No ha sido reconocida la mano la 1° vez...')
			return None, None

		# --- Predicción
		for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
			# Obtener la información sobre la mano (izquierda o derecha)
			handedness = results.multi_handedness[idx]
			# Acceder a la propiedad hand_label para obtener la etiqueta de la mano (izquierda o derecha)
			hand_label = handedness.classification[0].label

		# Girando a espejo (aspecto original)
		img = cv2.flip(img, 1)
  
		# Procesando la imagen con MediaPipe (Identificando puntos)
		results = handMediaPipe.process(cv2.blur(img.copy(), (7,7), cv2.BORDER_DEFAULT))
	else:
		if h != w:
			if h > 1920 and w > 1080:
				img = cv2.resize(img, (1080, 1920), interpolation = cv2.INTER_AREA)
				h, w, _ = img.shape
			elif h > 1080 and w > 1920:
				img = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
				h, w, _ = img.shape
	
		# Si la imagen fue capturada horizontalmente
		if w > h:
			img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # Rotando Imagen 90°
			
			# Alto y Ancho de la imagen
			h, w, _ = img.shape

		# Girando a espejo
		img = cv2.flip(img, 1)

		# Punto central de la imagen
		centroImg = (int(w/2), int(h/2))

		# Creación de secciones para la orientación de la imagen
		#_____ Primera REGION - Vertical
		region1 = [(0, 0), (w, centroImg[1])]

		#_____ Primera REGION - Horizontal
		region3 = [(0, 0), (centroImg[0], h)]

		# Invirtiendo canales de la imagen para procesar Img por MediaPipe
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Procesando la imagen con MediaPipe (Identificando puntos)
		results = handMediaPipe.process(cv2.blur(img.copy(), (7,7), cv2.BORDER_DEFAULT))

		# MediaPipe no pudó identificar a la mano en la imagen
		if results.multi_hand_landmarks is None:
			print('No ha sido reconocida la mano la 1° vez...')
			return None, None
		else:
			# --- Predicción
			for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
				# Obtener la información sobre la mano (izquierda o derecha)
				handedness = results.multi_handedness[idx]
				# Acceder a la propiedad hand_label para obtener la etiqueta de la mano (izquierda o derecha)
				hand_label = handedness.classification[0].label
			
			# Validando si fue identificada la mano en la imagen
			if results.multi_hand_landmarks:
				# Resultados de lo identificado por MediaPipe
				hand_landmarks = results.multi_hand_landmarks[0]
				
				# Coordenas de la muñeca en la imagen
				x = int((hand_landmarks.landmark[0].x) * w)
				y = int((hand_landmarks.landmark[0].y) * h)
				
				# Coordenas de MCP dedo medio
				x2 = int((hand_landmarks.landmark[9].x) * w)
				y2 = int((hand_landmarks.landmark[9].y) * h)

				# Punto medio
				x_middle = int((x+x2)/2)
				y_middle = int((y+y2)/2)

				# Calculando pendiente
				dividend = (x-x2)
				if dividend == 0:
					dividend = 1
				
				m = (y-y2)/dividend
				if (m == 0):
					m = 1
				
				m_perpendicular = int(-1/(m))

				# Indicando que la línea se encuentra verticalmente
				if m_perpendicular == 0:# rotado:
					# Vericando si la imagen esta de cabeza (region 1)
					if (-w <= x_middle <= region1[1][0] and -h <= y_middle <= region1[1][1]):
						# Girando a espejo (aspecto original)
						img = cv2.flip(img, 1)
						img = cv2.flip(img, 0)
				else:
					# Rotación de 90 grados de la imagen, en sentido horario
					img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # Rotando Imagen 90°
					
					# Vericando si la imagen esta a la izquierda (region 3)
					if (-w <= x_middle <= region3[1][0] and -h <= y_middle <= region3[1][1]):
						# Girando a espejo (aspecto original)
						img = cv2.flip(img, 1)
						img = cv2.flip(img, 0)

				# Girando a espejo (aspecto original)
				img = cv2.flip(img, 1)
				cv2.imwrite(file, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
				
				# Procesando la imagen con MediaPipe (Identificando puntos)
				results = handMediaPipe.process(cv2.blur(img.copy(), (7,7), cv2.BORDER_DEFAULT))

	# Alto y Ancho de la imagen
	h, w, _ = img.shape
 
	pixelsForCropping = 500
	if w == 1080 and h == 1920:
		pixelsForCropping = 200
	
	cv2.namedWindow('imgMano', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('imgMano', h_w, w_w)
	cv2.imshow('imgMano', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	cv2.waitKey(0)
	
	# Determinando la coordenadas correspondientes a cada articulación
	pointsJoints = pointIdentification(results, w, h)
	
	# Recortando la sección de la mano
	img = cut_outHandSection(img, pointsJoints, pixelsForCropping)
 
	cv2.namedWindow('RecorteMano', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('RecorteMano', h_w, w_w)
	cv2.imshow('RecorteMano', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
 
   	# Procesando la imagen con MediaPipe (Identificando puntos)
	results = handMediaPipe.process(cv2.blur(img.copy(), (7,7), cv2.BORDER_DEFAULT))
	
 	# MediaPipe no pudó identificar a la mano en la imagen
	if results.multi_hand_landmarks is None:
		print('No ha sido reconocida la mano después del recorte...')
		return None, None
	
	# Alto y Ancho de la imagen
	h, w, _ = img.shape
	
	#plt.imshow(img)
	#plt.title('img recortada'+str(img.shape))
	#plt.show()
	
	# ******* Detectando nuevamente los puntos a la imagen recortada	
	pointsJoints = pointIdentification(results, w, h)
	
	#Redimensionamiento de la imagen
	img = cv2.resize(img, (w_r, h_r), interpolation = cv2.INTER_AREA)
	
	#plt.imshow(img)
	#plt.title('img redimensionada'+str(img.shape))
	#plt.show()

	cv2.namedWindow('SeccionMano', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('SeccionMano', h_w, w_w)
	cv2.imshow('SeccionMano', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
 
	for pointJoint in pointsJoints:
		x, y = pointJoint[0], pointJoint[1]
		pointJoint[0], pointJoint[1] = int((x/w)*w_r), int((y/h)*h_r)
	
 	# Orientación de la mano
	pointsJoints.append(hand_label)
	
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	return img, pointsJoints

# Función que por medio del modelo Yolov5, recorta de la imagen las secciones que puedan ser observadas en la imagen
def cut_outJointSections(pathImg, path_results, nameFile):	
	detectYolov5.run(weights = 'ModelYolov5/WEIGHTS_TRAINED/Yolov5_RGB.pt',
				source = pathImg, name = nameFile+'_resultsYolo', 
				data = 'ModelYolov5/WEIGHTS_TRAINED/customdataRGB.yaml',
				project = path_results, save_crop = True, # save cropped prediction boxes
				#save_txt = True,  # clases, ubicaciones
				#save_csv = True,  # save results in CSV format
				#save_conf = True,  # save confidences in --save-txt labels
				conf_thres = 0.50, iou_thres=0.5, agnostic_nms=True, max_det = 11)
	
	# Redimensionando los recortes obtenidos por Yolo
	try:
		for joint in ['DIP','PIP2','PIP3','PIP4','PIP5','MCP1','MCP2','MCP3','MCP4','MCP5','CMC']:
			pathImgJoint = path_results+'/'+nameFile+'_resultsYolo/crops/'+joint+'/'+nameFile+'.jpg'
			pathImgJoint2 = path_results+'/'+nameFile+'_resultsYolo/crops/'+joint+'/'+joint+'.jpg'
   
			if not (os.path.exists(pathImgJoint) or os.path.isfile(pathImgJoint)):
				# No existe esta articulación dentro de las que identifico Yolov5
				try:
					print('\n > Carpeta que contendrá una pseudo imagen de la articulación ['+joint+'] faltante  ', end='\r')
					os.mkdir(path_results+'/'+nameFile+'_resultsYolo/crops/'+joint)
					print('\n   Carpeta creada...  \n')
				except FileExistsError as err:
					print('\n   La carpeta ya habia sido creada previamente...  \n')
				except Exception as err:
					print('\n [-- Error en ejecución --]')
					print(f" - Error: {err}")
					print(f" {type(err).__name__}\n\n")
					break
				
				imgJoint = np.zeros((256, 256, 3), dtype = 'uint8')
				imgJoint = cv2.cvtColor(imgJoint, cv2.COLOR_BGR2GRAY)
			else:
				imgJoint = cv2.imread(pathImgJoint)
				'''cv2.namedWindow('imgJoint', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('imgJoint', h_w, w_w)
				cv2.imshow('imgJoint', imgJoint)
				cv2.waitKey(0)
				cv2.destroyAllWindows()'''

				# Redimensionamiento de la imagen identificada
				imgJoint = cv2.resize(imgJoint, (256, 256), interpolation = cv2.INTER_AREA)
			
			cv2.imwrite(pathImgJoint2, imgJoint)
	except Exception as err:
		print('\n [-- Error durante la manipulación de las \"imgs Joint\" de los resultados obtenidos por YoloV5 ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None



# ____ Procesamiento de la imagen
# Función que realiza el calculo del angulo dado entre dos vectores
def angleBetweenVectors(pointOrigin, p1, p2):
	# Calcular los vectores a partir de los puntos
	vector1 = np.array([pointOrigin[0] - p1[0], pointOrigin[1] - p1[1]])
	vector2 = np.array([pointOrigin[0] - p2[0], pointOrigin[1] - p2[1]])

	# Calcular el producto punto entre los vectores
	scalarProduct = np.dot(vector1, vector2)

	# Calcular las magnitudes de los vectores
	vectorQuantity1 = np.linalg.norm(vector1)
	vectorQuantity2 = np.linalg.norm(vector2)

	# Calcular el coseno del ángulo entre los vectores
	dividend = (vectorQuantity1 * vectorQuantity2)
	if dividend == 0:
		dividend = 1
 
	coseno_theta = scalarProduct / dividend

	# Calcular el ángulo en radianes
	theta_radianes = np.arccos(np.clip(coseno_theta, -1.0, 1.0))

	# Convertir a grados
	theta_grados = np.degrees(theta_radianes)

	return theta_grados

# * Función para determinar el punto de intersección entre las 2 rectas de referencia
def puntoInterseccionEjesRef(punto1_recta1, punto2_recta1, punto1_recta2, punto2_recta2):
	x1, y1 = punto1_recta1
	x2, y2 = punto2_recta1

	x3, y3 = punto1_recta2
	x4, y4 = punto2_recta2

	# Calcular pendientes
	dv1 = (x2 - x1)
	if dv1 == 0:
		dv1 = 1
	
	dv2 = (x4 - x3)
	if dv2 == 0:
		dv2 = 1

	m1 = (y2 - y1) / dv1 #(x2 - x1)
	m2 = (y4 - y3) / dv2 #(x4 - x3)

	m3 = (m1 - m2)
	if m3 == 0:
		m3 = 1

	# Calcular coordenadas del punto de intersección
	x_interseccion = (m1 * x1 - m2 * x3 + y3 - y1) / m3
	y_interseccion = m1 * (x_interseccion - x1) + y1

	return (int(x_interseccion), int(y_interseccion))

# Función para determinar una línea perpendicular a partir de una línea proyecta por 2 puntos
def lineaPerpendicular(x1, y1, x2, y2, band, ancho_recta):
	# Centro de la recta de referencia - punto medio
	centro_x = int((x1 + x2)/2)
	centro_y = int((y1 + y2)/2)

	# ___ Proyectando una línea perpendicular en el eje de referencia
	# Calcula la pendiente de la línea original
	dividend = x2 - centro_x
	if dividend == 0:
		dividend = 1
	
	m = ((y2+1) - centro_y)/(dividend)
	if m == 0:
		m = 1
	
	# Calcula la pendiente de la línea perpendicular que es
	m_perpendicular = -1/m

	# Calcula los extremos de la línea perpendicular
	if m_perpendicular == float('inf'):
		x3, y3 = centro_x + ancho_recta, centro_y
		x4, y4 = centro_x - ancho_recta, y3
	else:
		x3 = centro_x - ancho_recta
		x4 = centro_x + ancho_recta
		
		if(m_perpendicular >= 0):
			x3 = centro_x + ancho_recta
			x4 = centro_x - ancho_recta

		y3 = int(centro_y + m_perpendicular * (x3 - centro_x))
		y4 = int(centro_y + m_perpendicular * (x4 - centro_x))

	media_y = int((y3+y4)/2) # Mitad entre las coordenadas "y" de los puntos
	aux1 = abs(media_y - y2)

	if band == 1:
		y3 += aux1
		y4 += aux1
	else:
		# Coordenadas del punto destino hacia el cual se desea mover la línea perpendicular
		p_destino = (x1, y1)

		# Calcular el vector de desplazamiento
		delta_x = p_destino[0] - centro_x
		delta_y = p_destino[1] - centro_y

		# Desplazar la línea perpendicular hacia el punto destino
		#if x3 > x4 and y3 > y4:
		x3, y3 = (x3 - delta_x), (y3 - delta_y)
		x4, y4 = (x4 - delta_x), (y4 - delta_y)
		#else:
		#	x3, y3 = (x3 - delta_x), (y3 - delta_y)
		#	x4, y4 = (x4 + delta_x), (y4 + delta_y)

	if x3 > x4 and y3 > y4:
		aux_x, aux_y = x3, y3
		x3, x4 = x4, aux_x
		y3, y4 = y4, aux_y

	return x3, x4, y3, y4

def debuggerObjects(output_image2):
	obj, output, stats, _ = cv2.connectedComponentsWithStats(output_image2, connectivity=4)
	obj -= 1
	sizes = stats[1:, -1]  # Áreas (en píxeles) de los objetos identificados
	sortSizes = np.sort(sizes, None) # Ordenando áreas de menor a mayor
	#print(sortSizes)
	
	if(len(sortSizes) > 1):
		min_size = sortSizes[-1] # Área maxima
		output_image2 = np.zeros((output_image2.shape), dtype = 'uint8')
		for j in range(0, obj):
			if sizes[j] >= min_size:                  # Solo dejar a los objetos cuya área en píxeles sea mayor la indicada como mínima
				output_image2[output == j + 1] = 255
    
	return output_image2
	
# Función para detectar la mano y crear una máscara binaria.
def imgBinHandCurveConvex(image):
    # EE
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
 
	# Convertir la imagen a escala de grises
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9,9))

	# Aplicar un desenfoque para reducir el ruido
	blurred = cv2.GaussianBlur(clahe.apply(gray), (9, 9), 0)
	_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# Realiza la operación de apertura
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
	
	# Detección de contornos
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Convertir a BGR para dibujar contornos en color
	output_image = cv2.drawContours(gray, contours, -1, (255,255,255), 5)  # Contornos en verde con grosor de 2
	_, output_image = cv2.threshold(output_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	# Alto y Ancho de la imagen
	h, w, _ = image.shape
	pruebaImg = output_image[5:h-5, 5:w-5]
	deb1 = debuggerObjects(pruebaImg)#output_image.copy())
	cv2.namedWindow('DepuradorBin', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('DepuradorBin', h_w, w_w)
	cv2.imshow('DepuradorBin', deb1)
	
	deb2 = debuggerObjects(255-deb1)
	cv2.namedWindow('DepuradorBin', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('DepuradorBin', h_w, w_w)
	cv2.imshow('DepuradorBin', deb2)
	
	output_image2 = debuggerObjects(255-deb2)
	cv2.namedWindow('Binarizada', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Binarizada', h_w, w_w)
	cv2.imshow('Binarizada', output_image2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
 
	#output_image_ones = np.zeros((h+5, w), dtype = 'uint8')

	# Insertar 'small_image' en 'large_image'
	#output_image_ones[5:h, 0:w] = output_image2
	contours, _ = cv2.findContours(output_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
 	# Calcular la envoltura convexa del contorno
	for contour in contours:
		hull = cv2.convexHull(contour)

	# Dibujar la Convex Hull
	output_image2 = cv2.cvtColor(output_image2, cv2.COLOR_GRAY2RGB)
	cv2.polylines(output_image2, [hull], True, (0, 255, 0), 6)

	return output_image2

# Función que traza curvas convexas a partir de las puntas de los dedos
def imgHandCurveConvex(convexPoints, convexPoints2, imgProyections, img_orginal): 
 	# Encontrar el casco convexo
	points = np.array(convexPoints, dtype=np.int32)
	allpoints = np.array(convexPoints2, dtype=np.int32)
 
	hull = cv2.convexHull(points)
	hull_all = cv2.convexHull(allpoints)
	
	'''cv2.namedWindow('DistanciasOriginal', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('DistanciasOriginal', h_w, w_w)
	cv2.imshow('DistanciasOriginal', imgProyections)
	cv2.waitKey(0)
	cv2.destroyAllWindows()'''
	
	# Dibujar el casco convexo
	imgProyections1 = cv2.polylines(imgProyections.copy(), [hull], isClosed=True, color=(255, 255, 255), thickness=5)
	imgProyections1_1 = cv2.polylines(imgProyections.copy(), [hull_all], isClosed=True, color=(255, 255, 255), thickness=5)
	
	# Rellenando el casco convexo
	imgProyections2 = np.zeros((h_r*2, w_r*2, 3), dtype = 'uint8')
	_,th = cv2.threshold(cv2.cvtColor(imgProyections1.copy(),cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
	contorno,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
 	# Crear una imagen en blanco para el relleno
	filled_image = np.zeros_like(cv2.drawContours(imgProyections2, contorno, -1, (0,255,0), 3))
	
	# Rellenar el contorno con cv2.fillPoly
	imgProyections2 = cv2.fillPoly(filled_image, [np.array(contorno).reshape((-1, 1, 2))], (255,255,255))

	#imgProyections2 = imgProyections2.astype("uint8")
	connectivity = 4  
	# Perform the operation
	_, thresh = cv2.threshold(cv2.cvtColor(imgProyections2.copy(),cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY)
	_,_,stats,_ = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_8UC1)
	
	y, h, x, w = stats[1][1], stats[1][3], stats[1][0], stats[1][2]

	# Distancias proyectadas
	imgProyections = imgProyections[(y-50):y+(h+50), (x-50):x+(w+50)]
	# Curvas convexas
	imgProyections1 = imgProyections1[(y-50):y+(h+50), (x-50):x+(w+50)]
 	# Curvas convexas2
	imgProyections1_1 = imgProyections1_1[(y-50):y+(h+50), (x-50):x+(w+50)]
	# Curvas convexas3
	imgProyections1_2 = imgBinHandCurveConvex(img_orginal)
	# Curvas convexas rellenada
	imgProyections2 = imgProyections2[(y-50):y+(h+50), (x-50):x+(w+50)]
  
	cv2.namedWindow('Distancias', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Distancias', h_w, w_w)
	cv2.imshow('Distancias', imgProyections)
 
	cv2.namedWindow('CurvasConvexas', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas', h_w, w_w)
	cv2.imshow('CurvasConvexas', imgProyections1)
 
	cv2.namedWindow('CurvasConvexas2', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas2', h_w, w_w)
	cv2.imshow('CurvasConvexas2', imgProyections1_1)
 
	cv2.namedWindow('CurvasConvexas3', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas3', h_w, w_w)
	cv2.imshow('CurvasConvexas3', imgProyections1_2)
 
	cv2.namedWindow('CurvasConvexasRellenada', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexasRellenada', h_w, w_w)
	cv2.imshow('CurvasConvexasRellenada', imgProyections2)
 
	cv2.waitKey(0)
	cv2.destroyAllWindows()
 
	return imgProyections, imgProyections1, imgProyections1_1, imgProyections1_2, imgProyections2

# Función que determina nuevas mediciones (promedio y sumatoria) a partir de las distancias y angulos previamente calculados
def othersMeasurements(data_results, nPx, verbose):
	# Columnas de los valores correspondientes a distancias
	dists = [0, 8, 16, 24, 32]

 	# Columnas de los valores correspondientes a angulos
	angls = [4, 12, 20, 28, 35]
		
	# Lista de los dedos procesados
	fingers = ['Index', 'Middle', 'Ring', 'Little', 'Thumb']
 	
	# Leyendo dataset de las mediciones obtenidas
	df = pd.DataFrame(data_results, index=[0])#, columns = list(data_results.keys()))
	df2 = df.copy()
	
  	# Dataframe original
	df_original = df2.iloc[:,1:-1]
	#print(df_original)
	
	# Columnas del dataset
	df_columns = df.columns[1:-1]

	#________ Comparativa entre distancias
	dists_aux = [dist+4 for dist in dists]
	dists_aux[-1] -= 1 

	sumFingers, meanFingers = 0, 0
	for col in range(0, len(dists)):
		sumFinger, meanFinger = 0, 0
		data_aux = df_original.iloc[nPx, dists[col]:dists_aux[col]]
		
		sumFinger = round(np.sum(data_aux), 3)
		data_results['sumFinger-'+fingers[col]+'-dists'] = sumFinger
		sumFingers += sumFinger
		
		meanFinger = round(np.mean(data_aux), 3)
		data_results['meanFinger-'+fingers[col]+'-dists'] = meanFinger
		meanFingers += meanFinger
		
		if verbose:
			print(' - ', fingers[col])
			print(list(df_columns[dists[col]:dists_aux[col]]))
			print(data_aux, '\n')
			print('sum:', sumFinger)
			print('mean:', meanFinger, '\n')
	
	sumFingers = round(sumFingers,3)
	meanFingers = round((meanFingers/5), 3) #round((sumFingers/5), 3)
	
	if verbose:
		print('sum-Total: ', sumFingers)
		print('mean-Total: ', meanFingers, '\n')
		print(df.iloc[nPx, 0])
	
	#metrics['idImg'].append(df.iloc[nPx, 0])
	data_results['sumFingers-dists'] = sumFingers
	data_results['meanFingers-dists'] = meanFingers
  
	#df_metrics = pd.DataFrame(metrics)
	#df = pd.concat([df, df_metrics], axis=1)
	#df_metrics.to_csv('dataset_metricas_distancias.csv', index=False)
	#print(data_results)
	
	#___________ Comparativa entre angulos
	angls_aux = [angl+4 for angl in angls]
	angls_aux[-1] -= 1 

	# Dataframe original
	df_original = df2.iloc[:,1:-1]
	#print(df_original)

	sumFingers, meanFingers  = 0, 0
	for col in range(0, len(angls)):
		sumFinger, meanFinger = 0, 0
		data_aux = df_original.iloc[nPx, angls[col]:angls_aux[col]]
		
		sumFinger = round(np.sum(data_aux), 3)
		data_results['sumFinger-'+fingers[col]+'-angls'] = sumFinger
		sumFingers += sumFinger
		
		meanFinger = round(np.mean(data_aux), 3)
		data_results['meanFinger-'+fingers[col]+'-angls'] = meanFinger
		meanFingers += meanFinger
		
		if verbose:
			print(' - ', fingers[col])
			print(list(df_columns[angls[col]:angls_aux[col]]))
			print(data_aux, '\n')
			print('sum:', sumFinger)
			print('mean:', meanFinger, '\n')
	
	sumFingers = round(sumFingers,3)
	meanFingers = round((meanFingers/5), 3)#round((sumFingers/5), 3)
	
	if verbose:
		print('sum-Total: ', sumFingers)
		print('mean-Total: ', meanFingers, '\n')
		print(df.iloc[nPx, 0])
	
	#metrics2['idImg'].append(df.iloc[nPx, 0])
	data_results['sumFingers-angls'] = sumFingers
	data_results['meanFingers-angls'] = meanFingers

	#df_metrics2 = pd.DataFrame(metrics2)
	#df_metrics2.to_csv('dataset_metricas_angulos.csv', index=False)
	#df = pd.concat([df, df_metrics2], axis=1)
	#print(data_results)
	#print(df_metrics2)
 
	return data_results

# Función para calcular y obtener mediciones a partir de los puntos identificados por MediaPipe
def measurementsHand_CMC(file, image, pointsJoints, path_imgs, path_results, nombImg, indManoDom, n, data_results, nombfile, output, label):
	#try:
	print('\n ** Calculando mediciones (distancias y ángulos) teniendo como referencia a la muñeca (CMC).', end = '')
	
	# Creando carpetas donde serán almacenados los resultados obtenidos de la imagen procesada
	if indManoDom == 0:
		path_imgs2 = path_imgs+'/'+nombImg
		path_imgs2_1 = path_imgs+'/'+nombImg+'/Hand_CMC'
		path_imgs3 = path_imgs+'/CNN-CMC'
		path_imgs4 = path_imgs+'/CNN-CMC/'+nombImg

		path_results2_1 = path_results+'/'+nombImg
		path_results2 = path_results+'/'+nombImg+'/Hand_CMC'

	try:
		os.mkdir(path_imgs2)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
	
	try:
		os.mkdir(path_imgs2_1)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
	
	try:
		os.mkdir(path_imgs3)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None

	try:
		os.mkdir(path_imgs4)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
		
	try:
		os.mkdir(path_results2_1)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None

	try:
		os.mkdir(path_results2)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None

	# Alto y Ancho de la imagen
	h, w, _ = image.shape
	
	# Configurando para la impresión de mensajes
	font_size = int((h*0.002)/2) #2
	if font_size > 6:
		font_size = 6

	# Imagen donde serán trazados los puntos de refencia
	img_pnts_rfnc = image.copy()
	
	# ____ Proyección del eje de referencia (Muñeca-MCP_DEDO_MEDIO)
	# Puntos por los que pasará la recta (línea de referencia)
	x1, y1 = pointsJoints[9][0], pointsJoints[9][1] # MCP-DEDO_MEDIO
	x2, y2 = pointsJoints[0][0], pointsJoints[0][1] # MUÑECA

	# Calculando línea perpendicular de una línea
	x3, x4, y3, y4 = lineaPerpendicular(x1, y1, x2, y2, 1, int(w/2))

	# Determinando el punto de intersección entre ambas rectas de referencia
	interseccion = puntoInterseccionEjesRef((x1, y1), (x2, y2), (x3, y3), (x4, y4))

	# Coordenadas de secciones de referencia MANO
	referencia = {
		'eje_y': [(x1, y1), (x2, y2)],
		'eje_x': [(x3, y3), (x4, y4)],
		'origen': [interseccion],
		'munieca': [(x2, y2)]
	}

	# Dibujando línea de referencia (muñeca a MCP - DEDO INDICE)
	img_pnts_rfnc = cv2.line(img_pnts_rfnc, referencia['eje_y'][0], referencia['eje_y'][1], (0,255,0), font_size*3) 

	# - Línea desplazada en el eje X (Colocada los más cerca de la muñeca)
	img_pnts_rfnc = cv2.line(img_pnts_rfnc, referencia['eje_x'][0], referencia['eje_x'][1], (255,255,255), font_size*3)
	
	# Punto intersección - entre ejes de referencia
	img_pnts_rfnc = cv2.circle(img_pnts_rfnc, referencia['origen'][0], font_size*3, (0,0,0), -1)# (x2, y2)
	
	# Punto origen
	img_pnts_rfnc = cv2.circle(img_pnts_rfnc, referencia['munieca'][0], font_size*3, (150,150,150), -1)# (x2, y2)

	try:
		# Almacenando imagen con referencias proyectadas
		cv2.imwrite(path_imgs2_1+'/PuntosReferenciaMano.jpg', img_pnts_rfnc)
	except Exception as err:
		print(' [-- Error durante el almacenamiento de la IMG \"PuntosDeReferenciaMano\" --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None

	
	# ***************** Calculando métricas con MediaPipe
	# Iniciando en el punto 5 deacuerdo a las Hand Landmarks de MediaPipe
	dedo_label = ['Index', 'Middle', 'Ring', 'Little', 'Thumb'] # -> ['indice', 'medio', 'anular', 'menique', 'pulgar']
	hand_results = []

	# Máscara de Proyecciones de angulos
	imgMaskAngles = np.zeros((image.shape), dtype = 'uint8')
	
	
	# Máscara de Proyecciones de distancias
	imgMaskDistances = np.zeros((h_r*2, w_r*2, 3), dtype = 'uint8')
	
	# Indicador de articulación
	joint = 5
	salto = 4 # --- Brincando 4 articulaciones -> MCP, PIP, DIP, TIP

	# Coordenadas en X,Y para la grafica de curvas convexas
	convexPoints = []
	convexPoints2 = []

	# Calculo de las métricas - (Inicia con el dedo indice hacia el meñique)
	convexPoints2.append([pointsJoints[0][0]+int((w_r)/2), pointsJoints[0][1]+int((h_r)/2)])
	for dedo in range(0, len(dedo_label)):
		# Imagen donde serán trazados las measurements (DISTANCIAS) calculadas por MediaPipe en cada dedo
		imgResultsFinger = image.copy()

		# Imagen donde serán trazados las measurements (ANGULOS) calculadas por MediaPipe en cada dedo
		imgResultsFinger2 = image.copy()

		# Nombre de la imagen a procesar
		data_results['nombImg'] = nombImg
		
		if dedo == 4:
			salto = 3
			joint = 2
			# Articulaciones en el pulgar
			# CMC, MCP, DIP, TIP = 0, joint, joint+1, joint+2
			CMC, MCP, PIP, DIP = 0, joint, joint+1, joint+2
			
			# Añadiendo las coordenadas de las articulaciones MCP y TIP
			#convexPoints.append(((pointsJoints[MCP][0], pointsJoints[MCP][1]), 
			#					(pointsJoints[DIP][0], pointsJoints[DIP][1])))

			#convexPoints.append([pointsJoints[MCP][0], pointsJoints[MCP][1]]) 
			convexPoints.append([pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)])

			convexPoints2.append([pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)])
		else:
			# Articulaciones de los dedos
			CMC, MCP, PIP, DIP, TIP = 0, joint, joint+1, joint+2, joint+3

			# Añadiendo las coordenadas de las articulaciones MCP y TIP
			#convexPoints.append(((pointsJoints[MCP][0], pointsJoints[MCP][1]), 
			#					(pointsJoints[TIP][0], pointsJoints[TIP][1])))
			#convexPoints.append([pointsJoints[MCP][0], pointsJoints[MCP][1]]) 
			convexPoints.append([pointsJoints[TIP][0]+int((w_r)/2), pointsJoints[TIP][1]+int((h_r)/2)])
		
			convexPoints2.append([pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[TIP][0]+int((w_r)/2), pointsJoints[TIP][1]+int((h_r)/2)])

		# Cuadro descriptor de métricas obtenidas
		imgResultsFinger = cv2.rectangle(imgResultsFinger, (0, 0), (w, int(h*.12)), (0,0,0), -1)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' * Distancias ', (20, int(h*.03)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		
			
		
		# --- Obteniendo longitudes entre articulaciones -------------------------------------------------------	  
		# === CMC - MCP
		# Extraer las coordenadas x e y de los puntos
		x_coords = [pointsJoints[CMC][0], pointsJoints[MCP][0]]
		y_coords = [pointsJoints[CMC][1], pointsJoints[MCP][1]]

		# Graficar la línea entre los dos puntos
		plt.plot(x_coords, y_coords, marker='', linestyle='-', color='white')
		
		dist0 = distance.euclidean((pointsJoints[CMC][0], pointsJoints[CMC][1]), (pointsJoints[MCP][0], pointsJoints[MCP][1]))
		dist0 = (round(dist0, 3)/h)*100 # --> Normalizando distancia

		# Dibujando la proyeccción (máscara) del distancia entre articulaciones CMC y MCP
		imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[CMC][0]+int((w_r)/2), pointsJoints[CMC][1]+int((h_r)/2)), (pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)), (255,255,255), font_size*3)

		# Ploteando el resultado obtenido en la imagen
		x_text = int((pointsJoints[CMC][0] + pointsJoints[MCP][0])/2)
		y_text = int((pointsJoints[CMC][1] + pointsJoints[MCP][1])/2)
		
		imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[CMC][0], pointsJoints[CMC][1]), (pointsJoints[MCP][0], pointsJoints[MCP][1]), (0,0,0), font_size*3)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D0 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D0 (CMC-MCP): '+str(round(dist0, 3))+' px', (20, int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		data_results['dist_CMC-MCP_'+dedo_label[dedo]] = round(dist0, 3)
		
		

		
		# === MCP - PIP
		# Extraer las coordenadas x e y de los puntos
		x_coords = [pointsJoints[MCP][0], pointsJoints[PIP][0]]
		y_coords = [pointsJoints[MCP][1], pointsJoints[PIP][1]]

		# Graficar la línea entre los dos puntos
		plt.plot(x_coords, y_coords, marker='', linestyle='-', color='white')

		dist1 = distance.euclidean((pointsJoints[MCP][0], pointsJoints[MCP][1]), (pointsJoints[PIP][0], pointsJoints[PIP][1]))
		dist1 = (round(dist1, 3)/h)*100
		
		# Dibujando la proyeccción (máscara) del distancia entre articulaciones MCP y PIP
		imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)), (pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)), (255,255,255), font_size*3)

		# Ploteando el resultado obtenido en la imagen
		x_text = int((pointsJoints[MCP][0] + pointsJoints[PIP][0])/2)
		y_text = int((pointsJoints[MCP][1] + pointsJoints[PIP][1])/2)

		imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[MCP][0], pointsJoints[MCP][1]), (pointsJoints[PIP][0], pointsJoints[PIP][1]), (0,0,0), font_size*3)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D1 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		if dedo == 4:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D1 (MCP-DIP): '+str(round(dist1, 3))+' px', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_MCP-DIP_'+dedo_label[dedo]] = round(dist1, 3)
		else:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D1 (MCP-PIP): '+str(round(dist1, 3))+' px', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_MCP-PIP_'+dedo_label[dedo]] = round(dist1, 3)


		

		# === PIP - DIP
		# Extraer las coordenadas x e y de los puntos
		x_coords = [pointsJoints[PIP][0], pointsJoints[DIP][0]]
		y_coords = [pointsJoints[PIP][1], pointsJoints[DIP][1]]

		# Graficar la línea entre los dos puntos
		plt.plot(x_coords, y_coords, marker='', linestyle='-', color='white')

		dist2 = distance.euclidean((pointsJoints[PIP][0], pointsJoints[PIP][1]), (pointsJoints[DIP][0], pointsJoints[DIP][1]))
		dist2 = (round(dist2, 3)/h)*100

		# Dibujando la proyeccción (máscara) del distancia entre articulaciones PIP y DIP
		imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)), (pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)), (255,255,255), font_size*3)
		
		# Ploteando el resultado obtenido en la imagen
		x_text = int((pointsJoints[PIP][0] + pointsJoints[DIP][0])/2)
		y_text = int((pointsJoints[PIP][1] + pointsJoints[DIP][1])/2)
		
		imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[PIP][0], pointsJoints[PIP][1]), (pointsJoints[DIP][0], pointsJoints[DIP][1]), (0,0,0), font_size*3)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D2 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		
		if dedo == 4:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D2 (DIP-TIP): '+str(round(dist2, 3))+' px', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_DIP-TIP_'+dedo_label[dedo]] = round(dist2, 3)
		else:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D2 (PIP-DIP): '+str(round(dist2, 3))+' px', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_PIP-DIP_'+dedo_label[dedo]] = round(dist2, 3)
			

		if dedo < 4:
			# === DIP - TIP
			# Extraer las coordenadas x e y de los puntos
			x_coords = [pointsJoints[DIP][0], pointsJoints[TIP][0]]
			y_coords = [pointsJoints[DIP][1], pointsJoints[TIP][1]]

			# Graficar la línea entre los dos puntos
			plt.plot(x_coords, y_coords, marker='', linestyle='-', color='white')

			dist3 = distance.euclidean((pointsJoints[DIP][0], pointsJoints[DIP][1]), (pointsJoints[TIP][0], pointsJoints[TIP][1]))
			dist3 = (round(dist3, 3)/h)*100

			# Dibujando la proyeccción (máscara) del distancia entre articulaciones DIP y TIP
			imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)), (pointsJoints[TIP][0]+int((w_r)/2), pointsJoints[TIP][1]+int((h_r)/2)), (255,255,255), font_size*3)

			# Ploteando el resultado obtenido en la imagen
			x_text = int((pointsJoints[DIP][0] + pointsJoints[TIP][0])/2)
			y_text = int((pointsJoints[DIP][1] + pointsJoints[TIP][1])/2)

			imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[DIP][0], pointsJoints[DIP][1]), (pointsJoints[TIP][0], pointsJoints[TIP][1]), (0,0,0), font_size*3)
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D3 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D3 (DIP-TIP): '+str(round(dist3, 3))+' px', (int(w*.5), int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_DIP-TIP_'+dedo_label[dedo]] = round(dist3, 3)
		
		# -------------------------------------------------------------- Obteniendo ángulos entre articulaciones ---
		# Cuadro descriptor de métricas obtenidas
		imgResultsFinger2 = cv2.rectangle(imgResultsFinger2, (0, 0), (w, int(h*.12)), (0,0,0), -1)
		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' * Angulos ', (20, int(h*.03)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)		
		
		# Punto de referencia
		p2 = (x3, y3)
		if pointsJoints[-1] == 'Left':
			p2 = (x4, y4)

		# ==== MCP
		p1 = (pointsJoints[MCP][0], pointsJoints[MCP][1])	
		angulo1 = angleBetweenVectors(referencia['origen'][0], p1, p2)

		# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación 
		imgMaskAngles = cv2.line(imgMaskAngles, referencia['origen'][0], p1, (255,255,255), font_size*3)
		
		# Ploteando el resultado obtenido en la imagen
		x_text = int((p1[0]))#+referencia['origen'][0][0])/2)
		y_text = int((p1[1]))#+referencia['origen'][0][1])/2)

		imgResultsFinger2 = cv2.line(imgResultsFinger2, referencia['origen'][0], p2, (255,255,255), font_size*3)

		# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 1)
		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G1 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
		imgResultsFinger2 = cv2.line(imgResultsFinger2, referencia['origen'][0], p1, colorg1, font_size*3)
		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G1 (MCP): '+str(round(angulo1, 2))+' grados ', (20, int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt1, font_size, cv2.LINE_AA)
		data_results['angl_MCP_'+dedo_label[dedo]] = round(angulo1, 2)



		
		# ==== PIP
		p1 = (pointsJoints[PIP][0], pointsJoints[PIP][1])
		angulo2 = angleBetweenVectors(referencia['origen'][0], p1, p2)

		# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación 
		imgMaskAngles = cv2.line(imgMaskAngles, referencia['origen'][0], p1, (255,255,255), font_size*3)
		
		# Ploteando el resultado obtenido en la imagen
		x_text = int((p1[0]))#+referencia['origen'][0][0])/2)
		y_text = int((p1[1]))#+referencia['origen'][0][1])/2)

		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G2 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
		if dedo == 4:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 3)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, referencia['origen'][0], p1, colorg3, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G2 (DIP): '+str(round(angulo2, 2))+' grados ', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt3, font_size, cv2.LINE_AA)
			data_results['angl_DIP_'+dedo_label[dedo]] = round(angulo2, 2)
		else:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 2)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, referencia['origen'][0], p1, colorg2, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G2 (PIP): '+str(round(angulo2, 2))+' grados ', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt2, font_size, cv2.LINE_AA)
			data_results['angl_PIP_'+dedo_label[dedo]] = round(angulo2, 2)



		# ==== DIP
		p1 = (pointsJoints[DIP][0], pointsJoints[DIP][1])
		angulo3 = angleBetweenVectors(referencia['origen'][0], p1, p2)
		
		# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación
		imgMaskAngles = cv2.line(imgMaskAngles, referencia['origen'][0], p1, (255,255,255), font_size*3)

		# Ploteando el resultado obtenido en la imagen
		x_text = int((p1[0]))#+referencia['origen'][0][0])/2)
		y_text = int((p1[1]))#+referencia['origen'][0][1])/2)

		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G3 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
		if dedo == 4:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 4)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, referencia['origen'][0], p1, colorg4, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G3 (TIP): '+str(round(angulo3, 2))+' grados ', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt4, font_size, cv2.LINE_AA)
			data_results['angl_TIP_'+dedo_label[dedo]] = round(angulo3, 2)
		else:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 3)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, referencia['origen'][0], p1, colorg3, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G3 (DIP): '+str(round(angulo3, 2))+' grados ', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt3, font_size, cv2.LINE_AA)
			data_results['angl_DIP_'+dedo_label[dedo]] = round(angulo3, 2)
			
		if dedo < 4:
			# ==== TIP
			p1 = (pointsJoints[TIP][0], pointsJoints[TIP][1])
			angulo4 = angleBetweenVectors(referencia['origen'][0], p1, p2)
			
			# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación 
			imgMaskAngles = cv2.line(imgMaskAngles, referencia['origen'][0], p1, (255,255,255), font_size*3)
						
			# Ploteando el resultado obtenido en la imagen
			x_text = int((p1[0]))#+referencia['origen'][0][0])/2)
			y_text = int((p1[1]))#+referencia['origen'][0][1])/2)

			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G4 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 4)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, referencia['origen'][0], p1, colorg4, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G4 (TIP): '+str(round(angulo4, 2))+' grados ', (int(w*.5), int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt4, font_size, cv2.LINE_AA)
			data_results['angl_TIP_'+dedo_label[dedo]] = round(angulo4, 2)
		


		# Dedo pulgar
		if dedo == 4:
			# Almacenando measurements calculadas
			dedo_result = {
				'finger': dedo_label[-1],
				'label_len(px)': ['CMC-MCP', 'MCP-DIP', 'DIP-TIP'],
				'length': [dist0, dist1, dist2],
				'label_angl(grades)': ['MCP', 'DIP', 'TIP'],
				'angle': [round(angulo1, 2), round(angulo2, 2), round(angulo3, 2)]
			}
		else:
			# Almacenando measurements calculadas
			dedo_result = {
				'finger': dedo_label[dedo],
				'label_len(px)': ['CMC-MCP', 'MCP-PIP', 'PIP-DIP', 'DIP-TIP'],
				'length': [dist0, dist1, dist2, dist3],
				'label_angl(grades)': ['MCP', 'PIP', 'DIP', 'TIP'],
				'angle': [round(angulo1, 2), round(angulo2, 2), round(angulo3, 2), round(angulo4, 2)]
			}

		# Almacenando los resultados obtenidos del dedo
		hand_results.append(dedo_result)

		try:
			# Resultados de las imgs de cada dedo
			cv2.imwrite(path_imgs2_1+'/resultados_Dists_hand_CMC'+str(n)+'_'+dedo_result['finger']+'.jpg', imgResultsFinger)
			
			cv2.namedWindow('distanciasProyecciones', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('distanciasProyecciones', h_w, w_w)
			cv2.imshow('distanciasProyecciones', imgResultsFinger)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		except Exception as err:
			print('\n [-- Error durante el almacenamiento de la imagen ('+nombImg+') con las DISTANCIAS calculadas --]')
			print(f" - Error: {err}")
			print(f" {type(err).__name__}\n\n")
			return None
		
		try:
			# Resultados de las imgs de cada dedo
			cv2.imwrite(path_imgs2_1+'/resultados_Angls_hand_CMC'+str(n)+'_'+dedo_result['finger']+'.jpg', imgResultsFinger2)
			cv2.namedWindow('distanciasProyecciones', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('distanciasProyecciones', h_w, w_w)
			cv2.imshow('distanciasProyecciones', imgResultsFinger2)
			cv2.waitKey(0)
		except Exception as err:
			print('\n [-- Error durante el almacenamiento de la imagen ('+nombImg+') con los ANGULOS calculados --]')
			print(f" - Error: {err}")
			print(f" {type(err).__name__}\n\n")
			return None


		if dedo == 4:
			data_results['output'] = output
			try:
				# Calculando nuevas mediciones a partir de las Distancias y Angulos determinados
				data_results = othersMeasurements(data_results, 0, False)
	
				with open(r''+nombfile, 'a', newline='') as f:
					writer = csv.DictWriter(f, fieldnames = data_results.keys())
					writer.writerow(data_results)
			except Exception as err:
				print('\n [-- Error durante el almacenamiento de los resultados obtenidos de \"'+nombImg+'\" --]')
				print(' - Resultados de todos los dedos en la imagen (archivo CSV) ')
				print(f" - Error: {err}")
				print(f" {type(err).__name__}\n\n")
				return None

		# Procesar el siguiente de dedo
		joint += salto
	
	#Mask_imgs
	imgMaskDistances, imgCurveConvex, imgCurveConvex2, imgCurveConvex3, imgCurveConvexRefilling  = imgHandCurveConvex(convexPoints, convexPoints2, imgMaskDistances.copy(), image.copy())
	
	# Guardar la gráfica trazada
	nameImgCurveConvex = '/resultados_CurvaConvexa_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgCurveConvex2 = '/resultados_CurvaConvexa2_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgCurveConvex3 = '/resultados_CurvaConvexa3_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgCurveConvexRefilling = '/resultados_CurvaConvexaRellena_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgDistances = '/resultados_Dists_IMG_'+label+'_'+str(n)+'.jpg'

	#if output == 1:
	#	nameImgCurveConvex = '/resultados_CurvaConvexa_IMG_AR'+str(n)+'.jpg'
	#	nameImgCurveConvexRefilling = '/resultados_CurvaConvexaRellena_IMG_AR'+str(n)+'.jpg'
	#	nameImgDistances = '/resultados_Dists_IMG_AR'+str(n)+'.jpg'
		
	'''
	cv2.namedWindow('DistanciasProyectadas', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('DistanciasProyectadas', h_w, w_w)
	cv2.imshow('DistanciasProyectadas', imgMaskDistances)
	
	cv2.namedWindow('CurvasConvexas', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas', h_w, w_w)
	cv2.imshow('CurvasConvexas', imgCurveConvex))

	cv2.namedWindow('CurvasConvexas2', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas2', h_w, w_w)
	cv2.imshow('CurvasConvexas2', imgCurveConvex2))
	
	cv2.namedWindow('CurvasConvexas3', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas3', h_w, w_w)
	cv2.imshow('CurvasConvexas3', imgCurveConvex3))
	
	cv2.namedWindow('CurvasConvexasRellenada', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexasRellenada', h_w, w_w)
	cv2.imshow('CurvasConvexasRellenada', imgCurveConvexRefilling))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	try:
		# Almacenando resultados obtenidos dibujadoos en la imagen procesada  
		imgMaskDistances = cv2.resize(imgMaskDistances, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgDistances, imgMaskDistances)
		
		imgCurveConvex = cv2.resize(imgCurveConvex, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvex, imgCurveConvex)

		imgCurveConvex2 = cv2.resize(imgCurveConvex2, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvex2, imgCurveConvex2)

		imgCurveConvex3 = cv2.resize(imgCurveConvex3, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvex3, imgCurveConvex3)
		
		imgCurveConvexRefilling = cv2.resize(imgCurveConvexRefilling, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvexRefilling, imgCurveConvexRefilling)
	except Exception as err:
		print('\n [-- Error durante el almacenamiento de las imágenes con las proyecciones (DISTANCIAS Y CURVAS CONVEXAS) obtenidas de \"'+nombImg+'\" --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
	
	try:
		# Almacenando resultados obtenidos en un JSON y almacenandolos dentro de la carpeta de resultados (path2)
		for result in hand_results:
			# Almacenar el diccionario en un archivo JSON
			with open(path_results2+'/'+result['finger']+'_hand_CMC_MediaPipe.json', 'w') as json_file:
				json.dump(result, json_file, indent=2)
	except Exception as err:
		print('\n [-- Error durante el almacenamiento de los resultados obtenidos de '+nombImg+' --]')
		print(' - Resultados por cada dedo procesado en la imagen (archivo JSON) ')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
	print(' --- OK ')
	
	print('\n	* Recortando las secciones que corresponden a Articulaciones en la imagen (Yolov5).', end = '')
	cut_outJointSections(file, path_imgs3, nombImg)
	print(' --- OK ')
	#except Exception as err:
	#	print('\n [-- Error durante ejecución. Error al procesar la img ('+grpImg+').')
	#	# Captura la excepción y muestra información sobre el error
	#	print(f" Error: {err}")
	#	print(f" Tipo de excepción: {type(err).__name__}")
	#	print(f" Línea donde ocurrió el error: {err.__traceback__.tb_lineno}")

# Función para calcular y obtener mediciones a partir de los puntos identificados por MediaPipe
def measurementsHand_MCP(file, image, pointsJoints, path_imgs, path_results, nombImg, indManoDom, n, data_results, nombfile2, output, label):
	print('\n ** Calculando mediciones (distancias y ángulos) teniendo como referencia a la muñeca (MCP).', end = '')
 
 	# Creando carpetas donde serán almacenados los resultados obtenidos de la imagen procesada
	if indManoDom == 0:
		path_imgs2 = path_imgs+'/'+nombImg
		path_imgs2_1 = path_imgs+'/'+nombImg+'/Hand_MCP'
		path_imgs3 = path_imgs+'/CNN-MCP'
		path_imgs4 = path_imgs+'/CNN-MCP/'+nombImg
  
		path_results2_1 = path_results+'/'+nombImg
		path_results2 = path_results+'/'+nombImg+'/Hand_MCP'

	try:
		os.mkdir(path_imgs2)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
  	
	try:
		os.mkdir(path_imgs2_1)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
  	
	try:
		os.mkdir(path_imgs3)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
  
	try:
		os.mkdir(path_imgs4)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
		
	try:
		os.mkdir(path_results2_1)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None

	try:
		os.mkdir(path_results2)
	except FileExistsError as err:
		pass
	except Exception as err:
		print('\n [-- Error en ejecución con ('+nombImg+') --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None


	# Alto y Ancho de la imagen
	h, w, _ = image.shape

	# Configurando para la impresión de mensajes
	font_size = int((h*0.002)/2) #2
	if font_size > 6:
		font_size = 6

	# Imagen donde serán trazados los puntos de refencia
	img_pnts_rfnc = image.copy()

	# ____ Proyección del eje de referencia (Muñeca-MCP_DEDO_MEDIO)
	# Puntos por los que pasará la recta (línea de referencia)
	x1, y1 = pointsJoints[9][0], pointsJoints[9][1] # MCP-DEDO_MEDIO
	x2, y2 = pointsJoints[0][0], pointsJoints[0][1] # MUÑECA

	# Calculando línea perpendicular de una línea
	x3, x4, y3, y4 = lineaPerpendicular(x1, y1, x2, y2, 1, int(w/2))
	
	# Determinando el punto de intersección entre ambas rectas de referencia
	interseccion = puntoInterseccionEjesRef((x1, y1), (x2, y2), (x3, y3), (x4, y4))

	# Coordenadas de secciones de referencia MANO
	referencia = {
		'eje_y': [(x1, y1), (x2, y2)],
		'eje_x': [(x3, y3), (x4, y4)],
		'origen': [interseccion],
		'munieca': [(x2, y2)]
	}
 
 	# Dibujando línea de referencia (muñeca a MCP - DEDO INDICE)
	img_pnts_rfnc = cv2.line(img_pnts_rfnc, referencia['eje_y'][0], referencia['eje_y'][1], (0,255,0), font_size*3) 

	# - Línea desplazada en el eje X (Colocada los más cerca de la muñeca)
	img_pnts_rfnc = cv2.line(img_pnts_rfnc, referencia['eje_x'][0], referencia['eje_x'][1], (255,255,255), font_size*3)
	
	# Punto intersección - entre ejes de referencia
	img_pnts_rfnc = cv2.circle(img_pnts_rfnc, referencia['origen'][0], font_size*3, (0,0,0), -1)# (x2, y2)
	
	# Punto origen
	img_pnts_rfnc = cv2.circle(img_pnts_rfnc, referencia['munieca'][0], font_size*3, (150,150,150), -1)# (x2, y2)

	try:
		# Almacenando imagen con referencias proyectadas
		cv2.imwrite(path_imgs2_1+'/PuntosReferenciaMano.jpg', img_pnts_rfnc)
	except Exception as err:
		print(' [-- Error durante el almacenamiento de la IMG \"PuntosDeReferenciaMano\" --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None


	# ***************** Calculando métricas con MediaPipe
	# Iniciando en el punto 5 deacuerdo a las Hand Landmarks de MediaPipe
	dedo_label = ['Index', 'Middle', 'Ring', 'Little', 'Thumb'] # -> ['indice', 'medio', 'anular', 'menique', 'pulgar']
	hand_results = []

	# Máscara de Proyecciones de angulos
	imgMaskAngles = np.zeros((image.shape), dtype = 'uint8')
	
	# Máscara de Proyecciones de distancias
	imgMaskDistances = np.zeros((int(h_r*2), int(w_r*2), 3), dtype = 'uint8')

	# Indicador de articulación
	joint = 5
	salto = 4 # --- Brincando 4 articulaciones -> MCP, PIP, DIP, TIP
 
	# Coordenadas en X,Y para la grafica de curvas convexas
	convexPoints = []
	convexPoints2 = []
	convexPoints2.append([pointsJoints[0][0]+int((w_r)/2), pointsJoints[0][1]+int((h_r)/2)])

	# Calculo de las métricas - (Inicia con el dedo indice hacia el meñique)
	for dedo in range(0, len(dedo_label)):
		# Imagen donde serán trazados las measurements (DISTANCIAS) calculadas por MediaPipe en cada dedo
		imgResultsFinger = image.copy()

		# Imagen donde serán trazados las measurements (ANGULOS) calculadas por MediaPipe en cada dedo
		imgResultsFinger2 = image.copy()
		
		# Nombre de la imagen a procesar
		data_results['nombImg'] = nombImg
		
		if dedo == 4:
			salto = 3
			joint = 2
			# Articulaciones en el pulgar
			# CMC, MCP, DIP, TIP = 0, joint, joint+1, joint+2
			CMC, MCP, PIP, DIP = 0, joint, joint+1, joint+2
   			
	  		# Añadiendo las coordenadas de las articulaciones MCP y TIP
			convexPoints.append([pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)])
   
			convexPoints2.append([pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)])
		else:
			# Articulaciones de los dedos
			CMC, MCP, PIP, DIP, TIP = 0, joint, joint+1, joint+2, joint+3
   
   			# Añadiendo las coordenadas de las articulaciones MCP y TIP
			convexPoints.append([pointsJoints[TIP][0]+int((w_r)/2), pointsJoints[TIP][1]+int((h_r)/2)])
   
			convexPoints2.append([pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)])
			convexPoints2.append([pointsJoints[TIP][0]+int((w_r)/2), pointsJoints[TIP][1]+int((h_r)/2)])

		# --- Obteniendo longitudes entre articulaciones -------------------------------------------------------	
		# Cuadro descriptor de métricas obtenidas
		imgResultsFinger = cv2.rectangle(imgResultsFinger, (0, 0), (w, int(h*.12)), (0,0,0), -1)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' * Distancias ', (20, int(h*.03)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		

		# === CMC - MCP
		dist0 = distance.euclidean((pointsJoints[CMC][0], pointsJoints[CMC][1]), (pointsJoints[MCP][0], pointsJoints[MCP][1]))
		dist0 = (round(dist0, 3)/h)*100 # --> Normalizando distancia

		# Dibujando la proyeccción (máscara) del distancia entre articulaciones CMC y MCP
		imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[CMC][0]+int((w_r)/2), pointsJoints[CMC][1]+int((h_r)/2)), (pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)), (255,255,255), font_size*3)

		# Ploteando el resultado obtenido en la imagen
		x_text = int((pointsJoints[CMC][0] + pointsJoints[MCP][0])/2)
		y_text = int((pointsJoints[CMC][1] + pointsJoints[MCP][1])/2)
		
		imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[CMC][0], pointsJoints[CMC][1]), (pointsJoints[MCP][0], pointsJoints[MCP][1]), (0,0,0), font_size*3)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D0 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D0 (CMC-MCP): '+str(round(dist0, 3))+' px', (20, int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		data_results['dist_CMC-MCP_'+dedo_label[dedo]] = round(dist0, 3)

		
		# === MCP - PIP
		dist1 = distance.euclidean((pointsJoints[MCP][0], pointsJoints[MCP][1]), (pointsJoints[PIP][0], pointsJoints[PIP][1]))
		dist1 = (round(dist1, 3)/h)*100
		
		# Dibujando la proyeccción (máscara) del distancia entre articulaciones MCP y PIP
		imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[MCP][0]+int((w_r)/2), pointsJoints[MCP][1]+int((h_r)/2)), (pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)), (255,255,255), font_size*3)

		# Ploteando el resultado obtenido en la imagen
		x_text = int((pointsJoints[MCP][0] + pointsJoints[PIP][0])/2)
		y_text = int((pointsJoints[MCP][1] + pointsJoints[PIP][1])/2)

		imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[MCP][0], pointsJoints[MCP][1]), (pointsJoints[PIP][0], pointsJoints[PIP][1]), (0,0,0), font_size*3)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D1 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		if dedo == 4:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D1 (MCP-DIP): '+str(round(dist1, 3))+' px', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_MCP-DIP_'+dedo_label[dedo]] = round(dist1, 3)
		else:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D1 (MCP-PIP): '+str(round(dist1, 3))+' px', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_MCP-PIP_'+dedo_label[dedo]] = round(dist1, 3)


		# === PIP - DIP
		dist2 = distance.euclidean((pointsJoints[PIP][0], pointsJoints[PIP][1]), (pointsJoints[DIP][0], pointsJoints[DIP][1]))
		dist2 = (round(dist2, 3)/h)*100

		# Dibujando la proyeccción (máscara) del distancia entre articulaciones PIP y DIP
		imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[PIP][0]+int((w_r)/2), pointsJoints[PIP][1]+int((h_r)/2)), (pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)), (255,255,255), font_size*3)
		
		# Ploteando el resultado obtenido en la imagen
		x_text = int((pointsJoints[PIP][0] + pointsJoints[DIP][0])/2)
		y_text = int((pointsJoints[PIP][1] + pointsJoints[DIP][1])/2)
		
		imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[PIP][0], pointsJoints[PIP][1]), (pointsJoints[DIP][0], pointsJoints[DIP][1]), (0,0,0), font_size*3)
		imgResultsFinger = cv2.putText(imgResultsFinger, ' D2 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
		if dedo == 4:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D2 (DIP-TIP): '+str(round(dist2, 3))+' px', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_DIP-TIP_'+dedo_label[dedo]] = round(dist2, 3)
		else:
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D2 (PIP-DIP): '+str(round(dist2, 3))+' px', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_PIP-DIP_'+dedo_label[dedo]] = round(dist2, 3)
			

		if dedo < 4:
			# === DIP - TIP
			dist3 = distance.euclidean((pointsJoints[DIP][0], pointsJoints[DIP][1]), (pointsJoints[TIP][0], pointsJoints[TIP][1]))
			dist3 = (round(dist3, 3)/h)*100

			# Dibujando la proyeccción (máscara) del distancia entre articulaciones DIP y TIP
			imgMaskDistances = cv2.line(imgMaskDistances, (pointsJoints[DIP][0]+int((w_r)/2), pointsJoints[DIP][1]+int((h_r)/2)), (pointsJoints[TIP][0]+int((w_r)/2), pointsJoints[TIP][1]+int((h_r)/2)), (255,255,255), font_size*3)

			# Ploteando el resultado obtenido en la imagen
			x_text = int((pointsJoints[DIP][0] + pointsJoints[TIP][0])/2)
			y_text = int((pointsJoints[DIP][1] + pointsJoints[TIP][1])/2)

			imgResultsFinger = cv2.line(imgResultsFinger, (pointsJoints[DIP][0], pointsJoints[DIP][1]), (pointsJoints[TIP][0], pointsJoints[TIP][1]), (0,0,0), font_size*3)
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D3 ', (x_text-50, y_text-20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			imgResultsFinger = cv2.putText(imgResultsFinger, ' D3 (DIP-TIP): '+str(round(dist3, 3))+' px', (int(w*.5), int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)
			data_results['dist_DIP-TIP_'+dedo_label[dedo]] = round(dist3, 3)

		# -------------------------------------------------------------- Obteniendo ángulos entre articulaciones ---
		# Cuadro descriptor de métricas obtenidas
		imgResultsFinger2 = cv2.rectangle(imgResultsFinger2, (0, 0), (w, int(h*.12)), (0,0,0), -1)
		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' * Angulos ', (20, int(h*.03)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_size, cv2.LINE_AA)

		# Determinando la línea de referencia que permita calcular un punto de referencia
		x3, x4, y3, y4 = lineaPerpendicular(pointsJoints[CMC][0], pointsJoints[CMC][1],
											pointsJoints[MCP][0], pointsJoints[MCP][1], 0, 35)

		# Pintando línea perpendicular
		#img_linea_referencia = cv2.line(imagen.copy(), (x3, y3), (x4, y4), (0,255,0), font_size*3)
		
		# Mostrar la imagen con las proyecciones de referencia
		#cv2.namedWindow('eje_perpendicular_'+dedo_label[dedo], cv2.WINDOW_NORMAL)
		#cv2.resizeWindow('eje_perpendicular_'+dedo_label[dedo], h_w, w_w)
		#cv2.imshow('eje_perpendicular_'+dedo_label[dedo], img_linea_referencia)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		
		# Determinando el punto de intersección entre ambas rectas de referencia
		interseccion = puntoInterseccionEjesRef((pointsJoints[CMC][0], pointsJoints[CMC][1]),
												(pointsJoints[MCP][0], pointsJoints[MCP][1]), 
												(x3, y3), (x4, y4))

		'''# Pintando línea de referencia
		img_linea_referencia = cv2.circle(imagen.copy(), (x3, y3), 20, (255,0,0), -1) # Azul
		img_linea_referencia = cv2.circle(img_linea_referencia, (x4, y4), 20, (0,0,255), -1) # Rojo
		img_linea_referencia = cv2.line(img_linea_referencia, (x3, y3), (x4, y4), (0,255,0), font_size*3)
		img_linea_referencia = cv2.line(img_linea_referencia, (pointsJoints[CMC][0], pointsJoints[CMC][1]), (pointsJoints[MCP][0], pointsJoints[MCP][1]), (255,255,255), font_size*3)
		
		# Mostrar la imagen con las proyecciones de referencia
		cv2.namedWindow('eje_referencia_'+dedo_label[dedo], cv2.WINDOW_NORMAL)
		cv2.resizeWindow('eje_referencia_'+dedo_label[dedo], h_w, w_w)
		cv2.imshow('eje_referencia_'+dedo_label[dedo], img_linea_referencia)
		cv2.waitKey(0)
		cv2.destroyAllWindows()'''
		
		# Punto de referencia
		p2 = (x3, y3)
		p2_2 = referencia['eje_x'][0]
		if pointsJoints[-1] == 'Left':
			p2 = (x4, y4)
			p2_2 = referencia['eje_x'][1]
		
		puntoReferencia = int((x3+x4)/2), int((y3+y4)/2)

		# ==== MCP
		p1 = (pointsJoints[MCP][0], pointsJoints[MCP][1])
		puntoRef = referencia['origen'][0]

		angulo1 = angleBetweenVectors(puntoRef, p1, p2_2)

		# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación 
		imgMaskAngles = cv2.line(imgMaskAngles, puntoRef, p1, (255,255,255), font_size*3)
		
		
		# Ploteando el resultado obtenido en la imagen
		x_text = int((p1[0]))#+puntoRef[0])/2)
		y_text = int((p1[1]))#+puntoRef[1])/2)

		imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoRef, p2_2, (255,255,255), font_size*3)

		# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 1)
		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G1 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
		imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoRef, p1, colorg1, font_size*3)
		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G1 (MCP): '+str(round(angulo1, 2))+' grados ', (20, int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt1, font_size, cv2.LINE_AA)
		data_results['angl_MCP_'+dedo_label[dedo]] = round(angulo1, 2)


		# ==== PIP
		p1 = (pointsJoints[PIP][0], pointsJoints[PIP][1])
		angulo2 = angleBetweenVectors(puntoReferencia, p1, p2)

		# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación 
		imgMaskAngles = cv2.line(imgMaskAngles, puntoReferencia, p1, (255,255,255), font_size*3)

		
		# Ploteando el resultado obtenido en la imagen
		x_text = int((p1[0]))#+puntoReferencia[0])/2)
		y_text = int((p1[1]))#+puntoReferencia[1])/2)
		
		imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoReferencia, p2, (255,255,255), font_size*3)

		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G2 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
		if dedo == 4:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 3)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoReferencia, p1, colorg3, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G2 (DIP): '+str(round(angulo2, 2))+' grados ', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt3, font_size, cv2.LINE_AA)
			data_results['angl_DIP_'+dedo_label[dedo]] = round(angulo2, 2)
		else:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 2)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoReferencia, p1, colorg2, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G2 (PIP): '+str(round(angulo2, 2))+' grados ', (int(w*.5), int(h*.06)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt2, font_size, cv2.LINE_AA)
			data_results['angl_PIP_'+dedo_label[dedo]] = round(angulo2, 2)


		# ==== DIP
		p1 = (pointsJoints[DIP][0], pointsJoints[DIP][1])
		angulo3 = angleBetweenVectors(puntoReferencia, p1, p2)
		
		# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación
		imgMaskAngles = cv2.line(imgMaskAngles, puntoReferencia, p1, (255,255,255), font_size*3)

		# Ploteando el resultado obtenido en la imagen
		x_text = int((p1[0]))#+puntoReferencia[0])/2)
		y_text = int((p1[1]))#+puntoReferencia[1])/2)

		imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G3 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
		if dedo == 4:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 4)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoReferencia, p1, colorg4, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G3 (TIP): '+str(round(angulo3, 2))+' grados ', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt4, font_size, cv2.LINE_AA)
			data_results['angl_TIP_'+dedo_label[dedo]] = round(angulo3, 2)
		else:
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 3)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoReferencia, p1, colorg3, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G3 (DIP): '+str(round(angulo3, 2))+' grados ', (20, int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt3, font_size, cv2.LINE_AA)
			data_results['angl_DIP_'+dedo_label[dedo]] = round(angulo3, 2)


		if dedo < 4:
			# ==== TIP
			p1 = (pointsJoints[TIP][0], pointsJoints[TIP][1])
			angulo4 = angleBetweenVectors(puntoReferencia, p1, p2)
			
			# Dibujando la proyeccción (máscara) del ángulo entre muñeca y articulación 
			imgMaskAngles = cv2.line(imgMaskAngles, puntoReferencia, p1, (255,255,255), font_size*3)
						
			# Ploteando el resultado obtenido en la imagen
			x_text = int((p1[0]))#+puntoReferencia[0])/2)
			y_text = int((p1[1]))#+puntoReferencia[1])/2)

			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G4 ', (x_text+25, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_size, cv2.LINE_AA)
			# Dibujando la proyeccción de la distancia entre muñeca y articulación (color 4)
			imgResultsFinger2 = cv2.line(imgResultsFinger2, puntoReferencia, p1, colorg4, font_size*3)
			imgResultsFinger2 = cv2.putText(imgResultsFinger2, ' G4 (TIP): '+str(round(angulo4, 2))+' grados ', (int(w*.5), int(h*.09)), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt4, font_size, cv2.LINE_AA)
			data_results['angl_TIP_'+dedo_label[dedo]] = round(angulo4, 2)

	
		# Dedo pulgar
		if dedo == 4:
			# Almacenando measurements calculadas
			dedo_result = {
				'finger': dedo_label[-1],
				'label_len(px)': ['CMC-MCP', 'MCP-DIP', 'DIP-TIP'],
				'length': [dist0, dist1, dist2],
				'label_angl(grades)': ['MCP', 'DIP', 'TIP'],
				'angle': [round(angulo1, 2), round(angulo2, 2), round(angulo3, 2)]
			}
		else:
			# Almacenando measurements calculadas
			dedo_result = {
				'finger': dedo_label[dedo],
				'label_len(px)': ['CMC-MCP', 'MCP-PIP', 'PIP-DIP', 'DIP-TIP'],
				'length': [dist0, dist1, dist2, dist3],
				'label_angl(grades)': ['MCP', 'PIP', 'DIP', 'TIP'],
				'angle': [round(angulo1, 2), round(angulo2, 2), round(angulo3, 2), round(angulo4, 2)]
			}

		# Almacenando los resultados obtenidos del dedo
		hand_results.append(dedo_result)

		try:
			# Resultados de las imgs de cada dedo
			cv2.imwrite(path_imgs2_1+'/resultados_Dists_hand_MCP'+str(n)+'_'+dedo_result['finger']+'.jpg', imgResultsFinger)
		except Exception as err:
			print('\n [-- Error durante el almacenamiento de la imagen ('+nombImg+') con las DISTANCIAS calculadas --]')
			print(f" - Error: {err}")
			print(f" {type(err).__name__}\n\n")
			return None
		
		try:
			# Resultados de las imgs de cada dedo
			cv2.imwrite(path_imgs2_1+'/resultados_Angls_hand_MCP'+str(n)+'_'+dedo_result['finger']+'.jpg', imgResultsFinger2)
		except Exception as err:
			print('\n [-- Error durante el almacenamiento de la imagen ('+nombImg+') con los ANGULOS calculados --]')
			print(f" - Error: {err}")
			print(f" {type(err).__name__}\n\n")
			return None
   

		if dedo == 4:
			data_results['output'] = output
			try:
	   			# Calculando nuevas mediciones a partir de las Distancias y Angulos determinados
				data_results = othersMeasurements(data_results, 0, False)
	
				with open(r''+nombfile2, 'a', newline='') as f:
					writer = csv.DictWriter(f, fieldnames = data_results.keys())
					writer.writerow(data_results)
			except Exception as err:
				print('\n [-- Error durante el almacenamiento de los resultados obtenidos de \"'+nombImg+'\" --]')
				print(' - Resultados de todos los dedos en la imagen (archivo CSV) ')
				print(f" - Error: {err}")
				print(f" {type(err).__name__}\n\n")
				return None
	
		# Procesar el siguiente de dedo
		joint += salto
	

	#cv2.namedWindow('Resultado_1', cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('Resultado_1', h_w, w_w)
	#cv2.imshow('Resultado_1', imgMaskDistances)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	imgMaskDistances, imgCurveConvex, imgCurveConvex2, imgCurveConvex3, imgCurveConvexRefilling  = imgHandCurveConvex(convexPoints, convexPoints2, imgMaskDistances.copy(), image.copy())	
 	# Guardar la gráfica trazada
	nameImgCurveConvex = '/resultados_CurvaConvexa_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgCurveConvex2 = '/resultados_CurvaConvexa2_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgCurveConvex3 = '/resultados_CurvaConvexa3_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgCurveConvexRefilling = '/resultados_CurvaConvexaRellena_IMG_'+label+'_'+str(n)+'.jpg'
	nameImgDistances = '/resultados_Dists_IMG_'+label+'_'+str(n)+'.jpg'

	'''
 	cv2.namedWindow('DistanciasProyectadas', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('DistanciasProyectadas', h_w, w_w)
	cv2.imshow('DistanciasProyectadas', imgMaskDistances)
	
 	cv2.namedWindow('CurvasConvexas', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas', h_w, w_w)
	cv2.imshow('CurvasConvexas', imgCurveConvex))
 
	cv2.namedWindow('CurvasConvexas2', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas2', h_w, w_w)
	cv2.imshow('CurvasConvexas2', imgCurveConvex2))
 
	cv2.namedWindow('CurvasConvexas3', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexas3', h_w, w_w)
	cv2.imshow('CurvasConvexas3', imgCurveConvex3))
	
  	cv2.namedWindow('CurvasConvexasRellenada', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('CurvasConvexasRellenada', h_w, w_w)
	cv2.imshow('CurvasConvexasRellenada', imgCurveConvexRefilling))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
 
	try:
		# Almacenando resultados obtenidos dibujados en la imagen procesada
		imgMaskDistances = cv2.resize(imgMaskDistances, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgDistances, imgMaskDistances)
		
		imgCurveConvex = cv2.resize(imgCurveConvex, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvex, imgCurveConvex)
  
		imgCurveConvex2 = cv2.resize(imgCurveConvex2, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvex2, imgCurveConvex2)
  
		imgCurveConvex3 = cv2.resize(imgCurveConvex3, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvex3, imgCurveConvex3)
		
		imgCurveConvexRefilling = cv2.resize(imgCurveConvexRefilling, (256, 256), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path_imgs4+nameImgCurveConvexRefilling, imgCurveConvexRefilling)
	except Exception as err:
		print('\n [-- Error durante el almacenamiento de las imágenes con las proyecciones (DISTANCIAS Y CURVAS CONVEXAS) obtenidas de \"'+nombImg+'\" --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
	
	try:
		# Almacenando resultados obtenidos en un JSON y almacenandolos dentro de la carpeta de resultados (path2)
		for result in hand_results:
			# Almacenar el diccionario en un archivo JSON
			with open(path_results2+'/'+result['finger']+'_hand_MCP_MediaPipe.json', 'w') as json_file:
				json.dump(result, json_file, indent=2)
	except Exception as err:
		print('\n [-- Error durante el almacenamiento de los resultados obtenidos de '+nombImg+' --]')
		print(' - Resultados por cada dedo procesado en la imagen (archivo JSON) ')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
		return None
	print(' --- OK ')
 
	# Recortando secciones de las articulaciones en la imagen original
	print('\n	* Recortando las secciones que corresponden a Articulaciones en la imagen (Yolov5).', end = '')
	cut_outJointSections(file, path_imgs3, nombImg)
	print(' --- OK ')

# Función que crea los datos que serán usados para el entrenamiento de un modelo clasificador
def newDatasetToModel(path_imgs, path_results, PointReference, grpImgs):
	print('\n\n > Creando carpetas con la inf obtenida ('+PointReference+'). ', end='\r')
	
	# Tiempo de ejecución
	now_time = time.strftime("%m%d%Y_%H%M%S", time.localtime())
	pathDir = 'dirToModel_'+grpImg+'_'+PointReference+'_'+now_time
  
	try:
		print('\n   - Carpeta que contendrá la inf. ha enviar al modelo Clasificador ', end='\r')
		os.mkdir(pathDir)
		print('\n   '+pathDir+' - Carpeta creada...  \n')
	except FileExistsError as err:
		print('\n   '+pathDir+' - Ya ha sido creada la carpeta...  \n')
	except Exception as err:
		print('\n [-- Error en ejecución --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
  
	try:
		print('\n   - Carpeta que contendrá las Imágenes.', end='\r')
		pathImgsModel = pathDir+'/imgs'
		os.mkdir(pathImgsModel)
		print('\n   '+pathImgsModel+' - Carpeta creada...  \n')
	except FileExistsError as err:
		print('\n   '+pathImgsModel+' - Ya ha sido creada la carpeta...  \n')
	except Exception as err:
		print('\n [-- Error en ejecución --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")
  
	try:
		print('\n   - Carpeta que contendrá los datos.', end='\r')
		pathDataModel = pathDir+'/data' 
		os.mkdir(pathDataModel)
		print('\n   '+pathDataModel+' - Carpeta creada...  \n')
	except FileExistsError as err:
		print('\n   '+pathDataModel+' - Ya ha sido creada la carpeta...  \n')
	except Exception as err:
		print('\n [-- Error en ejecución --]')
		print(f" - Error: {err}")
		print(f" {type(err).__name__}\n\n")

	#_____ Extrayendo las mediciones obtenidas de la imagen (datos)
	shutil.copy(path_results+'/'+grpImgs+'_data_results_hand_'+PointReference+'.csv', pathDataModel+'/data.csv')
	#shutil.rmtree(path_results)
 
  	#_____ Extrayendo las imágenes obtenidas
	df = pd.read_csv(pathDataModel+'/data.csv')
	df = df.iloc[:, 0]
	
	i = 0
	for id in df:
		path_root = path_imgs+'/CNN-'+PointReference+'/'+id
		
		with os.scandir(path_root+'/') as ficheros:
			ficheros = [fichero.name for fichero in ficheros if fichero.is_file()]
			#print(ficheros)
		
		# 1) Extrayendo imagen DISTANCIAS PROYECTADAS
		#shutil.copy(path_root+'/resultados_Dists_IMG_'+grpImgs+'_'+str('*')+'.jpg', pathImgsModel+'/img'+str(i)+'_dists.jpg')
		shutil.copy(path_root+'/'+ficheros[-1], pathImgsModel+'/'+id+'_dists.jpg')
  
		# 2) Extrayendo imagen CURVA CONVEZA PROYECTADA (Rellena)
		#shutil.copy(path_root+'/resultados_CurvaConvexaRellena_IMG_'+grpImgs+'_'+str('*')+'.jpg', pathImgsModel+'/img'+str(i)+'_curveconvex.jpg')
		shutil.copy(path_root+'/'+ficheros[2], pathImgsModel+'/'+id+'_curveconvex.jpg')
  		
		# 2.1) Extrayendo imagen CURVA CONVEZA PROYECTADA
		shutil.copy(path_root+'/'+ficheros[0], pathImgsModel+'/'+id+'_curveconvex_mask.jpg')
  
  		# 2.2) Extrayendo imagen CURVA CONVEZA PROYECTADA BINARIA MANO
		shutil.copy(path_root+'/'+ficheros[1], pathImgsModel+'/'+id+'_curveconvex_hand.jpg')

		# 3) Extrayendo RECORTES de las ARTICULACIONES
		for joint in ['DIP','PIP2','PIP3','PIP4','PIP5','MCP1','MCP2','MCP3','MCP4','MCP5','CMC']:
			shutil.copy(path_root+'_resultsYolo/crops/'+joint+'/'+joint+'.jpg', pathImgsModel+'/'+id+'_'+joint+'.jpg')
		i+=1

	

#_____________________________________________________________________ VARIABLES
# Colores para el trazo de inf. dentro de las imágenes
colorg0, txt0 = (255,255,255), (255,255,255)
colorg1, txt1 = (255,0,215), (255,95,230)
colorg2, txt2 = (0,255,0), (85, 255, 85)
colorg3, txt3 = (255,210,100), (255,210,100)
colorg4, txt4 = (0,85,255), (50,195,255)

# Resolución de la ventana que mostrará los resultados brevemente con OpenCV
w_w, h_w = 600, 400

# Temporizador para visualizar las imágenes mostradas por OpenCV
tiempo = 1000

# Bandara para indicar si se desean visualizar los resultos obtenidos del procesamiento de la imagen
visulResults = True

# Ruta donde serán extraídas las imágenes a procesar
path_root = 'imgs'

# Grupos de imágenes a procesar
grpImgs = ['class0', 'class1', 'class2']

# Salida del modelo
output_model = {'class0':0, 'class1':1, 'class2':2}

# Ruta donde serán almacenados las imágenes con los resultados trazados
path1 = 'RESULTS_IMGS'

# Ruta donde serán almacenados los datos de los resultados obtenidos
path2 = 'RESULTS_DATA'

# Numero de paciente
nPX, nImgProcessed = 0, 0

# Medidas para el redimensionamiento de las imágenes
w_r, h_r = 1000, 1024  #1024, 1024 #2448, 2448

# Lista donde serán almacenadas la orientación de las imágenes procesadas
listIndicadorMano = {}

# Lista que almacena el nombre de las imágenes que no puedó identificar MediaPipe
listImgNoIdentificadas = []

#_____________________________________________________________________ EJECUCIÓN PRINCIPAL
# Configurando modelo MediaPipe
print(' - Preparando modelo MediaPipe: ', end = ' ')
#mp_drawing = mp.solutions.drawing_utils
#mp_hands = mp.solutions.hands
handMediaPipe = mp.solutions.hands.Hands(static_image_mode=True, 
									max_num_hands=1, 
									min_detection_confidence=0.25,
									min_tracking_confidence=0.25)
print(' Ok ', end='')

print('\n - Procesando imágenes dentro de la carpeta ['+path_root+'],\n   especificamente las imágenes dentro de la carpeta ', grpImgs, ' \n')

# Tiempo de ejecución
now_time = time.strftime("%m%d%Y_%H%M%S", time.localtime())

try:
	print('\n > Carpeta de almacenamiento de resultados obtenidos en el procesamineto de las imágenes: ', end='\r')
	path1+='_'+now_time
	os.mkdir(path1)
	print('\n   '+path1+' - Carpeta creada...  \n')
except FileExistsError as err:
	print('\n   '+path1+' - Ya ha sido creada la carpeta...  \n')
except Exception as err:
	print('\n [-- Error en ejecución --]')
	print(f" - Error: {err}")
	print(f" {type(err).__name__}\n\n")
	
try:
	print('\n > Carpeta de almacenamiento de resultados obtenidos del procesamiento: ', end='\r')
	path2+='_'+now_time
	os.mkdir(path2)
	print('\n   '+path2+' - Carpeta creada...  \n')
except FileExistsError as err:
	print('\n   '+path2+' - Ya ha sido creada la carpeta...  \n')
except Exception as err:
	print('\n [-- Error en ejecución --]')
	print(f" - Error: {err}")
	print(f" {type(err).__name__}\n\n")

print('\n','===='*10,end='\n')
#'''
for grpImg in grpImgs:
	# Lista para la orientación de las manos procesadas:
	listIndicadorMano[grpImg] = []
	
	# Extracción de imágenes
	conjuntoimgs = imagePathExtraction(path_root+'/'+grpImg)
	nPX = 0

	try:
		if len(conjuntoimgs) > 0:
			# Creando dataset de los resultados obtenidos - Mano 
			data_results, nombfile = dataresults(0, grpImg, path2)

			# Creando dataset de los resultados obtenidos - Mano 
			#data_results2, nombfile2 = dataresults(1, grpImg, path2)

			for file in conjuntoimgs:
				#file = 'imgs\class2\RL_IAAR037.jpg'
				# Procesamiento de la imagen
				nombImg = os.path.splitext(os.path.split(file)[-1])[0]
				print('\n\n  ================================================ Procesando imagen: '+nombImg, '\n\n Path: '+file)
				
				# ------------------------------------------------------ RGB/CELULAR
				# Extracción de hand_mark por mediaPipe
				img, pointsJoints_MdPp = hand_landmarksMediaPipe(file)
				
				if pointsJoints_MdPp is None:
					listImgNoIdentificadas.append(grpImg+' - '+nombImg)
					print('\n - MediaPipe: La mano dentro la imagen no pudó ser indentificada.\n\n')
				else:
					nImgProcessed += 1
					#cv2.namedWindow('imagen_a_procesar', cv2.WINDOW_NORMAL)
					#cv2.resizeWindow('imagen_a_procesar', h_w, w_w)
					#cv2.imshow('imagen_a_procesar', img)
					#cv2.waitKey(tiempo*2)
					#cv2.destroyAllWindows()

					print('\n - MediaPipe: La mano dentro de la imagen ha sido identificada.')
					print('\n Datos obtenidos: \n', pointsJoints_MdPp, '\n')
					listIndicadorMano[grpImg].append(pointsJoints_MdPp[-1])
										
					# 3) ---- Extrayendo measurements por medio de lo detectado por MediaPipe
					measurementsHand_CMC(file, img.copy(), pointsJoints_MdPp.copy(), path1, path2, nombImg, 0, nPX, data_results.copy(), nombfile, output_model[grpImg], grpImg)
					#measurementsHand_MCP(file, img.copy(), pointsJoints_MdPp.copy(), path1, path2, nombImg, 0, nPX, data_results2.copy(), nombfile2, output_model[grpImg], grpImg)
				nPX += 1
				#if nPX == 8:
				break # ***************** Quitar

			if pointsJoints_MdPp is not None:
				newDatasetToModel(path1, path2, 'CMC', grpImg)
				#newDatasetToModel(path1, path2, 'MCP', grpImg)
			print('\n\n  ** Grupo de imágenes ['+grpImg+'] ('+str(nPX)+' imágenes) procesado.')
			print('   Fueron procesadas '+str(nImgProcessed)+' imágenes.')
		#break # ***************** Quitar
	except Exception as err:
		print('\n [-- Error durante ejecución. Error al procesar la img ('+grpImg+').')
		# Captura la excepción y muestra información sobre el error
		print(f" Error: {err}")
		print(f" Tipo de excepción: {type(err).__name__}")
		print(f" Línea donde ocurrió el error: {err.__traceback__.tb_lineno}")
		break

#shutil.rmtree(path1)
#shutil.rmtree(path2)

if len(listImgNoIdentificadas) !=0:
	print('\n --- MediaPipe no logró identificar una mano en: ')
	for nombImg in listImgNoIdentificadas:
		print('  - '+nombImg)

# Liberando memoria de MediaPipe
handMediaPipe.close()
print('\n Orientación de las imágenes procesadas: \n', listIndicadorMano)
print('\n','===='*10,end='\n')
#'''