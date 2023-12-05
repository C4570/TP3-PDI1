import cv2
from matplotlib import pyplot as plt
import numpy as np

#Hacemos un iterador para correr todos los videos
for tirada in range(1,5):
# --- Leer y grabar un video ------------------------------------------------
    cap = cv2.VideoCapture(f'tirada_{tirada}.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(f'Video-Output-tirada-{tirada}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    coordenadas_actuales = {}
    coordenadas_anteriores = {}
    imagen_flag = False
    cont = 0
    #Funcion de visualizacion
    def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
        if new_fig:
            plt.figure()
        if color_img:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        if not ticks:
            plt.xticks([]), plt.yticks([])
        if colorbar:
            plt.colorbar()
        if new_fig:        
            plt.show(block=blocking)
    #Funcion de deteccion de numero en los datos
    def dados(frame, imagen):
        """Funcion que recibe el primer frame (una imagen dilatada y binarizada) de cuando los dados estan 'detenidos',
            calcula las componentes conectadas de esa imagen, filtra por area para quedarse solo con los dados, y vuelve a analizar las componentes conectadas
            pero en este caso de los 'dados', obtenemos todos los contornos 'hijos' que tiene cada dado, y nos quedamos con los que tienen area mayor a 20,
            la cantidad de esos contornos sera el numero del dado          """
        img = imagen.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        img3 = np.zeros_like(img)
        for ii in range(1, num_labels):
            obj = (labels == ii).astype(np.uint8)
            area = stats[ii,cv2.CC_STAT_AREA]
            ratio_aspecto = stats[ii,cv2.CC_STAT_HEIGHT] / stats[ii,cv2.CC_STAT_WIDTH]    
            if 7000 > area > 3500:
                img3+= obj
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img3)
        for jj in range(1, num_labels):
            obj = (labels == jj).astype(np.uint8)
            if (num_labels - 1) == 5:
                centroid_x, centroid_y = centroids[jj]
                contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contours_filtrados = [contour for contour in contours if cv2.contourArea(contour) > 20]
                numero_dado = len(contours_filtrados) - 1
                #Reemplazar por frame
                cv2.putText(frame, f'{numero_dado}', (int(centroid_x), int(centroid_y)+ 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 2)
                cv2.putText(img, f'{numero_dado}', (int(centroid_x), int(centroid_y)+ 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 2)

        return frame, img

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # --- Procesamiento ---------------------------------------------
            frame2 = frame.copy()
            #Pasamos a HSV 
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #Filtramos para quedarnos con los colores rojos (Dados son rojos)
            rojo_bajo = np.array([0, 50, 50], dtype=np.uint8)
            rojo_alto = np.array([15, 255, 255], dtype=np.uint8)

            mascara_rojo_baja = cv2.inRange(hsv_frame, rojo_bajo, rojo_alto)

            rojo_bajo = np.array([160, 50, 50], dtype=np.uint8)
            rojo_alto = np.array([180, 255, 255], dtype=np.uint8)

            mascara_rojo_alta = cv2.inRange(hsv_frame, rojo_bajo, rojo_alto)

            mascara_rojo = cv2.bitwise_or(mascara_rojo_baja, mascara_rojo_alta)

            resultado = cv2.bitwise_and(frame, frame, mask=mascara_rojo)
            resultado = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)

            #Binarizamos y dilatamos
            th, binary = cv2.threshold(resultado, 20, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5,5), np.uint8)
            dilatado = cv2.dilate(binary, kernel, iterations=1)
            #Buscamos los contornos de los frames
            contornos, jerarquia = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            """Nuestro criterio de 'detencion' sera: encontrar los contornos, y sus coordenadas. Quedarnos con aquellos que sean 'cuadrados',
                pues los dados tienen ese aspecto. Y filtrando aquellos mayores a cierto umbral, para eliminar ruido. La idea es analizar las coordenadas
                de los contornos, si un contorno por 5 frames (nuestro criterio) tiene las misma coordenadas x,y consideramos que el dado esta quieto. """
            frame_contornos = frame2.copy()
            coordenadas_actuales = {}
            for idx, cnt in enumerate(contornos):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                area = w * h
                if np.isclose(aspect_ratio, 1, atol=0.1) and area > 200:
                    dado_actual = f'dado{idx + 1}'
                    coordenadas_actuales[dado_actual] = [x, y, cnt]
            
            if coordenadas_anteriores and coordenadas_actuales:
                for dado_act, coords_act in coordenadas_actuales.items():
                    x_act, y_act, cnt_act = coords_act

                    # Busca el dado correspondiente en coordenadas_anteriores basándose en la distancia euclidiana
                    dado_ant, coords_ant = min(coordenadas_anteriores.items(), key=lambda item: np.linalg.norm(np.array(item[1][:2]) - np.array([x_act, y_act])))
                    x_ant, y_ant, cnt_ant = coords_ant

                    resta_x = abs(x_act - x_ant)
                    resta_y = abs(y_act - y_ant)
                    margen = 3 #pixels
                    cantidad_contornos = len(coordenadas_actuales)
                    "A la condicion de 'dados quietos' le agregamos que los contornos detectados tienen que ser 5, pues hay 5 dados."
                    if resta_x <= margen and resta_y <= margen and cantidad_contornos == 5: #COndicion de dados quietos
                        cont +=1
                    else:
                        cont = 0
                    if cont >= 5:
                        """Una vez que los dados esten quietos, tomamos la primer captura disponible y le ejecutamos nuestra funcion de deteccion
                        de numero. Esa funcion devuelva la imagen clasificada con los numeros, y el 'nuevo' frame con los numeros de los dados."""
                        if not(imagen_flag):
                            imagen = binary
                            imagen_flag = True
                        frame_contornos, imagen_numeros = dados(frame_contornos, imagen)
                        cv2.imshow('Imagen', cv2.resize(imagen_numeros, dsize=(int(width/3), int(height/3) )))
                        # cv2.imwrite(f'imagen_tirada_{tirada}.png', imagen_numeros)
                        cv2.drawContours(frame_contornos, [cnt_ant], -1, (255, 0, 0), 2)                
        
                    cv2.putText(frame_contornos, f'Contornos: {cantidad_contornos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            coordenadas_anteriores = coordenadas_actuales.copy()
            

            # # # --- Muestro por pantalla ------------

            frame_show = cv2.resize(frame_contornos, dsize=(int(width/3), int(height/3)))
            cv2.imshow('Frame', frame_show)
            
            # ---------------------------------------------------------------
            out.write(frame_contornos)  # grabo frame --> IMPORTANTE: frame debe tener el mismo tamaño que se definio al crear out.
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


    
