from ffpyplayer.player import MediaPlayer
import cv2
import numpy as np
import cv2 as cv
from keras.models import load_model
import tensorflow as tf

model = load_model('model.h5')
x1, x2 = 100, 600
y1, y2 = 120, 400

predic = ""



def reproducir_video (predic):
    cap = cv2.VideoCapture("Videos/{0}.mp4".format(predic))
    player = MediaPlayer("Videos/{0}.mp4".format(predic))

    while(True):

        ret, frame = cap.read()
        audi_frame, val = player.get_frame()
       
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == 32:
            print("space")
            player.stop()
            # cap.release()
            break
        elif cv2.waitKey(1) == 13:
            print("enter")
            player.stop()
            scanner()


def hacer_prediccion(img):

    etiquetas = ['Bulbasaur', 'Charmander', 'Squirtle']
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255. #normalizando
    img = cv2.resize(img,(150,150))
    img = np.reshape(img,[1,150,150,3]) #Volviendolos tensores
    result = model.predict(img,use_multiprocessing = True)
    pokemones = etiquetas[(np.argmax(result))]
    #pres redondea valores de porcentajes a 1
    pres = round(((result[0][np.argmax(result)])*100),2)

    mensaje ="{0}".format(pokemones)

    return mensaje
    
def scanner():
    cap = cv2.VideoCapture(0) #tomar el video de la camara
    while(True):

        ret, frame = cap.read()
        imgAux = frame.copy()
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0, 0), 2)
        imgAux = imgAux[y1:y2, x1:x2]
        predic = hacer_prediccion(imgAux)
        print(predic)
        cv2.putText(frame, predic, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255),3)  

        cv2.imshow('video', frame)
        if cv2.waitKey(1) == 27:
            cap.release()
            reproducir_video(predic)
            break

    
scanner()




