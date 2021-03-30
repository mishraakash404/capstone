class model1:
    def webmodel(self,path):
        print("path received in model.py",path)
        from keras.models import load_model
        import cv2

        import numpy as np

        import keras.backend.tensorflow_backend as tb

        tb._SYMBOLIC_SCOPE.value = True

        classes = {0:'freshapples',1:'freshbanana',2:'freshoranges',3:'rottenapples',4:'rottenbanana',5:'rottenoranges'}

        print("classes built")
        model = load_model('Models/model_cnn.h5')

        print('model fetched successfully')

        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

        img = cv2.imread(str(path))
        img=cv2.resize(img,(45,45))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = np.reshape(img,[1,45,45,3])

        class1= model.predict_classes(img)
        del model
        return(str(classes[class1[0]]))