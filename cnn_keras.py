from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

class CNN:
    def __init__(self,path):
        self.path = path
        self.cnn_classifier = self.build_classifier()
        self.training_set = None
        self.test_set = None


    def build_classifier(self, adding_dropout=True):
        cnn_classifier = Sequential()
        cnn_classifier.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        cnn_classifier.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_classifier.add(Conv2D(32, (3, 3), activation='relu'))
        cnn_classifier.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_classifier.add(Flatten())
        cnn_classifier.add(Dense(units=128, activation='relu'))
        cnn_classifier.add(Dense(units=1, activation='sigmoid'))
        cnn_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return cnn_classifier

    def tensorflow_memory_usage(self):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)


    def fitting_the_cnn_to_images(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.training_set = train_datagen.flow_from_directory('{}training_set'.format(self.path),target_size=(64, 64),batch_size=32,class_mode='binary')

        self.test_set = test_datagen.flow_from_directory('{}test_set'.format(self.path),target_size=(64, 64),batch_size=32,class_mode='binary')

    def train_the_cnn(self):
        self.cnn_classifier.fit_generator(self.training_set,steps_per_epoch=8000,epochs=25,validation_data=self.test_set,validation_steps=2000)

    def save_the_cnn_model(self):
        classifier_json = self.classifier.to_json()
        with open('classifier.json', "w") as json_file:
            json_file.write(classifier_json)
            json_file.close()
        self.cnn_classifier.save_weights("classifier.h5")
        print("saved model to disc")

    def load_test_image(self,path):
        test_image = image.load_img(path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        return np.expand_dims(test_image, axis=0)

    def load_the_cnn_model(self):
        json_file = open('classifier.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.classifier = model_from_json(loaded_model_json)
        # load weights into new model
        self.classifier.load_weights("classifier.h5")
        print("Loaded model from disk")

    def make_a_prediction(self):
        result = self.classifier.predict(self.load_test_image('data/frames/single_prediction/what_2.jpg'))
        print(self.training_set.class_indices)
        if result[0][0] == 0:
            prediction = 'sleep'
        else:
            prediction = 'wake'
        print(prediction)


if __name__ == "__main__":
    cnn = CNN("data/frames/")
    cnn.fitting_the_cnn_to_images()
    cnn.tensorflow_memory_usage()
    cnn.train_the_cnn()
    cnn.save_the_cnn_model()
    cnn.make_a_prediction()


