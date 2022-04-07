from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def load_image(filename):
    img_path = "assets/"
    img = load_img(img_path+filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

def run_example():
    img1 = load_image('sample_image1.jpeg')
    img2 = load_image('sample_image2.jpeg')
    model_path = "results/"
    model = load_model(model_path + 'final_model.h5')
    result1 = model.predict(img1)
    result2 = model.predict(img2)
    print("dog" if result1[0] == 1 else "cat")
    print("dog" if result2[0] == 1 else "cat")


if __name__ == "__main__":
    run_example()
