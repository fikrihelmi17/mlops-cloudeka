import requests
import tensorflow as tf

filename = 'images\\rock.jpg'

def images_preprocessing(filename):
    
    image = tf.io.decode_image(open(filename, 'rb').read(), channels=3)
    image = tf.image.resize(image, [150, 150])
    image = image/255.
    
    image_tensor = tf.expand_dims(image, 0)
    image_tensor = image_tensor.numpy().tolist()
    
    return image_tensor

# Prepare the data that is going to be sent in the POST request
image_tensor = images_preprocessing(filename=filename)
json_data = {
    "instances": image_tensor
}

# Define the endpoint with format: http://localhost:8501/v1/models/MODEL_NAME:predict
endpoint = "http://localhost:8501/v1/models/rps_model:predict"

# Send the request to the Prediction API
response = requests.post(endpoint, json=json_data)

map_labels = {0: "Paper", 1: "Rock", 2: "Scissors"}
prediction = tf.argmax(response.json()['predictions'][0]).numpy()
print(map_labels[prediction])
