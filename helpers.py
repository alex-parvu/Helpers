import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def create_model(model_url, num_classes=10):
  """
  Takse a tensorFlow Hub URL and create a Keras Sequential model with it 

  ARGS:
    model_url (str): A tensoflow hub feature extraction URL
    num_classes (int): Number of output neurons in the output layer
      Should be equal to the number of output classes, defautl 10

  Returns:
    An uncompiled Keras Sequntial model with model_url as features extractor
    Lyer and dense output layer with num_classes output neurons.
  """

  # Download the pretrained model and save it as a keras layer

  feature_extractor_layer = hub.KerasLayer(handle=model_url, trainable=False, name='feature_extraction_layer', input_shape=IMAGE_SHAPE+(3,)) # freeze the already learned patterns

  # Create our model

  model = Sequential([
                      feature_extractor_layer
                      , Dense(num_classes,activation='softmax', name='output_layer')
  ])

  return model
  

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
  
def plot_history(history):
  pd.DataFrame(history.history).plot(figsize=(14,10));

  
def classification_results(y_true, y_pred, figsize=(10,10), dpi=100, normalize = 'true'):
  print(classification_report(y_true=y_true, y_pred=y_pred))
  plt.figure(figsize=figsize, dpi=dpi)
  sns.heatmap(confusion_matrix(y_true=y_true, y_pred=y_pred,  normalize=normalize), annot=True)
  plt.ylabel('True Values')
  plt.xlabel('Predicted Values');
  
  
def rand_img_from_dir(target_dir):
  img_name = np.random.choice( os.listdir(target_dir))

  img = mpimg.imread(target_dir+img_name)

  plt.figure(dpi=150)
  plt.imshow(img)
  plt.axis(False);
