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
import seaborn as sns
import numpy as np


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
  
  
def percentile_hist(data, percentile, figsize=(12,10), dpi=75, bins=25):
  """
  A function to calculate and plot the value underneeth the provided percentile

  Args:
    data      : the data to be ploted and used for calculations
    percentile: the value of the percentile to be used for the calculations
    figisize  : the size to be used in the plot, default is (12,10)
    dpi       : the dpi value to be used in the plot
    bins      : the number of bins to be used in the plot
  """

  output_seq_len = int(np.percentile(data, percentile))
  print(f"The value that contains the {percentile} percentile value is {output_seq_len}\n")
  import matplotlib.pyplot as plt
  plt.figure(figsize=figsize, dpi=dpi)
  plt.hist(data, bins=bins)
  plt.axvline(output_seq_len,c='red');

 

def plot_time_series(timestamp
                     , values
                     , start=0
                     , end=None
                     , label=None
                     , figsize = (12,10)
                     , dpi = 100
                     , color='blue'):
  
  """
  Plots a series of points in time (a series of points in time) agains values (a series of values across time)
  Paramaters:
    timesteps : an array of timestamps
    values    : an array of values
    start     : at which point in time to start
    end       : at which point in time to stop
    label     : the name of the plot
    figsize   : size of the plot
    dpi       : the dots per inch of the plot
  """
   # Plot Series
  if end is None:
    end = len(values)

  timestamp = timestamp[start:end]
  values = values[start:end]

  plt.figure(figsize=figsize, dpi=dpi)
  sns.lineplot(x=timestamp, y =values, color=color, label=label);



def read_bad_json(filename, bad_expresions = [ 'ObjectId(', 'ISODate(', ')']):
  """
  A function for reading baddly formated json files  into a pandas Data Frame
  Arguments:
    filename        : the name of the file or the path to the file that is to be read, need not be a json but is should have a json structure inside.
    bad_expresions  : the expresions that need to be replaced with a '' in order for the file to be converted into a DataFrame
                      by default these expresions are  ['ObjectId(', 'ISODate(', ')'] (expresions found in MongoDb jsons) but any can be added.
  """
  with open(filename, mode='r', buffering=8092) as f:
    data_list = f.read()

  for bad_expresion in bad_expresions:
    data_list = data_list.replace(bad_expresion, '')

  data_list = pd.read_json(data_list)
  data_list = data_list.iloc[:,0].to_list()
  return pd.DataFrame(data_list)
