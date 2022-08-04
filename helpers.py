import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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



def plot_time_series(timestamp
                     , values
                     , start=0
                     , end=None
                     , label=None
                     , figsize = (12,10)
                     , dpi = 100
                     , use_figure = True
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
    use_figure: if set to False it allows the plot to be overlayed over a previou plot, default is true thus 
                each calling of this function shall generate another separate plot
    color     : color of the line plot to be used.
  """
   # Plot Series
  if end is None:
    end = len(values)

  timestamp = timestamp[start:end]
  values = values[start:end]
  
  if use_figure:
    plt.figure(figsize=figsize, dpi=dpi)
  
  sns.lineplot(x=timestamp, y =values, color=color, label=label)

def regression_results(y_true, y_pred, figsize=(12,10), dpi=100, alpha=0.5, scaler=None, return_values=True):
    """
    This is a function that provides a detailed result between the predicted values and the actual value following the predictions of a regression algorithm
    The function prints out the Root Mean Scaled Error and the Scaled Root Mean Squared Error of of the provided values, as well as plotting the values over an ideal
    case of y_true always being equal to y_pred.
    The function also returns the rmse and srmse in order to be storeed if required.
    
    Paramaters:
    y_true        : the true values from the test set
    y_pred        : the predicted values according to the model
    figsize       : the size of the figure to be plotet
    dpi           : the detail level of the plot to be used
    alpha         : the alpha scale to be used on the dots of the plot
    scaler        : an sklearn scaler model that has been fitted to a set of data having a shape of (n,1)
    return_values : default value as True, if set to False no values are returned, they are only printed out.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plt.figure(figsize=figsize, dpi=dpi) # Set up figure size and detail
    sns.lineplot(x=y_true, y=y_true, color='green') # Set up ideal case where y_true is always equal to y_pred
    sns.scatterplot(x=y_true, y=y_pred, color='blue', alpha = alpha) # Show the actual case in comparison to the ideal one
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values');

    rmse = np.sqrt( mean_squared_error(y_true=y_true, y_pred=y_pred))

    print(f'Root Mean Squared Error: {rmse}')
    
    # The predicted and actual values might be in a one dimensional shape. The scaler expects a two dimensional object
    # Thus we shall reshape the true and predicted values to be two dimensional. 
    
    true_shape = y_true.shape[0] 
    pred_shape = y_pred.shape[0]

    y_true = y_true.reshape(true_shape,-1)
    y_pred = y_pred.reshape(pred_shape,-1)
    
    
    if scaler is None: # If no scaler is provided then a standard MinMax Scaler shall be used and fitted to the true values
        scaler = MinMaxScaler()

        y_true = scaler.fit_transform(y_true)
        y_pred = scaler.transform(y_pred)
    else: # If a scaler is provided then the true and predicted values shall be scalled according to provided scaler.
        y_true = scaler.transform(y_true)
        y_pred = scaler.transform(y_pred)
        
    srmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

    print(f'Scaled Root Mean Squared Error: {srmse}')
    if return_values: # Only returns the metric values if return_values is stated as True
        return rmse, srmse

def plot_precission_vs_recall(precision,recall,thresholds):
    """
    A function that plots the precision vs recall taking into account the thresolds used
    The expected inputs are the outpus from sklearn.metrics.precision_recall_curve
    Outputs from other resources can be passed but the function expects the precission and recall arrays to have a dimension of n+1 when compared to ne n dimensional thresolds array
    
    Parameters:
    precision  : an array of precission values for each subsecquent threashold
    recall     : an array recall values given each subsequent threshold
    thresholds : an arrat of thresholds
    
    """
    plt.figure(figsize=(12,8), dpi = 100)
    plt.plot(thresholds, precision[:-1], '--', label='Precision', c='blue')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    equilibrian = thresholds[ precision[:-1] ==  recall[:-1]]
    plt.axvline(equilibrian, c='black')
    plt.legend();

    print(f'Precission and Recall are equal at a threshold of {equilibrian}')

