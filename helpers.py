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
    
    y_true = np.array(y_true).reshape(-1,)
    y_pred = np.array(y_pred).reshape(-1,)

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


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y, random_state=None, figsize=(12,10), dpi=100):
    """
    A function used to plot the train and validation loss values for an sklearn model. The function assumes that the model is a regression.
    
    Parameters:
    
    model        : an sklearn model 
    X            : the independent variables that can be used to train the provided sklearn model 
    y            : the dependent variable/variables that can be used to train the model
    random_state : the random state to be used in the calculations
    figsize      : the size of the figure that is going to get ploted
    dpi          : the dots per inch value that is going to be used by the ploting function
    
    
    Ouput:
        A plot that is going to represent the learning rate for the train and the validation sets.
    """
    
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    train_error, val_error = [],[]
    
    for m in range(1, len(X_train)):
        
        model.fit(X_train[:m], y_train[:m])
        
        y_train_preds = model.predict(X_train[:m])
        y_val_preds = model.predict(X_val)
        
        train_error.append( mean_squared_error(y_train[:m], y_train_preds) )
        val_error.append( mean_squared_error(y_val_preds, y_val) )
        
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot( np.sqrt(train_error), 'r--', linewidth=1, label='train' )
    plt.plot( np.sqrt(val_error), 'b-', linewidth=2, label='val' )
    plt.legend();


def background_classification(estimator, X, y, figsize=(10,10), dpi=100, title=None, step = 0.01, s = 50):
    """
    This is a function that plots the predict space of an sklearn classification estimator behind the actual points and their classification.
    Parameters:
    estimator   : a fitted sklearn classification estimator
    X           : a numpy array containing the two independent variables that were used to fit the system or the two independent variables of the test set
                : the array is assumed to be of the shape (n,2) where n can take any value
    y           : the labels used for the training of the classification estimator or the labels of the test set.
                : the array it is assumed to be of shape (n,) where n can take any value
    NOTE        : the X and y parameters n value must be the same for both arrays
    figsie      : the size of the plot that is to be displayed, default (10,10)
    dpi         : the dpi value to be used on the displayed plot, default 100
    title       : the title that is to be given to the plot
    step        : a parameter that determins how tight the x_axis and y_axis background points are going to be grouped
    s           : a parameter that determins how large the x_axis and y_axis background points are going to be
    """

    from concurrent.futures import ThreadPoolExecutor

    x_min, x_max = X[:,0].min()-0.2, X[:,0].max()+0.2
    y_min, y_max = X[:,1].min()-0.2, X[:,1].max()+0.2

    x_axis = np.arange(x_min, x_max, step)
    y_axis = np.arange(y_min, y_max, step)

    predict_space = []
    for x in x_axis:
        with ThreadPoolExecutor() as executor:
            [ executor.submit(predict_space.append, [x,y, estimator.predict([[x,y]])[0] ]   ) for y in  y_axis ]


    predict_space = np.array(predict_space)        
    plt.figure(figsize=figsize, dpi=dpi)
    sns.scatterplot(x=predict_space[:,0], y=predict_space[:,1], hue=predict_space[:,2], alpha=0.1, legend=False, s=s)
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)
    plt.title(title);
    
  
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

def opt_cluster(estimator, data, cluster_range = np.arange(2,15), figsize=(14,5), dpi=100):
    """
    
    This is a function that allowes you to detect what would be the optimal number of clusters for a given datasets
    
    Parameters:
    
    estimator     : an sklearn cluster estimator from sklearn.cluster, the function assumes that the estimator has the n_clusters parameter and
                    has the .inertia_ and .labels_
    data          : the data that is to be fited to the provided estimator
    cluster_range : a range of ints that are to be used to generate clusters to investigate to optimal number of clusters
    figsize       : the size of the plots that are to be displayed
    dpi           : the dpi value to be used in the plots
    
    Ouput:
    
    This function shall plot two graphs one is the K clusters vs the inertia data and one is the K clusters vs Silhouette Data \n
    
    """

    inertia_data = []
    for k in cluster_range:
        cluster_generator = estimator(n_clusters=k).fit(X)
        inertia_data.append( (k, cluster_generator.inertia_) )

    inertia_data = np.array(inertia_data)

    sil_score = []
    for k in cluster_range:
        cluster_generator = estimator(n_clusters=k).fit(X)
        sil_score.append( (k, silhouette_score(X=X, labels=cluster_generator.labels_)) )

    sil_score = np.array(sil_score)

    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)
    
    axes[0].set_title('Inertia Data')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    axes[0].plot(inertia_data[:,0], inertia_data[:,1])

    axes[1].set_title('Silhouette Data')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].plot(sil_score[:,0], sil_score[:,1])
    
    plt.tight_layout();

def img_segmentation(image, n_color_chanels, figsize=(12,10), dpi=100, plot_image = True, return_new_image = False):
    """
    This is a function that takes an image and segments it by n colors
    
    Parameters:
    image            : an image file of shape (m,n,3)
    n_color_chanels  : the number of color chanels to be used when plottig the image
    figize           : the size of the image to be ploted
    dpi              : the dpi to be used in the ploted image
    plot_image       : default value is True, change it to False if you don't want to plot the image
    return_new_image : default value is False, change it to True if you want the function to return the image
    
    
    Output:
    
    The function shall reduce the numbers of colors in the image o the number of chanels provdide. These shall be the cluster centers from a 
    KMeans clustering algorithm. Once the clusters are computed each pixel shall be replaced with their cluster's center values and the image
    shall be ploted
    """
    from sklearn.cluster import KMeans
    
    shape = image.shape
    X = image.reshape(-1,3)
    
    kmeans = KMeans(n_clusters=n_color_chanels).fit(X)
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image= segmented_image.reshape(shape)
    if plot_image:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(segmented_image/255)
        plt.axis(False);

    if return_new_image:
        return segmented_image/255


def show_lemmas(doc):
    """
    A function that prints out on each line the text of the token the pos_ the lemma hash and the lemma_
    
    Prameters:
    doc : a spacey document that is generaly created by passing some text to the a nlp object created as follows nlp = spacy.load('en_core_web_sm')
    
    Output:
    for each token prints out the following
    {token.text:{12}} {token.pos_:{8}} {token.lemma:<{22}} {token.lemma_}
    """
    for token in doc:
        print(f'{token.text:{12}} {token.pos_:{8}} {token.lemma:<{22}} {token.lemma_}')

def balance_train_set(X, y, sample_size=100, resampling=10):
    """
    This is a function with the purpose of balancing out a train set that has an unbalanced label.
    The method by which it achieves that is by sampling an equal amount of observations for each label (with replacement), for a given amount of times(resampling)
    
    Parameters:
    X           : The X values (independent values), of a training set as a pandas Data Frame or a numpy array, it can have as many features as posible
    y           : The y values (dependent values), of a training set in the form of a single colum pandas Data Frame or a pandas series or a numpy array
    sample_size : An integer (default value = 100, but the value should be smaller than the maximum number of observations of the least represented class), signifies the number
                of values that each sample shall contain from each class.
    resampleing : An integer (can not be 0), representing the number of samples that shall be taken for each individual class, can be as high as needed.
    
    Output:
    This function outputs an X and y where the classes are equily balanced, with the number of observations beeing equaled to sample_size * resampleing * [number of classes]
    """
    X = pd.DataFrame(X)
    if type(y) == pd.core.frame.DataFrame:
        y = pd.Series(y.values.reshape(-1))
    elif type(y) == np.ndarray:
        y = pd.Series(y.reshape(-1))
    else:
        y = pd.Series(y)
    
    unique_values = y.unique()

    Xs = []
    ys = []

    for sample in range(resampling):
        for value in unique_values:
            Xs.append(X[y==value].sample(sample_size))
            ys = ys + [value]*sample_size

    Xs = pd.concat(Xs)
    Xs['ys'] = ys
    Xs = Xs.sample(frac=1)
    
    return Xs.drop('ys',axis=1), Xs.ys
