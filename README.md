# Neural Network Charity Analysis
Alphabet Soup is a non-profit foundation that provides funding to organizations dedicated to protecting the environment, improving people’s well-being and unifying the world. To help them predict which organizations to provide funding to I created a deep learning model using TensorFlow. Then I optimized the model for accuracy.

<details><summary> <h2>Deliverable 1: Preprocessing Data for a Neural Network Model</h2></summary>

1. I imported dependencies  and read the data into a DataFrame.
```
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import tensorflow as tf

# Import checkpoint dependencies
import os
from tensorflow.keras.callbacks import ModelCheckpoint

#Random forest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#  Import and read the charity_data.csv.
import pandas as pd 
application_df = pd.read_csv("Resources/charity_data.csv")
application_df.head()
```
2. I used the .drop function to remove the name and EIN columns from the Dataframe.
```
application_df = application_df.drop(["EIN", "NAME"], 1)
```
![Dropped Name & EIN DF](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/dropped_ein_name.png)
  
3. I found the columns with more than ten unique object values using the  .nunique function, so that these columns could be binned.
```
application_df.nunique()
```
 ![Uique opject value counts](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/nunique.png)
  
4. I set the value counts for the application type and classification columns to variables.
```
app_type_counts = application_df.APPLICATION_TYPE.value_counts()
classification_counts = application_df.CLASSIFICATION.value_counts()
```
5. I determined which values to replace by the value count results and used a for loop to replace them in the DataFrame.
```
# Application Type
replace_application = list(app_type_counts[app_type_counts < 500].index)
for app in replace_application:
    application_df.APPLICATION_TYPE = application_df.APPLICATION_TYPE.replace(app,"Other")
application_df.APPLICATION_TYPE.value_counts()

# Classification
replace_class = list(classification_counts[classification_counts < 777].index)
for cls in replace_class:
    application_df.CLASSIFICATION = application_df.CLASSIFICATION.replace(cls,"Other")    
application_df.CLASSIFICATION.value_counts()

```
![Binning Application Types](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/app_type_counts.png)
![Binning Classification](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/classification_counts.png)
  
6. I grabbed all the columns that had objects in them. Then I created the one hot encoder to fit/transform the object variable columns to numbers and added the new column names to a DataFrame.
```
application_cat = application_df.dtypes[application_df.dtypes == "object"].index.tolist()
enc = OneHotEncoder(sparse=False)
encode_df = pd.DataFrame(enc.fit_transform(application_df[application_cat]))
encode_df.columns = enc.get_feature_names(application_cat)
```
  ![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/application_cat.png)
  ![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/encode_df.png)
7. I merged the encode_df with the application_df and removed the application_cat columns.
```
application_df = application_df.merge(encode_df,left_index=True, right_index=True)
application_df = application_df.drop(application_cat,1)
```
 ![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/merged_application_df.png)
8. I split the data into features, target arrays, training and testing data sets.
```
y = application_df["IS_SUCCESSFUL"].values
X = application_df.drop(["IS_SUCCESSFUL"],1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```
9. I created  and fit a StandardScaler instances. Then scalled the training and test data.
```
# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```
</details>
<details><summary><h2>Deliverable 2: Compile, Train, and Evaluate the Model</h2></summary>

1. First I defined the model’s input features, hidden nodes, layers and activation functions.
```
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 3

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

nn.summary()
```
![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/NN_Summary.png)
  
2. I created the callback, compiled the model and trained the model.
```
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

os.makedirs("checkpoints/",exist_ok=True)
checkpoint_path = "checkpoints/weights.{epoch:02d}.hdf5"
cb = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=1000)

fit_model = nn.fit(X_train_scaled,y_train,epochs=20,callbacks=[cb])
```
![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/Model1.png)
  
3. I summarized the model evaluation.
```
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/Model1_Summary.png)
  
4.  I exported the model.
```
nn.save("AlphabetSoupCharity.h5")
```
5. To compare the neural network to another machine learning algorithm I used random forest.
```
rfc_model = RandomForestClassifier(n_estimators=128, random_state=42) 
rfc_model = rfc_model.fit(X_train_scaled, y_train)
y_pred = rfc_model.predict(X_test_scaled)
print(f" Random forest model accuracy: {accuracy_score(y_test,y_pred):.3f}")
```
 accuracy: 0.716
</details>

<details><summary><h2>Deliverable 3: Optimize the Model</h2></summary>

<h3>How many neurons, layers, and activation functions did you select for your neural network model, and why?</h3>

I experimented with different numbers of neurons and layers. I also experiment with the different activation functions and their order in the model to determine what would give me the most accuracy and least loss in my model.

![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/nn_summary2.png)

<h3>What steps did you take to try and increase model performance?</h3>
<details><summary>To optimize the model first I tried adding additional layers with different number of neurons and activation functions, but these changes didn’t quite push the accuracy over the .75 goal.</summary>
  
```
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 5
hidden_nodes_layer3 = 3
nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="sigmoid"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```
</details>
<details><summary>Binning: Next I added the name column back. I replaced all the names of doners who donated less than 5 times with other. This raised the accuracy to around .77.</summary>
  
```
# NAME counts
name_counts = application_df.NAME.value_counts()
# Determine which values to replace if counts are less than ...?
replace_name = list(name_counts[name_counts <= 5].index)

# Replace in dataframe
for name in replace_name:
    application_df.NAME = application_df.NAME.replace(name,"Other")
    
# Check to make sure binning was successful
application_df.NAME.value_counts()
```
 ![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/name_counts.png)
						       
</details>
	
<details><summary>Last I was able to get the accuracy to .7876 by dropping the status and special considerations columns.</summary>
	
  
```
application_df.drop(["EIN","STATUS",'SPECIAL_CONSIDERATIONS'],1)
```
  ![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/drop_ein_stat_sc.png)
  </details>

 ### Conclusions: Data Preprocessing
- **What variable(s) are considered the target(s) for your model?** 
	- IS_SUCCESSFUL—Was the money used effectively
- **What variable(s) are considered to be the features for your model?**
  - ASK_AMT—Funding amount requested
  - AFFILIATION—Affiliated sector of industry
  - USE_CASE—Use case for funding
  - ORGANIZATION—Organization type
  - INCOME_AMT—Income classification
  - NAME—Identification column
  - APPLICATION_TYPE—Alphabet Soup application type
  - CLASSIFICATION—Government organization classification
- **What variable(s) are neither targets nor features, and should be removed from the input data?**
	- EIN—Identification column
	- SPECIAL_CONSIDERATIONS—Special consideration for application
	- STATUS—Active status	
	
<h3>Were you able to achieve the target model performance?</h3> Yes 

![](https://github.com/MichelaZ/Neural_Network_Charity_Analysis/blob/main/Resources/Model2_Summary.png)
  
</details>

## Summary: 
I was able to train the model to have an accuracy of  0.7867 with .4410 loss by removing the EIN, status, and special circumstances columns; binning the name, application type, and classification columns; and experimenting with the number of hidden layers, nodes and activation functions. This meets the clients goals of attaining an accuracy of .75, but using a Random Forest I was able to acheive an accuraccy of .772. Random forests are less computationally expensive, more interpretable and require less data than neural networks, so this might be a better model to use. 
	
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)	
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
