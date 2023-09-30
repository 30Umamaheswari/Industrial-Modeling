#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd

data = pd.read_excel("C:\\Users\\visha\\Desktop\\Copper_Set.xlsx")
# data


# In[3]:


data.shape


# In[4]:


data.head(3)


# In[107]:


l = data[data['status'] == 'Lost']
l


# In[5]:


data.info()


# In[111]:


# Remove rows with NaN values in the 'quantity tons' column
data = data.dropna(subset=['quantity tons'])

# Calculate the minimum and maximum values
min_value = data['quantity tons'].min()
max_value = data['quantity tons'].max()

print("Minimum value:", min_value)
print("Maximum value:", max_value)


# In[110]:


non_numeric_values = data[~data['quantity tons'].apply(pd.to_numeric, errors='coerce').notna()]['quantity tons']
print("Non-numeric values:", non_numeric_values)


# In[7]:


data.isnull().sum()


# In[8]:


# data['material_ref'].unique()


# In[9]:


columns = ['quantity tons', 'country', 'status', 'item type', 'application', 'thickness', 'width', 'product_ref','selling_price']
df = data[columns]
# df


# In[10]:


df.isnull().sum()


# In[11]:


df['status'].dtype
print(type(df['status'].iloc[0]))


# In[12]:


df['status'].value_counts()


# In[13]:


df1 = df.copy()


# In[14]:


df1['quantity tons'] = pd.to_numeric(df1['quantity tons'], errors='coerce')


# In[15]:


df1['selling_price'] = pd.to_numeric(df1['selling_price'])
df1['width'] = pd.to_numeric(df1['width'])
df1['thickness'] = pd.to_numeric(df1['thickness'])
df1['product_ref'] = pd.to_numeric(df1['product_ref'])
df1['country'] = pd.to_numeric(df1['country'])
df1['application'] = pd.to_numeric(df1['application'])


# In[16]:


mode_value = df1['status'].mode()
mode_value


# In[17]:


df1['status'] = df1['status'].str.strip()
df1['status'].fillna(df1['status'].mode()[0], inplace=True)


# In[18]:


sns.boxplot(df1['thickness'])
plt.show()


# In[19]:


df1['thickness'] = df1['thickness'].fillna(df1['thickness'].median())


# In[20]:


sns.boxplot(df1['selling_price'])
plt.show()


# In[21]:


df1['selling_price'] = df1['selling_price'].fillna(df1['selling_price'].mean())


# In[22]:


df1['quantity tons'] = df1['quantity tons'].fillna(df1['quantity tons'].mean())


# In[108]:


df1['quantity tons'] = pd.to_numeric(df1['quantity tons'], errors='coerce')

min_value = df1['quantity tons'].min()
max_value = df1['quantity tons'].max()

print("Minimum value:", min_value)
print("Maximum value:", max_value)


# In[23]:


sns.boxplot(df1['country'])
plt.show()


# In[24]:


df1['country'] = df1['country'].fillna(df1['country'].mean())


# In[25]:


sns.boxplot(df1['product_ref'])
plt.show()


# In[26]:


df1['product_ref'] = df1['product_ref'].fillna(df1['product_ref'].mean())


# In[27]:


sns.boxplot(df1['application'])
plt.show()


# In[28]:


df1['application'] = df1['application'].fillna(df1['application'].median())


# In[29]:


df1.isnull().sum()


# # Data Visualization

# In[30]:


sns.countplot(data=df1, x = df1['status'])
plt.xticks(rotation=45)

plt.show()


# In[31]:


df1['status'].unique()


# In[32]:


import numpy as np

item = df1['item type'].unique()

mean_selling_price = df1.groupby('item type')['selling_price'].mean()

colors = plt.cm.tab20(np.arange(len(item)))


plt.bar(item, mean_selling_price, color=colors)
plt.xlabel('Item Type')
plt.ylabel('Selling Price')
plt.title('Selling Price by Item Type')
plt.xticks(rotation=45)

plt.show()


# In[33]:


import numpy as np

item = df1['item type'].unique()

mean_quantity = df1.groupby('item type')['quantity tons'].mean()

colors = plt.cm.tab20(np.arange(len(item)))


plt.bar(item, mean_quantity, color=colors)
plt.xlabel('Item Type')
plt.ylabel('Quantity Tons')
plt.title('Quantity Tons by Item Type')
plt.xticks(rotation=45)

plt.show()


# # Skew view

# In[34]:


sns.distplot(df1['quantity tons'])
plt.show()


# In[35]:


sns.distplot(df1['thickness'])
plt.show()


# In[36]:


sns.distplot(df1['width'])
plt.show()


# In[37]:


sns.distplot(df1['selling_price'])
plt.show()


# In[38]:


sns.distplot(df1['country'])
plt.show()


# In[39]:


sns.distplot(df1['application'])
plt.show()


# In[40]:


sns.distplot(df1['product_ref'])
plt.show()


# In[43]:


df1['quantity tons'] = pd.to_numeric(df1['quantity tons'], errors='coerce')
df1['selling_price'] = pd.to_numeric(df1['selling_price'], errors='coerce')
df1['width'] = pd.to_numeric(df1['width'], errors='coerce')
df1['thickness'] = pd.to_numeric(df1['thickness'], errors='coerce')
df1['country'] = pd.to_numeric(df1['country'], errors='coerce')
df1['application'] = pd.to_numeric(df1['application'], errors='coerce')
df1['product_ref'] = pd.to_numeric(df1['product_ref'], errors='coerce')


# # Log Transformation

# In[44]:


df1['selling_price'] = np.log(df1['selling_price'])
sns.distplot(df1['selling_price'])
plt.show()


# In[45]:


df1['quantity tons_log'] = np.log(df1['quantity tons'])
sns.distplot(df1['quantity tons_log'])
plt.show()


# In[46]:


df1['thickness_log'] = np.log(df1['thickness'])
sns.distplot(df1['thickness_log'])
plt.show()


# In[47]:


df1['width_log'] = np.log(df1['width'])
sns.distplot(df1['width_log'])
plt.show()


# In[48]:


modify_columns = df1.columns
modify_columns


# In[49]:


v = df1[['country', 'application', 'product_ref', 'selling_price', 'quantity tons_log',
       'thickness_log', 'width_log']]


# In[50]:


v.isnull().sum()


# In[51]:


sns.heatmap(v.corr(), annot=True, cmap='YlOrBr')


# In[52]:


df1[['quantity tons_log']] = df1[['quantity tons_log']].fillna(df1[['quantity tons_log']].median())
df1[['selling_price']] = df1[['selling_price']].fillna(df1[['selling_price']].median())


# In[53]:


df1[['quantity tons_log', 'selling_price']].isnull().sum()


# In[54]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


# In[55]:


x = df1[['country', 'application', 'product_ref', 'quantity tons_log',
       'thickness_log', 'width_log', 'status', 'item type']]
y = df1['selling_price']


# In[56]:


df1.isna().sum()


# In[57]:


df1['country'].value_counts()


# # Encoding Categorical Variable

# In[58]:


itemType_ohe = OneHotEncoder(handle_unknown='ignore')
itemType_ohe.fit(x[['item type']])

x_ohe_item = itemType_ohe.fit_transform(x[['item type']]).toarray()
x_ohe_item


# In[59]:


import numpy as np

# Check for NaN values in x_ohe_item
nan_indices = np.isnan(x_ohe_item)

# Count the number of NaN values
nan_count = np.sum(nan_indices)

# Display the indices and count of NaN values
print("Indices of NaN values:", np.argwhere(nan_indices))
print("Number of NaN values:", nan_count)


# In[60]:


status_ohe = OneHotEncoder(handle_unknown='ignore')
status_ohe.fit(x[['status']])

x_ohe_status = status_ohe.fit_transform(x[['status']]).toarray()
x_ohe_status


# In[61]:


x_ohe_item[0]


# In[62]:


# Handle missing values (NaN) in your dataset
from sklearn.impute import SimpleImputer


# In[63]:


print(df1['quantity tons_log'])


# In[64]:


x = np.concatenate((x[['country', 'application', 'product_ref', 'quantity tons_log',
       'thickness_log', 'width_log']].values, x_ohe_item, x_ohe_status), axis=1)

print(x)
# imputer = SimpleImputer(strategy='mean')
# x = imputer.fit_transform(x)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
# dtr = DecisionTreeRegressor()

# param_grid = {'max_depth': [2, 5, 10, 20],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4],
#               'max_features': ['sqrt', 'log2']}

# grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid,cv=5)
# grid_search.fit(X_train, y_train)
# print("Hyperparameters:", grid_search.best_params_)

# best_model = grid_search.best_estimator_
# y_predict = best_model.predict(X_test)

# mse = mean_squared_error(y_test, y_predict)
# r2 = r2_score(y_test, y_predict)
# print('Mean squared error:', mse)
# print('R-squared:', r2)

# Hyperparameters: {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10}
# Mean squared error: 0.0008131798794653432
# R-squared: 0.9429559094466783


# In[ ]:





# In[65]:


print(x[0])


# In[66]:


print(y)


# In[67]:


y.describe()


# In[68]:


print(np.exp(6.643822))


# In[69]:


print(np.log(18))


# In[70]:


new_sample = np.array([[np.log(40), 10, np.log(250), 0, 28,1670798778,'PL','Won']])
new_sample_ohe = itemType_ohe.transform(new_sample[:, [6]]).toarray()
new_sample_be = status_ohe.transform(new_sample[:, [7]]).toarray()
new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, ]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scaler.transform(new_sample)
new_pred = best_model.predict(new_sample1)
print('Predicted selling price:', np.exp(new_pred))


# In[71]:


print(new_sample)


# In[72]:


print(new_sample_ohe)


# In[73]:


print(new_sample_be)


# In[74]:


import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('type.pkl', 'wb') as f:
    pickle.dump(itemType_ohe, f)
with open('status.pkl', 'wb') as f:
    pickle.dump(status_ohe, f)


# In[76]:


# pip show scikit-learn


# In[77]:


# !pip install scikit-learn==1.3.1


# # Status

# In[83]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier


# In[79]:


# x = df1[['country', 'application', 'product_ref', 'quantity tons_log',
#           'thickness_log', 'width_log', 'item type']]
# y = df1['status']


# In[84]:


item_type = LabelEncoder()
df1['item type encoded'] = item_type.fit_transform(df1['item type'])


x = df1[['country', 'application', 'product_ref', 'quantity tons_log',
          'thickness_log', 'width_log', 'item type encoded']]
y = df1['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# k = 3  
# knn_classifier = KNeighborsClassifier(n_neighbors=k)
# knn_classifier.fit(x_train, y_train)

# y_pred = knn_classifier.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy of the K-NN model: {accuracy:.2f}') # Accuracy of the K-NN model: 0.69


# In[89]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelBinarizer

Y = df1['status']
X= df1[['quantity tons_log','selling_price','item type','application','thickness_log','width','country','product_ref']]

# encoding categorical variables
type_encoding = OneHotEncoder(handle_unknown='ignore')
type_encoding.fit(X[['item type']])

X_type_encoding = type_encoding.fit_transform(X[['item type']]).toarray()

y_encode = LabelBinarizer()
y_encode.fit(Y) 

y = y_encode.fit_transform(Y)

# independent features after encoding
X = np.concatenate((X[['quantity tons_log', 'selling_price','application', 'thickness_log', 'width','country','product_ref']].values, X_type_encoding), axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[90]:


print(classification_report(y_test, y_pred))


# In[95]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# Assume y_pred contains the predicted probabilities for each class
# Make sure y_pred has the shape (n_samples, n_classes)

# Binarize the labels (one-hot encoding)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)  # y_test should be the true labels

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class')
plt.legend(loc="lower right")
plt.show()


# In[112]:


new_sample = np.array([[np.log(11), np.log(1500), 10, np.log(4), 1186, 78.0, 1668701718, 'W']])

# Encode the 'item type' in new_sample
new_sample_ohe = type_encoding.transform(new_sample[:, [7]]).toarray()

# Create the sample for prediction by excluding the 'item type' column
new_sample_for_prediction = new_sample[:, [0, 1, 2, 3, 4, 5, 6]]

# Concatenate the encoded 'item type'
new_sample = np.concatenate((new_sample_for_prediction, new_sample_ohe), axis=1)

# Standardize the data
new_sample = scaler.transform(new_sample)

# Predict the status
new_pred = dtc.predict(new_sample)

if (new_pred == 1).any():
    print('The status is: Won')
else:
    print('The status is: Lost')


# In[102]:


import pickle

with open('model_status.pkl', 'wb') as file:
    pickle.dump(dtc, file)
with open('scaler_status.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('c_status.pkl', 'wb') as f:
    pickle.dump(type_encoding, f)

