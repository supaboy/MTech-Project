{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt \n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense,LSTM,Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "</div>"
      ],
      
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv('_train_data.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"Close\"]=pd.to_numeric(data.Close,errors='coerce')\n",
    "data = data.dropna()\n",
    "trainData = data.iloc[:,4:5].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 1149 entries, 0 to 1257\n",
      "Data columns (total 6 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Date    1149 non-null   object \n",
      " 1   Open    1149 non-null   float64\n",
      " 2   High    1149 non-null   float64\n",
      " 3   Low     1149 non-null   float64\n",
      " 4   Close   1149 non-null   float64\n",
      " 5   Volume  1149 non-null   object \n",
      "dtypes: float64(4), object(2)\n",
      "memory usage: 62.8+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1149, 1)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sc = MinMaxScaler(feature_range=(0,1))\n",
    "trainData = sc.fit_transform(trainData)\n",
    "trainData.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = []\n",
    "y_train = []\n",
    "\n",
    "for i in range (60,1149): #60 : timestep // 1149 : length of the data\n",
    "    X_train.append(trainData[i-60:i,0]) \n",
    "    y_train.append(trainData[i,0])\n",
    "\n",
    "X_train,y_train = np.array(X_train),np.array(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1089, 60, 1)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #adding the batch_size axis\n",
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "\n",
    "model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "model.add(LSTM(units=100, return_sequences = True))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "model.add(LSTM(units=100, return_sequences = True))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "model.add(LSTM(units=100, return_sequences = False))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "model.add(Dense(units =1))\n",
    "model.compile(optimizer='adam',loss=\"mean_squared_error\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      " - 12s - loss: 0.0322\n",
      "Epoch 2/20\n",
      " - 10s - loss: 0.0163\n",
      "Epoch 3/20\n",
      " - 12s - loss: 0.0097\n",
      "Epoch 4/20\n",
      " - 16s - loss: 0.0086\n",
      "Epoch 5/20\n",
      " - 13s - loss: 0.0083\n",
      "Epoch 6/20\n",
      " - 13s - loss: 0.0071\n",
      "Epoch 7/20\n",
      " - 11s - loss: 0.0063\n",
      "Epoch 8/20\n",
      " - 10s - loss: 0.0074\n",
      "Epoch 9/20\n",
      " - 9s - loss: 0.0064\n",
      "Epoch 10/20\n",
      " - 9s - loss: 0.0080\n",
      "Epoch 11/20\n",
      " - 9s - loss: 0.0066\n",
      "Epoch 12/20\n",
      " - 9s - loss: 0.0060\n",
      "Epoch 13/20\n",
      " - 9s - loss: 0.0059\n",
      "Epoch 14/20\n",
      " - 10s - loss: 0.0058\n",
      "Epoch 15/20\n",
      " - 9s - loss: 0.0051\n",
      "Epoch 16/20\n",
      " - 12s - loss: 0.0061\n",
      "Epoch 17/20\n",
      " - 15s - loss: 0.0052\n",
      "Epoch 18/20\n",
      " - 14s - loss: 0.0054\n",
      "Epoch 19/20\n",
      " - 13s - loss: 0.0048\n",
      "Epoch 20/20\n",
      " - 12s - loss: 0.0045\n"
     ]
    }
   ],
   "source": [
    "hist = model.fit(X_train, y_train, epochs = 20, batch_size = 32, verbose=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(hist.history['loss'])\n",
    "plt.title('Training model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(192, 60, 1)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "testData = pd.read_csv('_test_data.csv')\n",
    "testData[\"Close\"]=pd.to_numeric(testData.Close,errors='coerce')\n",
    "testData = testData.dropna()\n",
    "testData = testData.iloc[:,4:5]\n",
    "y_test = testData.iloc[60:,0:].values \n",
    "#input array for the model\n",
    "inputClosing = testData.iloc[:,0:].values \n",
    "inputClosing_scaled = sc.transform(inputClosing)\n",
    "inputClosing_scaled.shape\n",
    "X_test = []\n",
    "length = len(testData)\n",
    "timestep = 60\n",
    "for i in range(timestep,length):  \n",
    "    X_test.append(inputClosing_scaled[i-timestep:i,0])\n",
    "X_test = np.array(X_test)\n",
    "X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))\n",
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted_price = sc.inverse_transform(y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(y_test, color = 'red', label = 'Actual Stock Price')\n",
    "plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')\n",
    "plt.title('stock price prediction')\n",
    "plt.xlabel('Time')\n",
    "plt.ylabel('Stock Price')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
