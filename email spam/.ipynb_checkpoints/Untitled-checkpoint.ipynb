{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "52f659bd-631f-4eac-8f40-91862932eacf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "15489c63-8d08-46c2-915d-f7c8179a711d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('spam.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "07c03e39-581a-43aa-ae36-8124f5c39fe6",
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
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Category                                            Message\n",
       "0      ham  Go until jurong point, crazy.. Available only ...\n",
       "1      ham                      Ok lar... Joking wif u oni...\n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      ham  U dun say so early hor... U c already then say...\n",
       "4      ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d4f1e3e7-6d04-49f9-bf81-3983fa0e6d29",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Category    0\n",
       "Message     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c55e86cb-39f2-40ed-bfee-75124ba125ea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5572 entries, 0 to 5571\n",
      "Data columns (total 2 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   Category  5572 non-null   object\n",
      " 1   Message   5572 non-null   object\n",
      "dtypes: object(2)\n",
      "memory usage: 87.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4d2b52c0-1bf9-4793-9325-6796ac4ee7f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df['Category'] == 'spam' , 'Category',] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "380d6f38-b7cc-4c4d-a1f0-b59adf96756d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df['Category'] == 'ham' , 'Category',] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6c08ec0c-f4e2-4296-87b0-68efc09c083b",
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
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5567</th>\n",
       "      <td>0</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5568</th>\n",
       "      <td>1</td>\n",
       "      <td>Will ü b going to esplanade fr home?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5569</th>\n",
       "      <td>1</td>\n",
       "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5570</th>\n",
       "      <td>1</td>\n",
       "      <td>The guy did some bitching but I acted like i'd...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5571</th>\n",
       "      <td>1</td>\n",
       "      <td>Rofl. Its true to its name</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5572 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Category                                            Message\n",
       "0           1  Go until jurong point, crazy.. Available only ...\n",
       "1           1                      Ok lar... Joking wif u oni...\n",
       "2           0  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3           1  U dun say so early hor... U c already then say...\n",
       "4           1  Nah I don't think he goes to usf, he lives aro...\n",
       "...       ...                                                ...\n",
       "5567        0  This is the 2nd time we have tried 2 contact u...\n",
       "5568        1               Will ü b going to esplanade fr home?\n",
       "5569        1  Pity, * was in mood for that. So...any other s...\n",
       "5570        1  The guy did some bitching but I acted like i'd...\n",
       "5571        1                         Rofl. Its true to its name\n",
       "\n",
       "[5572 rows x 2 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "002fcac1-2eb8-4f3a-9a0c-12380cb186fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.Message"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "461d7a23-edf3-44ce-b79b-81e1353b7ff1",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df.Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "446608a4-3bc5-4de8-8771-536a23dc3396",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4aeefd0a-94a6-49a5-95f5-e7b807506744",
   "metadata": {},
   "outputs": [],
   "source": [
    "vector =TfidfVectorizer(min_df = 1 , lowercase =True , stop_words = 'english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "b4a25242-ec1c-4d25-855b-10777f790a63",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_vec = vector.fit_transform(X_train)\n",
    "X_test_vec = vector.fit_transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "27c64d04-9f80-4d59-9710-eb5d25846244",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = y_train.astype(int)\n",
    "y_test = y_test.astype(int)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "f760f532-0fcd-4cc0-842a-1795830bf7e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LogisticRegression()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c2efcfc4-151c-48a0-9f97-088ed02e1e48",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: 'Reply to win £100 weekly! Where will the 2006 FIFA World Cup be held? Send STOP to 87239 to end service'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[28], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mmodel\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX_train\u001b[49m\u001b[43m \u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my_train\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\base.py:1389\u001b[0m, in \u001b[0;36m_fit_context.<locals>.decorator.<locals>.wrapper\u001b[1;34m(estimator, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1382\u001b[0m     estimator\u001b[38;5;241m.\u001b[39m_validate_params()\n\u001b[0;32m   1384\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[0;32m   1385\u001b[0m     skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[0;32m   1386\u001b[0m         prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[0;32m   1387\u001b[0m     )\n\u001b[0;32m   1388\u001b[0m ):\n\u001b[1;32m-> 1389\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfit_method\u001b[49m\u001b[43m(\u001b[49m\u001b[43mestimator\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:1222\u001b[0m, in \u001b[0;36mLogisticRegression.fit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m   1219\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   1220\u001b[0m     _dtype \u001b[38;5;241m=\u001b[39m [np\u001b[38;5;241m.\u001b[39mfloat64, np\u001b[38;5;241m.\u001b[39mfloat32]\n\u001b[1;32m-> 1222\u001b[0m X, y \u001b[38;5;241m=\u001b[39m \u001b[43mvalidate_data\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1223\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1224\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1225\u001b[0m \u001b[43m    \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1226\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mcsr\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1227\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43m_dtype\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1228\u001b[0m \u001b[43m    \u001b[49m\u001b[43morder\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mC\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1229\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_large_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msolver\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01mnot\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mliblinear\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43msag\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43msaga\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1230\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1231\u001b[0m check_classification_targets(y)\n\u001b[0;32m   1232\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclasses_ \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39munique(y)\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\utils\\validation.py:2961\u001b[0m, in \u001b[0;36mvalidate_data\u001b[1;34m(_estimator, X, y, reset, validate_separately, skip_check_array, **check_params)\u001b[0m\n\u001b[0;32m   2959\u001b[0m         y \u001b[38;5;241m=\u001b[39m check_array(y, input_name\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124my\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mcheck_y_params)\n\u001b[0;32m   2960\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 2961\u001b[0m         X, y \u001b[38;5;241m=\u001b[39m \u001b[43mcheck_X_y\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mcheck_params\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   2962\u001b[0m     out \u001b[38;5;241m=\u001b[39m X, y\n\u001b[0;32m   2964\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m no_val_X \u001b[38;5;129;01mand\u001b[39;00m check_params\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mensure_2d\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mTrue\u001b[39;00m):\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\utils\\validation.py:1370\u001b[0m, in \u001b[0;36mcheck_X_y\u001b[1;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[0;32m   1364\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m   1365\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mestimator_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m requires y to be passed, but the target y is None\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   1366\u001b[0m     )\n\u001b[0;32m   1368\u001b[0m ensure_all_finite \u001b[38;5;241m=\u001b[39m _deprecate_force_all_finite(force_all_finite, ensure_all_finite)\n\u001b[1;32m-> 1370\u001b[0m X \u001b[38;5;241m=\u001b[39m \u001b[43mcheck_array\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1371\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1372\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43maccept_sparse\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1373\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_large_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43maccept_large_sparse\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1374\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1375\u001b[0m \u001b[43m    \u001b[49m\u001b[43morder\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43morder\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1376\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcopy\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcopy\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1377\u001b[0m \u001b[43m    \u001b[49m\u001b[43mforce_writeable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mforce_writeable\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1378\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_all_finite\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_all_finite\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1379\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_2d\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_2d\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1380\u001b[0m \u001b[43m    \u001b[49m\u001b[43mallow_nd\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mallow_nd\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1381\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_min_samples\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_min_samples\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1382\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_min_features\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_min_features\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1383\u001b[0m \u001b[43m    \u001b[49m\u001b[43mestimator\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mestimator\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1384\u001b[0m \u001b[43m    \u001b[49m\u001b[43minput_name\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mX\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1385\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1387\u001b[0m y \u001b[38;5;241m=\u001b[39m _check_y(y, multi_output\u001b[38;5;241m=\u001b[39mmulti_output, y_numeric\u001b[38;5;241m=\u001b[39my_numeric, estimator\u001b[38;5;241m=\u001b[39mestimator)\n\u001b[0;32m   1389\u001b[0m check_consistent_length(X, y)\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\utils\\validation.py:1055\u001b[0m, in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_all_finite, ensure_non_negative, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)\u001b[0m\n\u001b[0;32m   1053\u001b[0m         array \u001b[38;5;241m=\u001b[39m xp\u001b[38;5;241m.\u001b[39mastype(array, dtype, copy\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n\u001b[0;32m   1054\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1055\u001b[0m         array \u001b[38;5;241m=\u001b[39m \u001b[43m_asarray_with_order\u001b[49m\u001b[43m(\u001b[49m\u001b[43marray\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43morder\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43morder\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mxp\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mxp\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1056\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m ComplexWarning \u001b[38;5;28;01mas\u001b[39;00m complex_warning:\n\u001b[0;32m   1057\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m   1058\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mComplex data not supported\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(array)\n\u001b[0;32m   1059\u001b[0m     ) \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mcomplex_warning\u001b[39;00m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\utils\\_array_api.py:839\u001b[0m, in \u001b[0;36m_asarray_with_order\u001b[1;34m(array, dtype, order, copy, xp, device)\u001b[0m\n\u001b[0;32m    837\u001b[0m     array \u001b[38;5;241m=\u001b[39m numpy\u001b[38;5;241m.\u001b[39marray(array, order\u001b[38;5;241m=\u001b[39morder, dtype\u001b[38;5;241m=\u001b[39mdtype)\n\u001b[0;32m    838\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 839\u001b[0m     array \u001b[38;5;241m=\u001b[39m \u001b[43mnumpy\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43masarray\u001b[49m\u001b[43m(\u001b[49m\u001b[43marray\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43morder\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43morder\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    841\u001b[0m \u001b[38;5;66;03m# At this point array is a NumPy ndarray. We convert it to an array\u001b[39;00m\n\u001b[0;32m    842\u001b[0m \u001b[38;5;66;03m# container that is consistent with the input's namespace.\u001b[39;00m\n\u001b[0;32m    843\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m xp\u001b[38;5;241m.\u001b[39masarray(array)\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\pandas\\core\\series.py:1031\u001b[0m, in \u001b[0;36mSeries.__array__\u001b[1;34m(self, dtype, copy)\u001b[0m\n\u001b[0;32m    981\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    982\u001b[0m \u001b[38;5;124;03mReturn the values as a NumPy array.\u001b[39;00m\n\u001b[0;32m    983\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1028\u001b[0m \u001b[38;5;124;03m      dtype='datetime64[ns]')\u001b[39;00m\n\u001b[0;32m   1029\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   1030\u001b[0m values \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_values\n\u001b[1;32m-> 1031\u001b[0m arr \u001b[38;5;241m=\u001b[39m \u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43masarray\u001b[49m\u001b[43m(\u001b[49m\u001b[43mvalues\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1032\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m using_copy_on_write() \u001b[38;5;129;01mand\u001b[39;00m astype_is_view(values\u001b[38;5;241m.\u001b[39mdtype, arr\u001b[38;5;241m.\u001b[39mdtype):\n\u001b[0;32m   1033\u001b[0m     arr \u001b[38;5;241m=\u001b[39m arr\u001b[38;5;241m.\u001b[39mview()\n",
      "\u001b[1;31mValueError\u001b[0m: could not convert string to float: 'Reply to win £100 weekly! Where will the 2006 FIFA World Cup be held? Send STOP to 87239 to end service'"
     ]
    }
   ],
   "source": [
    "model.fit(X_train , y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48b0a0d8-7e16-4628-95bb-0efd30df1835",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
