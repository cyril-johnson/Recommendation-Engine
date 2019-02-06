import pandas as pd
import cbRec
from sklearn.cross_validation import train_test_split

users_songs_data = pd.read_csv('users_songs_data.csv')
songs_metadata = pd.read_csv('music.csv')

train_data, test_data = train_test_split(users_songs_data, test_size = 0.20, random_state =0) 
user_sample = list(set(train_data['user_id']).intersection(set(test_data['user_id'])))

iscb = cbRec.content_based_recommender(train_data, test_data)
recommended_songs = iscb.recommend(user_sample[1], users_songs_data, songs_metadata)
listened_songs_train_data = iscb.get_user_songs(user_sample[1], train_data)








