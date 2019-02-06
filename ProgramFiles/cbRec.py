import pandas as pd
import numpy as np
import operator

#Class for content based recommender system
class content_based_recommender():
    def __init__(self, trainData, testData):
        self.train_data = trainData
        self.test_data = testData

    #function to get the songs which he has already listened to, as according to the train data
    def get_user_songs(self, user_id, train_data):
        temp = train_data[train_data['user_id']==user_id]
        user_songs_list = temp['song_id'].tolist()
        return user_songs_list
    
    #function to get the songs which are new to the user
    def get_new_songs(self, user_id, dataset, listened_songs):
        allSongs = set(dataset['song_id'].unique().tolist())
        new_songs = list(allSongs - set(listened_songs))
        return new_songs

    #function to recommend new songs to the user based on item-similarity
    def recommend(self, user_id, users_songs_data, songs_metadata):
        
        #Get the songs which user has already listened to, as according to the train data
        listened_songs = self.get_user_songs(user_id, self.train_data)
        
        #Get the songs which user has not listened to
        #new_songs list also contains the songs user has listened to as contained in the test data
        #Our model doesn't know whether those songs are listened to by the user 
        new_songs = self.get_new_songs(user_id, users_songs_data, listened_songs)
        
        #Create dataframe for new songs
        new_songs = pd.DataFrame(new_songs, columns=['song_id'])
        
        #Create dataframe for songs already listened
        listened_songs = pd.DataFrame(listened_songs, columns=['song_id'])
        
        #Merging the song_id's with song's features 
        new_songs_df = pd.merge(songs_metadata, new_songs, on='song_id')
        listened_songs_df = pd.merge(songs_metadata, listened_songs, on='song_id')
        new_songs_df = new_songs_df[['song_id','loudness','tempo']]
        listened_songs_df = listened_songs_df[['song_id','loudness','tempo']]
        
        #Matrix of features for new songs
        new_songs_mat = np.array(new_songs_df[['loudness','tempo']])
        new_songs_songids = np.array(new_songs_df[['song_id']])
        
        #Matrix of features for old songs
        listened_songs_mat = np.array(listened_songs_df[['loudness','tempo']])
        listened_songs_songids = np.array(listened_songs_df[['song_id']])
                
        distance_dict = {}
        
        #loop to calculate similarity measure based on euclidean distance
        for i in range(new_songs_mat.shape[0]):
            sqdifference_matrix = (np.tile(new_songs_mat[i], (len(listened_songs), 1)) - listened_songs_mat)**2
            euclidean_distance_with_listened_songs = sqdifference_matrix.sum(axis = 1)
            distance_dict[new_songs_songids[i][0]] = euclidean_distance_with_listened_songs.sum()
        
        #sorting new songs by their similarity with listened songs
        sorted_distance = sorted(distance_dict.items(), key=operator.itemgetter(1))
        
        #getting top 5 most similar songs to already listened songs
        recommended_songids = []
                if len(sorted_distance) > 5:
            recommended_songids = sorted_distance[:5]
        else:
            recommended_songids = sorted_distance
        
        recommended_songs= []    
        
        i = 0
        
        for item in recommended_songids:
            recommended_songs_df = users_songs_data[users_songs_data['song_id']==item[0]]
            recommended_songs_df['song'] = recommended_songs_df['title'] + " - " + recommended_songs_df['artist_name']
            recommended_songs.append(recommended_songs_df['song'].tolist())
            i += 1
            if i==2:
                break
            
        return recommended_songs