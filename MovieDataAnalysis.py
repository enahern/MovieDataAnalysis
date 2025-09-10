import pandas as pd
import numpy as np
import math

'''------------ part i --------------'''
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('/Users/emilyahern/Documents/Sophomore Year/Big Data Analytics/ml-100k/u.data', sep='\t', names=columns)

df = df.iloc[:, :-1]



df = df.dropna()


'''--------------- part ii --------------------'''
umr_matrix = df.pivot_table(index = 'user_id', columns = 'item_id', values = 'rating')

'''----------------- part iii -----------------'''
matrix_copy = umr_matrix.copy()
np.random.seed(351)
users_h = np.random.choice(matrix_copy.index, size=int(0.2 * matrix_copy.shape[0]), replace=False)
ratings_to_predict = []
for user in users_h:
    rate_i = matrix_copy.iloc[user-1].dropna().index.tolist()
    ratings_h = np.random.choice(rate_i, size=int(0.2 * len(rate_i)), replace=False)
    matrix_copy.iloc[user-1, ratings_h-1] = np.nan
    for rate in ratings_h:
        ratings_to_predict.append([user-1,rate-1])


'''------------- part iv --------------'''
# print(matrix_copy.head())
# print(matrix_copy.iloc[0])
def cos_sim(user_1, user_2):
    '''cos_sim = (dot prod(user_1,user_2))/(mag(user_1)/*mag(user_2))'''
    a,b = user_1, user_2
    numerator = np.dot(a,b)
    denom = math.sqrt(np.dot(a,a))*math.sqrt(np.dot(b,b))
    if denom == 0:
        return 0
    else:
        return numerator/denom
    
def knn(user_i, k, matrix):
    sim_list = []
    rating = matrix.iloc[user_i].fillna(0).values
    for i, other_u in matrix.iterrows():
        if user_i != i-1:
            other_u = other_u.fillna(0).values

            cos = cos_sim(other_u, rating)
            sim_list += [[cos,i-1]]
    sim_list.sort(key=lambda x: x[0], reverse=True)
    return sim_list[0:k]


def predict(pred_val, matrix ):
    num = 0
    den = 0
    for sim, i in knn(pred_val[0], 10, matrix):
        o_rate = matrix.iloc[i,pred_val[1]]
        
        if not np.isnan(o_rate):
            num += o_rate *sim
            den += sim
    if den == 0:
        return matrix.iloc[pred_val[0]].mean()
    return num/den

def pred_all(matrix):
    fnl = []
    for user,movie in ratings_to_predict:
        pred = predict([user,movie],matrix)
        fnl.append([[user,movie],pred])
    return fnl
x = pred_all(matrix_copy)
# print(x)

'''----------------------- part v ---------------------------'''

def rmse(prediction, test):
    sum = 0
    for i in range(len(prediction)):
        sum += (prediction[i][1]-umr_matrix.iloc[test[i][0],test[i][1]])**2
    return math.sqrt(sum/len(prediction))

print('---- rmse of user-based cf ----')
print(rmse(x, ratings_to_predict))



'''---------------------- part vi ---------------------------'''
'''Overall, the performance of this model was pretty good! The RMSE, of 1.08
is relatively low. This model has a lot of strengths with utilizing the k-nearest neighbors 
to determine the most similar users. However, I think that one weakness that stood out to me 
was the idea that the users that have the lowest similarity could also be useful to predict
the ratings since their ratings would be very different. I think this could be stronger if 
the ratings based on the similar movies would help lower this too. In fact, item-based cf,
should have a lower RMSE.'''

'''

--------------------------------------------------------------------------------------------




'''

umr_matrix = df.pivot_table(index = 'item_id', columns = 'user_id', values = 'rating')

'''----------------- part iii -----------------'''
matrix_copy = umr_matrix.copy()
np.random.seed(351)
movie_h = np.random.choice(matrix_copy.index, size=int(0.2 * matrix_copy.shape[0]), replace=False)
ratings_to_predict = []
for movie in movie_h:
    rate_i = matrix_copy.iloc[movie-1].dropna().index.tolist()
    ratings_h = np.random.choice(rate_i, size=int(0.2 * len(rate_i)), replace=False)
    matrix_copy.iloc[movie-1, ratings_h-1] = np.nan
    for rate in ratings_h:
        ratings_to_predict.append([movie-1,rate-1])


'''------------- part iv --------------'''
# print(matrix_copy.head())
# print(matrix_copy.iloc[0])
def cos_sim(movie_1, movie_2):
    '''cos_sim = (dot prod(user_1,user_2))/(mag(user_1)/*mag(user_2))'''
    a,b = movie_1, movie_2
    numerator = np.dot(a,b)
    denom = math.sqrt(np.dot(a,a))*math.sqrt(np.dot(b,b))
    if denom == 0:
        return 0
    else:
        return numerator/denom
    
def knn(movie_i, k, matrix):
    sim_list = []
    rating = matrix.iloc[movie_i].fillna(0).values
    for i, other_u in matrix.iterrows():
        if movie_i != i-1:
            other_u = other_u.fillna(0).values

            cos = cos_sim(other_u, rating)
            sim_list += [[cos,i-1]]
    sim_list.sort(key=lambda x: x[0], reverse=True)
    return sim_list[0:k]


def predict(pred_val, matrix ):
    num = 0
    den = 0
    for sim, i in knn(pred_val[0], 10, matrix):
        o_rate = matrix.iloc[i,pred_val[1]]
        
        if not np.isnan(o_rate):
            num += o_rate *sim
            den += sim
    if den == 0:
        return matrix.iloc[pred_val[0]].mean()
    return num/den

def pred_all(matrix):
    fnl = []
    for movie,user in ratings_to_predict:
        pred = predict([movie,user],matrix)
        fnl.append([[movie,user],pred])
    return fnl
x = pred_all(matrix_copy)
# print(x)

'''----------------------- part v ---------------------------'''
def rmse(prediction, test):
    sum = 0
    for i in range(len(prediction)):
        sum += (prediction[i][1]-umr_matrix.iloc[test[i][0],test[i][1]])**2
    return math.sqrt(sum/len(prediction))
print('---- rmse of item-based cf ----')
print(rmse(x, ratings_to_predict))

'''The Item-based callaborative filtering has a lower RMSE than
the user-based cf. This shows us that similar movies are better 
at predicting ratings instead of similar users'''
