## Recommendation Engine
A Movie Recommendation Engine based on collaborative filtering to predict the name of the movie based upon the reviews of the other critics having similar taste.
![alt text](https://github.com/deepankarkotnala/Recommendation_Engine/blob/master/images/movie_recommendation.png)
__________________________________________________________________________________________________________

It's pretty clear from the below image that how a recommendation system works. Well, there are many more types of recommendation systems and below is just an example of an item-based recommendation system.

![alt text](https://github.com/deepankarkotnala/Recommendation_Engine/blob/master/images/recommendation_system.png)

##### Import Packages
```python
from math import sqrt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```
##### Getting more than one output Line
```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


##### Getting the Dataset
```python
movies= pd.read_csv(r"D:\iiitB\Python\Recommendation_Engine\movies.csv") # -------- nrows=5000)
movies.head()

ratings = pd.read_csv(r"D:\iiitB\Python\Recommendation_Engine\ratings.csv",usecols=['userId','movieId','rating']) # -------- nrows=1000)
ratings.head()

tags= pd.read_csv(r'D:\iiitB\Python\Recommendation_Engine\tags.csv')
tags.head()
```
![alt text](https://github.com/deepankarkotnala/Recommendation_Engine/blob/master/images/Data.JPG)

Deleting unnecessary columns [Not using the timestamp column for our analysis in the meanwhile]
```python
del tags['timestamp']
tags.head()
```

Having a look at the columns of movies df
```python
movies.info()
```

Having a look at the distribution of ratings in ratings df
```python
ratings['rating'].describe(include='all')

ratings.groupby('rating')['movieId'].nunique()
# %matplotlib inline
ratings.hist(column='rating', figsize=(6,6), bins=5, grid=False, edgecolor='black')
```
![alt text](https://github.com/deepankarkotnala/Recommendation_Engine/blob/master/images/Plot_1.JPG)
```python
tag_counts = tags['tag'].value_counts()
tag_counts[:15]
tag_counts[:15].plot(kind='bar' , figsize=(6,6), edgecolor='black')
```
![alt text](https://github.com/deepankarkotnala/Recommendation_Engine/blob/master/images/Plot_2.JPG)

```python
movies['movieId'].count()
```

Removing movies with no genre
```python
genre_filter= (movies['genres'] == '(no genres listed)')

movies=movies[~genre_filter]
```
###### Because removing filtered rows does not reindex the dataframe, so we have to reindex the dataframe by our own

```python
movies=movies.reset_index(drop=True)  
```

Checking total genres present in DataSet
```python
genres_count= {}
for row in range(movies['movieId'].count()):
    for genre in movies['genres'][row].split("|"):
        if(genre != ''):
            genres_count[genre]= genres_count.get(genre,0)+1
        
genres_count
```

```python
fig, ax = plt.subplots(figsize=(15,10))
plt.barh(range(len(genres_count)), list(genres_count.values()))
plt.yticks(range(len(genres_count)),list(genres_count.keys()))
plt.xlabel('Movie Count')
plt.title("Genre Popularty")
for i, v in enumerate(genres_count.values()):
    ax.text(v + 20, i + .10, v)
```
![alt text](https://github.com/deepankarkotnala/Recommendation_Engine/blob/master/images/Plot_3.JPG)

## Observations:

There are high number of movies from genre Drama & Comedy. So, they might create abias toward the movies which are from these genres.

Film-noir & IMAX are the least popular category for films

## Euclidean Distance Score

** Euclidean Distance is the square root of the sum of squared differences between corresponding elements of the two vectors.
* Euclidean distance is only appropriate for data measured on the same scale.
* Distance = 1/(1+sqrt of sum of squares between two points)
* Value varies between 0 to 1, where closeness to 1 implies higher similarity.

## Defining a function to calculate the Euclidean Distance between two points
```python
def euclidean_distance(person1,person2):
    #Getting details of person1 and person2
    df_first= ratings.loc[ratings['userId']==person1]
    df_second= ratings.loc[ratings.userId==person2]
    
    #Finding Similar Movies for person1 & person2 
    df= pd.merge(df_first,df_second,how='inner',on='movieId')
    
    #If no similar movie found, return 0 (No Similarity)
    if(len(df)==0): return 0
    
    #sum of squared difference between ratings
    sum_of_squares=sum(pow((df['rating_x']-df['rating_y']),2))
    return 1/(1+sum_of_squares)
``` 
Check whether this function works by passing similar ID, the Corerelation should be 1
```python
euclidean_distance(3,3) 
```

## Pearson Correlation Score 

* Correlation between sets of data is a measure of how well they are related. It shows the linear relationship between two sets of data. In simple terms, it answers the question, Can I draw a line graph to represent the data?

* Value varies between -1 to 1.[ 0-> Not related ; -1 -> perfect negatively corelated ; 1-> perfect positively corelated] 

* Slightly better than Euclidean because it addresses the the situation where the data isn't normalised. Like a User is giving high movie ratings in comparison to AVERAGE user.

## Defining a function to calculate the Pearson Correlation Score between two points
```python
def pearson_score(person1,person2):
    
    #Get detail for Person1 and Person2
    df_first= ratings.loc[ratings.userId==person1]
    df_second= ratings.loc[ratings.userId==person2]
    
    # Getting mutually rated items    
    df= pd.merge(df_first,df_second,how='inner',on='movieId')
    
    # If no rating in common
    n=len(df)
    if n==0: return 0

    #Adding up all the ratings
    sum1=sum(df['rating_x'])
    sum2=sum(df['rating_y'])
    
    ##Summing up squares of ratings
    sum1_square= sum(pow(df['rating_x'],2))
    sum2_square= sum(pow(df['rating_y'],2))
    
    # sum of products
    product_sum= sum(df['rating_x']*df['rating_y'])
    
    ## Calculating Pearson Score
    numerator= product_sum - (sum1*sum2/n)
    denominator=sqrt((sum1_square- pow(sum1,2)/n) * (sum2_square - pow(sum2,2)/n))
    if denominator==0: return 0
    
    r=numerator/denominator
    
    return r
```
Checking function by passing similar ID, Output should be 1
```python
pearson_score(1,1)
```

### Getting the results based on Pearson Score
Returns the best matches for person from the prefs dictionary.
Number of results and similarity function are optional params.
```python
def topMatches(personId,n=5,similarity=pearson_score):
    scores=[(similarity(personId,other),other) for other in ratings.loc[ratings['userId']!=personId]['userId']]
    # Sort the list so the highest scores appear at the top
    scores.sort( )
    scores.reverse( )
    return scores[0:n]

topMatches(1,n=3) ## Getting 3 most similar Users for Example 
```

## Defining a function to get the recommendations
Gets recommendations for a person by using a weighted average of every other user's rankings
```python
def getRecommendation(personId, similarity=pearson_score):
    '''
    totals: Dictionary containing sum of product of Movie Ratings by other user multiplied by weight(similarity)
    simSums: Dictionary containung sum of weights for all the users who have rated that particular movie.
    '''
    totals,simSums= {},{}
    
    df_person= ratings.loc[ratings.userId==personId]
    
    for otherId in ratings.loc[ratings['userId']!=personId]['userId']: # all the UserID except personID
        
        # Getting Similarity with OtherID
        sim=similarity(personId,otherId)
        
        # Ignores Score of Zero or Negatie correlation         
        if sim<=0: continue
            
        df_other=ratings.loc[ratings.userId==otherId]
        
        #Movies not seen by the personID
        movie=df_other[~df_other.isin(df_person).all(1)]
        
        for movieid,rating in (np.array(movie[['movieId','rating']])):
            #similarity* Score
            totals.setdefault(movieid,0)
            totals[movieid]+=rating*sim
            
            #Sum of Similarities
            simSums.setdefault(movieid,0)
            simSums[movieid]+=sim

        # Creating Normalized List
        ranking=[(t/simSums[item],item) for item,t in totals.items()]
        
        # return the sorted List
        ranking.sort()
        ranking.reverse()
        recommendedId=np.array([x[1] for x in ranking])
        
        return list(np.array(movies[movies['movieId'].isin(recommendedId)]['title'])[:20])
``` 
    
## Getting the Recommendation
Returns 20 recommended movie for the given UserID
userId can be ranged from 1 to 671
taking the User_Id as input and recommending movies for the user
```python
user_id = int(input("Enter Your UserId: "))

recommended_movies = getRecommendation(user_id)
print("____________________________________________________")
print("\nRecommended Movies: User {}".format(user_id))
print("____________________________________________________")
print(*recommended_movies, sep='\n')
print("____________________________________________________")

```
#### Sample Output:
![alt text](https://github.com/deepankarkotnala/Recommendation_Engine/blob/master/images/Output.JPG)
