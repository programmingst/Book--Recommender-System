#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books = pd.read_csv('books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv') 


# In[3]:


books.head()


# In[4]:


users.head()


# In[5]:


ratings.head()


# In[6]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[7]:


books.isnull().sum()


# In[8]:


users.isnull().sum()


# 

# In[9]:


books.duplicated().sum()


# In[10]:


ratings.duplicated().sum()


# In[11]:


users.duplicated().sum()


# ##### Popularity Based Recommender System
# 
# 
# 

# In[12]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[13]:


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df


# In[14]:


avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
avg_rating_df


# In[ ]:





# In[ ]:





# In[15]:


popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[16]:


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)


# In[17]:


popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# In[60]:


popular_df


# # Collaborative Filtering Based Recommender System
# 
# 
# 

# In[19]:


x =ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
padhe_likhe_users = x[x].index


# In[20]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[21]:


filtered_rating.groupby('Book-Title').count()['Book-Rating']


# In[22]:


filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50


# In[23]:


y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
y[y]


# In[24]:


y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[25]:


famous_books


# In[26]:


filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[27]:


final_ratings =filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[28]:


final_ratings.drop_duplicates()


# In[29]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[30]:


pt


# In[31]:


pt.fillna(0,inplace=True)


# In[32]:


pt


# In[33]:


from sklearn.metrics.pairwise import cosine_similarity


# In[34]:


cosine_similarity(pt)


# In[35]:


cosine_similarity(pt).shape


# In[36]:


similarity_scores = cosine_similarity(pt)


# In[37]:


similarity_scores


# In[38]:


similarity_scores[0]


# In[39]:


similarity_scores.shape


# # Def Recommender Function To Return the5 Book Names suggestions 
# 

# In[40]:


def recommend(book_name):
    # index fetch
    return suggestions


# In[41]:


np.where(pt.index=='Year of Wonders')[0][0]


# In[42]:


np.where(pt.index=='1984')[0][0]


# In[43]:


np.where(pt.index== '4 Blondes')[0][0]


# In[44]:


def recommend(book_name):
    # index fetch
    index = np.where(pt.index=="zoya")[0][0]
    distances = similarity_scores[index]
    return suggestions


# In[45]:


similarity_scores[0]


# In[46]:


list(enumerate(similarity_scores[0]))


# In[47]:


sorted(list(enumerate(similarity_scores[0])),key=lambda x:x[1])


# In[48]:


sorted(list(enumerate(similarity_scores[0])),key=lambda x:x[1], reverse=True)


# In[49]:


sorted(list(enumerate(similarity_scores[0])),key=lambda x:x[1], reverse=True)[1:6]


# In[85]:


def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return data


# In[83]:


recommend('1984')


# In[52]:


pt.index


# In[53]:


pt.index[47]


# In[54]:


pt.index[545]


# In[55]:


recommend('Message in a Bottle')


# In[56]:


recommend('The Notebook')


# In[57]:


recommend('The Da Vinci Code')


# In[58]:


pt.index[545]


# In[59]:


import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))


# In[84]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


# In[ ]:




