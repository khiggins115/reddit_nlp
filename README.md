# Project 3: Reddit NLP

### Problem Statement

Given text from two different subreddits (r/C_S_T and r/conspiracy) we train a model to accurately and consistently classify unseen text as belonging to the Conspiracy subreddit or the "Critical Shower Thoughts" subreddit?


### Executive Summary

Reddit, also known as the "front page of the internet" is a popular "American social news aggregation, web content rating, and discussion website" (Wikipedia). Topics for discussion are typically organized into "subreddits" where users post relevant news/info/questions. In recent years, Reddit has also become known less for the intellectually spirited discourses that it claims to foster and more as a breeding ground for the proliferation of conspiracy theories. But how do we distill thought exercises from conspiracy theories? Can we distinguish between them? This project uses Natural Language Processing and Text classification in an attempt to do just that. 

Using the PushShift API, random samples of posts from Jan 01 2017 - the present were pulled from the r/C_S_T and r/conspiracy subreddits. We scraped 500 posts per subreddit, pulling attributes such as title, the post text, the number of comments and the score (among others listed in the data dictionary). The data were then cleaned using RegExp and a WordNetLemmatizer. The clean data were explored and visualized to identify any potentially significant relationships between the variables. 

Once the data were sufficiently explored, models were created. Using pipelines and GridSearchCV (as well as incorporating different methods of text vectorization such as CountVectorizer and TfidfVectorizer), a variety of Naive Bayes models were fit and tuned to the optimal hyperparameters. The second phase of modeling explored other, non-Bayesian classifiers. Ultimately, a model incorporating CountVectorizer(), StandardScaler() and ComplementNB was determined to be the best at classifying the text. 



### Contents
- 01 Data Collection
- 02 Data Cleaning 
- 03 EDA & Data Viz
- 04 Model 1 - Bayes Classification
- 05 Model 2 - Additional Modeling
- Data
 - subreddits.csv
 - clean_subreddits.csv
- Deliverables
 - project_3_KH.pdf
 - project_3_KH.pptx
 - img


### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|num_comments|int64|Number of Comments|
|score|int64|Post score|
|selftext|object|Body text of submission post|
|subreddit|object|Name of the subreddit (target)|
|title|object|Title of submission post|
|clean_selftext|object|cleaned selftext|
|clean_titlet|object|cleaned title|
|wordcount_clean_selftext|int64|words in cleaned selftext|
|wordcount_clean_title|int64|words in cleaned title|


### Conclusion

While the Bayesian model outperformed a simple Logistic Regression, the interpretability of the model is always a factor worth consideration, and for that reason I would say that a finely tuned Logistic Regression model is the most optimal all around model. 

### Areas for Future Studies

In the future, I'd like to spend time doing multivariate modeling by including num_comments or score instead of focusing only on the selftext of each post. I would perhaps incorporate lemmatization as a preprocessor rather than in the data cleaning phase. I'd also like to explore the boosted models that we've experimented with in class. 


### Sources
Breakfast Hour
https://www.reddit.com/r/conspiracy/
https://www.reddit.com/r/C_S_T 
https://en.wikipedia.org/wiki/Reddit
https://pushshift.io/api-parameters/  
https://scikit-learn.org/stable/modules/naive_bayes.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html
