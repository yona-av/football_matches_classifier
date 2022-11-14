# Football matches binary classifier 


## Abstract 
In this project I created a binary classifier (win vs lose and draw). The ability to predic if a team will win, based on match statistics, could be very usefull tool to gain insights. Such a model could be used by a coaching staff of a team, for example develope a more targeted work on attributes that are highly correlated to winning. Another market that could benefit from this kind of a model is the betting market.

I gathered the data from rbref.com with a scraper I wrote using beautifulsoup. Then I did EDA and some data-processing that mostly included dealing with nan values. The next step was feature selction using: p-value test, correlation test and omitting features tha might indicate a win (e.g scored_goals, clean_sheet). 
For the training part I tried three different classifiers : **RandomForest**, **SGD** and **logistic regression** which gave the best results : **f-score** = 0.82 and cross-validation cv=3, gave an average of **0.82**, **0.79**, **0.78** for **accuracy**, **precision** and **recall** accordingly. 
The **prediction** using logistic regression classifier gave the the following results: 
* **f-score** : 0.79
* **accuracy** : 0.83




```python

```
