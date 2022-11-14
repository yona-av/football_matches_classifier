# La- Liga win classifier code


```python

# import modules 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys
import seaborn as sns
import scipy as sp
%matplotlib inline

# import pre-processing modules 
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel 

# import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# import evaluation modules
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
```


```python
df_original = pd.read_csv(r"C:/Users/Yehonatan/PycharmProject/DS/projects/la_liga_project/to ignore/la_liga_data.csv")
```

## Define custom functions that will be used in the code 


```python
# create a function that replaces nans with the mean of the column 

def deal_nans(df, percantage_nan):
    nans = df[df.columns[df.isnull().any()]]
    nan_col_names = list(nans)
    for i in nan_col_names:
        count_nan = df[i].isnull().sum()
        if count_nan/len(df[i]) >= 0.1:       # if nans are more than 10% delete the column 
            df = df.drop(columns=[i])
        else:
            col_avg = df[i].mean()
            df[i] = df[i].fillna(col_avg)
    return df 


# create a function that gets the features and target . calcs the p-values and if it rejects or accepts H_0 null hypothesis 

def pvalue_filter(target, features, alpha): # returns a list of columns that are possible drop, p_val > alpha, corr, pval
    features_columns_names = list(features)
    target_column_name = list(target)
    features_np = features.to_numpy()
    target_np = target.to_numpy()
    drop_index = []
    p_val_list =[]
    corr_list = []
    
    for i in range(len(features_columns_names)):
        corr, p_val = sp.stats.pearsonr(features_np[:,i], target_np)
        corr_list.append(round(corr,3))
        if p_val > alpha:           # accept the null hypothesis, no statisitcal significance  
            drop_index.append(i)
            p_val_list.append(p_val)
            
    drop_col = [features_columns_names[i] for i in drop_index]
    return drop_col, corr_list, p_val_list

# create a function that returns the selected and rejected columns by correlation between features 
def features_corr_filter(features, corr_cutoff): 
    corr = features.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool) # create boolean filter 

    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= corr_cutoff:
                columns[j] = False

    rejected_columns = x_train.columns[np.invert(columns)]
    #print('reject',rejected_columns)
    selected_columns = x_train.columns[columns]
    #print('selected',selected_columns)
    return selected_columns, rejected_columns

```

## EDA and data processing


```python

df = df_original.copy()

df = df.drop(columns = ['start_time', 'round', 'dayofweek', 'opponent']) #drop categorical columns that dont contribute 
df['venue'] = df['venue'].apply(lambda x : 1 if x == 'Home' else 0) # convert the feature to numerical categorical 

# for now we turn this classifier to be a WIN classifier so Ovr strategy
df['result'] = df['result'].apply(lambda x : 1 if x == 'W' else 0) 

#turn df into float 64 
df=df.astype('float64')

# deal with NAN values 
# specific columns to deal with

df['gk_save_pct'] = df['gk_save_pct'].fillna(100) # no shots on target means no saves in a way same effect as 100% saves
f = df.replace({'gk_save_pct': { -100 : 0}}) #replaced an outlier of -100 that should have been 0 
df['own_goals'] = df['own_goals'].fillna(0) # safe to assume that if there is a NAN there were no owngoals as it is a rare occasion 
df = df.drop(columns=['tackles_interceptions', 'Unnamed: 0']) # all Nan_s in this column

deal_nans(df, 0.1) # fill columns with more then 10% nans with the column's mean value 

# divide to x,y sets 
y = df['result']
x = df.drop(['result'], axis=1)

# divide into test set and train set   
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)

train_set = pd.concat([x_train,y_train],axis=1)

```

## Feature selection / elimination
### STEP 1
### Drop the statisticly insignificant features


```python
drop_col, corr_list, p_val_list = pvalue_filter(y_train, x_train, 0.05)
x_train = x_train.drop(drop_col, axis=1)
x_train.rename(columns = {'passes.1':'attempted_passes'}, inplace=True)

```

### STEP 2 
### Drop hand-picked features byt he logic in the readme.file(need to link it !!!)
### explaine to be added to readme note 
drop features that might indicate win or loss. for instance assists (= goals) it is quite obvious that many assists 
will most likely lead to win as the team scored many goals. 
I left features that on a first glance might not indicate a win like - gk_psxg   
As well I left features that are controllable - meaning features that can be improved and controlled to some degree by the coaching team


```python
hand_pick_drop = ['gk_clean_sheets','gk_goals_against', 'goals_for','goals_against','gk_psxg_net', 'dribbles_vs','dribbles_completed',
                  'passes_received_pct', 'assists', 'throw_ins','passes_left_foot', 'passes_right_foot',
                  'sca_shots','passes_head','gca','gca_passes_live', 'gca_passes_dead', 'gca_dribbles','gca_shots', 'gca_fouled', 
                  'gca_defense','goals','goals_per_shot', 'goals_per_shot_on_target','xg', 'npxg', 'npxg_per_shot', 
                  'pens_made', 'pens_att','xg_net', 'npxg_net','own_goals']


x_train = x_train.drop(hand_pick_drop, axis=1)
```

### STEP 3
### Drop highly correlated features by applying correlation filter on x_train


```python
selected_columns, rejected_columns = features_corr_filter(x_train, 0.9)
x_train = x_train[selected_columns]
x_test = x_test[selected_columns]
```


```python
# plot heatmap of top 4 positive and negative correlations
#train_set_corr = train_set.corr()
#train_set_target = train_set_corr['result'].sort_values(ascending=False)
#top_4_corr = pd.concat([train_set_target.iloc[0:5], train_set_target.iloc[-4:]]).index.tolist() # top 4 positive and negative correlations to target

#corr_plot_df = train_set[top_4_corr]
#correlation_heatmap(corr_plot_df)
```


```python
#### after all feature selection filtering
### now x_train has 84 features (half compared to the beginning) #### pvalue drop has been made and high corr between features 
```

## Data scaling 
Scale the data using standardscaler in order to prepare it for the classification algorithms 


```python
scaler = StandardScaler()
x_train_np = x_train.to_numpy()   
x_train_np = scaler.fit_transform(x_train_np)

x_train_df = pd.DataFrame(x_train_np, columns= list(selected_columns)) # df after scaling
y_train_np = y_train.to_numpy()

x_test_np = x_test.to_numpy()
x_test_np = scaler.transform(x_test_np)

y_test_np =  y_test.to_numpy()
```


```python
#plot histogram of top 4 positive and negative correlation to target 
#x_train_df[top_4_corr[1:]].hist(bins=50, figsize=(20,15))
#plt.show()
```


```python
#### apply classifiers ####

# tried three classifiers - ligostic reg gave slightly better rsults . yet it seems to not overfit as the others 
#and when I chose newton method the results were great

```

## Apply classifiers


```python
clf = RandomForestClassifier(n_estimators= 100, max_depth=2, random_state=9)
clf.fit(x_train_np, y_train_np)
y_train_pred = clf.predict(x_train_np)
```


```python
clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(x_train_np, y_train_np)
y_train_pred = clf.predict(x_train_np)
```


```python
# newton method as lbfgs didnt converge and it gave the best outcome . could check other algos 
clf = LogisticRegression(random_state=9, max_iter=500, solver = 'newton-cg') 
clf.fit(x_train_np, y_train_np)
y_train_pred = clf.predict(x_train_np)
```

### performance measurment of the training data 


```python
#### check the accuracy, precision and recall ###### all feature slection is applied

precision, recall, f_1, support = precision_recall_fscore_support(y_train_np, y_train_pred, average='binary')
accuracy = accuracy_score(y_train_np, y_train_pred)
print('accuracy :', np.round(accuracy,3))
print('precision :', np.round(precision,3))
print('recall :', np.round(recall,3))
print('f_score :', np.round(f_1,3))

```

    accuracy : 0.849
    precision : 0.823
    recall : 0.817
    f_score : 0.82
    

### cross-validation on the training data 


```python
cv_3_accuracy = cross_val_score(clf, x_train , y_train, cv=3, scoring='accuracy')
#cv_3_precision = cross_val_score(clf, x_train , y_train, cv=3, scoring='precision')
#cv_3_recall = cross_val_score(clf, x_train , y_train, cv=3, scoring='recall')

print('cv_avg_accuracy :', np.round(cv_3_accuracy.mean(),3))
#print('cv_avg_precision', cv_3_precision.mean())
#print('cv_avg_recall', cv_3_recall.mean())
```

    cv_avg_accuracy : 0.82
    

### Grid Search for logistic regression


```python

params_l1 = {
    'penalty' : ['l1'],
    'solver' : ['liblinear', 'saga'],
    'max_iter' : [200,500,750,1000]
}

params_l2 = {
    'penalty' : ['l2'],
    'solver' : ['newton-cg', 'saga', 'sag' ],
    'max_iter' : [200,500,750,1000]
}

grid_search_l1 = GridSearchCV(estimator = clf, param_grid = params_l1, scoring = 'f1', cv = 3, verbose = 0)
grid_search_l2 = GridSearchCV(estimator = clf, param_grid = params_l2, scoring = 'f1', cv = 3, verbose = 0)
```


```python
grid_search_l1.fit(x_train_np, y_train_np)
```




<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=LogisticRegression(max_iter=500, random_state=9,
                                          solver=&#x27;newton-cg&#x27;),
             param_grid={&#x27;max_iter&#x27;: [200, 500, 750, 1000], &#x27;penalty&#x27;: [&#x27;l1&#x27;],
                         &#x27;solver&#x27;: [&#x27;liblinear&#x27;, &#x27;saga&#x27;]},
             scoring=&#x27;f1&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=LogisticRegression(max_iter=500, random_state=9,
                                          solver=&#x27;newton-cg&#x27;),
             param_grid={&#x27;max_iter&#x27;: [200, 500, 750, 1000], &#x27;penalty&#x27;: [&#x27;l1&#x27;],
                         &#x27;solver&#x27;: [&#x27;liblinear&#x27;, &#x27;saga&#x27;]},
             scoring=&#x27;f1&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500, random_state=9, solver=&#x27;newton-cg&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500, random_state=9, solver=&#x27;newton-cg&#x27;)</pre></div></div></div></div></div></div></div></div></div></div>




```python
print('best parameters : ', grid_search_l1.best_params_)
print('best score for the parameters :', np.round(grid_search_l1.best_score_,3))
```

    best parameters :  {'max_iter': 200, 'penalty': 'l1', 'solver': 'liblinear'}
    best score for the parameters : 0.788
    


```python
grid_search_l2.fit(x_train_np, y_train_np)
```




<style>#sk-container-id-8 {color: black;background-color: white;}#sk-container-id-8 pre{padding: 0;}#sk-container-id-8 div.sk-toggleable {background-color: white;}#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-8 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-8 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-8 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-8 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-8 div.sk-item {position: relative;z-index: 1;}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-8 div.sk-item::before, #sk-container-id-8 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-8 div.sk-label-container {text-align: center;}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-8 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=LogisticRegression(max_iter=500, random_state=9,
                                          solver=&#x27;newton-cg&#x27;),
             param_grid={&#x27;max_iter&#x27;: [200, 500, 750, 1000], &#x27;penalty&#x27;: [&#x27;l2&#x27;],
                         &#x27;solver&#x27;: [&#x27;newton-cg&#x27;, &#x27;saga&#x27;, &#x27;sag&#x27;]},
             scoring=&#x27;f1&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=LogisticRegression(max_iter=500, random_state=9,
                                          solver=&#x27;newton-cg&#x27;),
             param_grid={&#x27;max_iter&#x27;: [200, 500, 750, 1000], &#x27;penalty&#x27;: [&#x27;l2&#x27;],
                         &#x27;solver&#x27;: [&#x27;newton-cg&#x27;, &#x27;saga&#x27;, &#x27;sag&#x27;]},
             scoring=&#x27;f1&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500, random_state=9, solver=&#x27;newton-cg&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500, random_state=9, solver=&#x27;newton-cg&#x27;)</pre></div></div></div></div></div></div></div></div></div></div>




```python
print('best parameters : ', grid_search_l2.best_params_)
print('best score for the parameters :', np.round(grid_search_l2.best_score_,3))
```

    best parameters :  {'max_iter': 200, 'penalty': 'l2', 'solver': 'newton-cg'}
    best score for the parameters : 0.782
    


```python
#clf.get_params()
```

### classification report (training data)


```python
print(classification_report(y_train_np, y_train_pred))  
```

                  precision    recall  f1-score   support
    
             0.0       0.87      0.87      0.87      1147
             1.0       0.82      0.82      0.82       829
    
        accuracy                           0.85      1976
       macro avg       0.85      0.84      0.85      1976
    weighted avg       0.85      0.85      0.85      1976
    
    


```python

```

## Prediction part


```python
y_test_pred = clf.predict(x_test_np)
```

### performance measurment of the test data 


```python
precision, recall, f_1, support = precision_recall_fscore_support(y_test_np, y_test_pred, average='binary')
accuracy = accuracy_score(y_test_np, y_test_pred)
print('accuracy :', np.round(accuracy,3))
print('precision :', np.round(precision,3))
print('recall :', np.round(recall,3))
print('f_score :', np.round(f_1,3))
```

    accuracy : 0.826
    precision : 0.813
    recall : 0.767
    f_score : 0.789
    

### classification report (test data)


```python
print(classification_report(y_test_np, y_test_pred))  
```

                  precision    recall  f1-score   support
    
             0.0       0.83      0.87      0.85       284
             1.0       0.81      0.77      0.79       210
    
        accuracy                           0.83       494
       macro avg       0.82      0.82      0.82       494
    weighted avg       0.83      0.83      0.83       494
    
    


```python
#x_train_df_scale.hist(bins=50, figsize=(20,15))
#plt.show()
```


```python
#x_train['passes_switches'].hist(bins=50, figsize=(20,15))
#plt.show()
```


```python
#x_train.nutmegs.value_counts() 
#x_train['gk_clean_sheets']
```


```python

```


```python
#### general NOTES ###
# few attributes were removed at fbref so they exist in my data but I dont have the exact definition of them 
```


```python

```


```python

```


```python

```


```python

```
