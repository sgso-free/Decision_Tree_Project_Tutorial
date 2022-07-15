
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv', sep=",")

#if value =0 and diabetic, set the mean of diabetic, else if value=0 and not diabetic, set the mean of not diabetic, 
# rest of the case use the data value

def set_value (data_value, outcome_value,mean_nodiab,mean_diab):
    if (outcome_value == 0 and data_value==0):
        return mean_nodiab
    elif (outcome_value ==1 and data_value ==0):
        return mean_diab
    else:
        return data_value

def set_use_mean (name_col):
    #calc the mean for diabetic and not diabetic that the data is not 0
    meanNoDiab = df_raw[(df_raw[name_col]>0) & (df_raw['Outcome']==0)][name_col].mean()
    meanDiab = df_raw[(df_raw[name_col]>0) & (df_raw['Outcome']==1)][name_col].mean()

    df_raw[name_col] = df_raw.apply(lambda x: set_value(x[name_col], x['Outcome'],meanNoDiab,meanDiab), axis=1)

set_use_mean('Insulin')
set_use_mean('Glucose')
set_use_mean('BloodPressure')
set_use_mean('SkinThickness')
set_use_mean('BMI')

#exclude pregnancy and BloodPressure to feature model
 
#X = df_raw[list(df_raw.columns[1:8])]
X = df_raw[['Glucose','SkinThickness','Insulin','BMI','Age','DiabetesPedigreeFunction']] 
y = df_raw[['Outcome']]
 
#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=34)

#create a tree model, with basic parameter. max-depth=5
tree_model = DecisionTreeClassifier(criterion='entropy',
                            min_samples_split=20,
                            min_samples_leaf=5,
                            max_depth = 5,
                            random_state=0)

tree_model.fit(X_train, y_train)
print('Accuracy:',tree_model.score(X_test, y_test))

print(f'Score:{tree_model.score(X_test, y_test)}')

# tree.feature_importances_ es un vector con la importancia estimada de cada atributo
for name, importance in zip(X.columns[0:], tree_model.feature_importances_):
    print(name + ': ' + str(importance))


#save the model to file
filename = 'models/finalized_model.sav' #use absolute path
pickle.dump(tree_model, open(filename, 'wb'))

#use the model save with new data to predicts prima

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

#Predict using the model 
Glucose=120
SkinThickness=23
Insulin=215
BMI=29
Age=24
DiabetesPedigreeFunction=0.520

#predigo el target para los valores seteados
print('Predicted Diabetic : \n', loaded_model.predict([[Glucose,SkinThickness,Insulin,BMI,Age,DiabetesPedigreeFunction]]))

Glucose=134
SkinThickness=30
Insulin=74
BMI=34
Age=24
DiabetesPedigreeFunction=0.75

#predigo el target para los valores seteados
print('Predicted Diabetic : \n', loaded_model.predict([[Glucose,SkinThickness,Insulin,BMI,Age,DiabetesPedigreeFunction]]))