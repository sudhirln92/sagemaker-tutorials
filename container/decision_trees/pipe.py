from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from ml import custom as ml_custom
from ml import fillna as ml_fillna
import category_encoders as ce


def ModelPipe():

    model = Pipeline(steps=[
        ('drop',ml_custom.ExtractColumn(columns=['Pclass', 'Sex', 'Age', 'SibSp',
           'Parch',  'Fare', 'Cabin', 'Embarked'])),
        ('na',ml_fillna.FillMissingValue(cat_cols=['Cabin','Embarked'],num_cols=['Age'])),
        ('ohe',ce.OneHotEncoder(cols=['Sex','Cabin','Embarked'])),
        ('model',DecisionTreeClassifier(max_depth=10))
    ])
    return model