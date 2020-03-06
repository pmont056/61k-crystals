from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd
df_62k = pd.read_json('df_62k.json', orient='split')
df_62k = df_62k.loc[df_62k.Embarked.notna(), ['canonical_smiles', 'xyz_pbe_relaxed']]

X = df_62k.drop('canonical_smiles', axis='column')
column_trans = make_column_transformer(
    (OneHotEncoder(), ['canonical_smiles']),
    remainder='passthrough')

column_trans.fit_transform(X)

Y = df_62k.xyz_pbe_relaxed

ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df_62k[['canonical_smiles']])

