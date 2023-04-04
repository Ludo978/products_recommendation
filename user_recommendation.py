import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pick import pick

print("Loading dataframes...")
df = pd.DataFrame(pd.read_csv('dataframes/KaDo.csv'))
df_items = pd.DataFrame(pd.read_csv('dataframes/items.csv', index_col='CLI_ID'))
data_k3 = pd.DataFrame(pd.read_csv('dataframes/data_k3.csv', index_col='CLI_ID'))
clients = pd.DataFrame(pd.read_csv('dataframes/clients.csv', index_col='CLI_ID'))
print("Dataframes loaded")

while True:
  client_id = int(input("Client ID: "))

  client = df[df['CLI_ID'] == client_id]

  if client.empty:
    print("Invalid client ID")
  else: 
    family = client['FAMILLE'].value_counts().sort_values(ascending=False).index[0]
    maille = client['MAILLE'].value_counts().sort_values(ascending=False).index[0]
    univers = client['UNIVERS'].value_counts().sort_values(ascending=False).index[0]
    item = client['LIBELLE'].value_counts().sort_values(ascending=False).index[0]
    price = client.agg({
        'PRIX_NET': ['mean', 'median', 'std', 'min', 'max'],
    })
    concat = pd.concat([
      price.rename(columns={"PRIX_NET": client_id}), 
      pd.DataFrame(clients.loc[client_id]), 
      pd.DataFrame(data_k3.loc[client_id].drop(['MonetaryValue', 'Frequency']))
    ]).to_string(header=False)

    print('Favorite family: ', family)
    print('Favorite maille: ', maille)
    print('Favorite universe: ', univers)
    print('Favorite item: ', item)
    print(concat)
    print('\n')

    title = 'Recommender type: '
    options = ['Simple', 'Complex']
    option, index = pick(options, title)

    if index == 1:
      recommendations = df_items.corrwith(df_items[item])
      recommendations.dropna(inplace=True)
      recommendations = pd.DataFrame(recommendations, columns=['correlation']).reset_index().sort_values(by='correlation', ascending=False).iloc[1:11 , :].to_string(header=False, index=False)
      print(recommendations)
    else:
      top_item_families = df.groupby(['FAMILLE', 'LIBELLE']).agg(
          Quantity=('TICKET_ID', 'nunique')
      ).reset_index().sort_values(by='Quantity', ascending=False).drop_duplicates('FAMILLE').set_index('FAMILLE')
      top_item_mailles = df.groupby(['MAILLE', 'LIBELLE']).agg(
          Quantity=('TICKET_ID', 'nunique')
      ).reset_index().sort_values(by='Quantity', ascending=False).drop_duplicates('MAILLE').set_index('MAILLE')
      top_item_universes = df.groupby(['UNIVERS', 'LIBELLE']).agg(
          Quantity=('TICKET_ID', 'nunique')
      ).reset_index().sort_values(by='Quantity', ascending=False).drop_duplicates('UNIVERS').set_index('UNIVERS')

      print('Based on family: ', top_item_families.loc[family][0])
      print('Based on maille: ', top_item_mailles.loc[maille][0])
      print('Based on universe: ', top_item_universes.loc[univers][0])

    print('\n')