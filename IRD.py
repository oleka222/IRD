# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:30:42 2021

@author: aleks
"""
import pandas as pd
import json
import numpy as np
with open('Wyscout/events_Germany.json') as f:
    train = json.load(f)
    
df = pd.DataFrame(train)
df = df[df['subEventName']=='Shot']


X_df = pd.DataFrame()
X_df["X"] = df.positions.apply(lambda cell: 100 - cell[0]['x'])
X_df["X"] = X_df["X"]*105/100
X_df["Y"] = df.positions.apply(lambda cell: abs(cell[0]['y']-50))
X_df["Y"] = X_df["Y"]*65/100
X_df["Distance"] = np.sqrt(X_df["X"]**2 + X_df["Y"]**2)


for i,shot in df.iterrows():
    X_df.at[i,'Goal'] = 0
    X_df.at[i, "Type"] = 0
    x=X_df.at[i,'X']
    y=X_df.at[i,'Y']
    for shottags in shot['tags']:
        if shottags['id']==101:
            X_df.at[i,'Goal'] = 1
        if shottags['id']==403:
            df.at[i, 'Type'] = 1
        if shottags["id"] == 401:
            df.at[i, 'Type'] = 2
        if shottags["id"] == 402:
            df.at[i, 'Type'] = 3
    a = np.arctan(7.32 * x /(x**2 + y**2 - (7.32/2)**2))
    if a<0:
        a=np.pi+a
    X_df.at[i,'Angle'] = a
X_df["Goal"].astype(int)
X_df["Type"].astype(int)


with open('Wyscout/players.json') as f:
    players = json.load(f)
    
playersDB = pd.DataFrame(players)
playersDB2 = pd.DataFrame()
playersDB2["playerId"] = playersDB["wyId"]
playersDB2["role"] = playersDB["role"]
playersDB2["foot"] = playersDB["foot"]
#Joining 2 dataframes
df = df.reset_index().merge(playersDB2, how = "inner", on = ["playerId"]).set_index("index")
df = df.sort_index()
for i, shot in df.iterrows():
    X_df.at[i, "Position"] = 1
    for postags in shot["role"]:
        if shot["role"]["code2"] == "MD":
            X_df.at[i, "Position"] = 2
        if shot["role"]["code2"] == "FW":
            X_df.at[i, "Position"] = 3        
df["foot"] = np.where(df["foot"] != "left", 1, 0)            
df["strong"] = np.where(df["Type"] == 1, 1, 0)
df["str_foot"] = np.where((((df["foot"] == 1) & (df["Type"] == 2)) | ((df["foot"] == 0) & (df["Type"] == 3))) , 2, 0)
df["str_foot_2"] = np.where((((df["foot"] == 0) & (df["Type"] == 2)) | ((df["foot"] == 1) & (df["Type"] == 3))) , 3, 0)
X_df["Type"] = df["strong"] + df["str_foot"] + df["str_foot_2"]