import os 
import pygmt 
import pandas as pd 
from dataprep import clean 


# plot a map of stations 
FIGUREDIR = "/home/timclements/CS249FINAL/FIGURES/"
csvfile = "/home/timclements/CS249FINAL/data/merge.csv"
df = pd.read_csv(csvfile)
NOISEdf = df[df['trace_category'] == "noise"]
EQdf = df[df['trace_category'] == "earthquake_local"]
EQdf = clean(EQdf)
EQdf = EQdf.drop_duplicates(subset=["network_code","receiver_code"])
NOISEdf = NOISEdf.drop_duplicates(subset=["network_code","receiver_code"])
NOISEdf = NOISEdf[~NOISEdf.receiver_code.isin(EQdf.receiver_code)]

# create directory for figues 
if not os.path.isdir(FIGUREDIR): 
    os.mkdir(FIGUREDIR)

# create station map 
fig = pygmt.Figure()
# Use region "d" to specify global region (-180/180/-90/90)
fig.coast(region="d", frame="afg", land="gray", projection="N12c")
fig.plot(x=NOISEdf.receiver_longitude, y=NOISEdf.receiver_latitude, style="i0.09c", color="red", pen="black",t=15)
fig.plot(x=EQdf.receiver_longitude, y=EQdf.receiver_latitude, style="i0.09c", color="dodgerblue", pen="black",t=15)
fig.close()