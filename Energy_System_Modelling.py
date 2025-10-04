import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
url = "https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/Power_system_data_DE.xlsx"

raw_data = pd.read_excel(url)

# print(raw_data.head(20))

from datetime import datetime
clustering_colors = {  -1: plt.get_cmap("Greys")(0.7) , # ; # noise 
                        0: plt.get_cmap("plasma")(0) ,
                        1: plt.get_cmap("plasma")(0.4) ,
                        2: plt.get_cmap("plasma")(0.8) ,
                        3: plt.get_cmap("plasma")(0.9) }

def get_colors(cluster_list):
    color_list = []
    for i in cluster_list:
        color_list.append(clustering_colors[i])
    return color_list  

raw_data.columns = ["utc_timestamp", "cet_timestamp", "demand", "wind", "offshore", "onshore", "solar"]
data = raw_data.drop(index=[0,1,2,3,4,5]).fillna(0)
for i, row in data.iterrows():
    data.at[i,"year"] = row["utc_timestamp"].year
    data.at[i,"month"] = row["utc_timestamp"].month
    data.at[i,"day"] = row["utc_timestamp"].day
    data.at[i,"date"] = row["utc_timestamp"].date()
    data.at[i,"time"] = row["utc_timestamp"].time()
    
data_2018 = data.loc[data["year"]==2018]

data_2018_pivot = data_2018.pivot(index=["date"], columns=["time"], values=["wind","solar","demand"])

import matplotlib.pyplot as plt
import numpy as np
f, (ax1, ax2, ax3) = plt.subplots(1,3)
for i, row in data_2018_pivot.iterrows():
    ax1.plot(row["wind"].values, lw=0.2, alpha=0.5, color=plt.get_cmap("Blues")(np.random.rand(1)))  
    ax2.plot(row["solar"].values, lw=0.2, alpha=0.5, color=plt.get_cmap("YlOrBr")(np.random.rand(1)*0.5))
    ax3.plot(row["demand"].values, lw=0.2, alpha=0.5, color=plt.get_cmap("Greys")(np.random.rand(1)))
    
ax1.set_xticks([0, 23, 47, 71, 95], labels=["0:00", "6:00", "12:00", "18:00", "24:00"])
ax2.set_xticks([0, 23, 47, 71, 95], labels=["0:00", "6:00", "12:00", "18:00", "24:00"])
ax3.set_xticks([0, 23, 47, 71, 95], labels=["0:00", "6:00", "12:00", "18:00", "24:00"])

ax1.set_ylim([0,1])
ax2.set_ylim([0,1])

ax1.set_xlim([0,95])
ax2.set_xlim([0,95])
ax3.set_xlim([0,95])

ax1.set_ylabel("wind capacity factor")
ax2.set_ylabel("solar capacity factor")
ax3.set_ylabel("power demand / MW")

plt.subplots_adjust(wspace=0.4)

f.set_size_inches([10,4])
wind_data = data_2018[["date", "utc_timestamp","wind"]].melt(id_vars="date", value_vars="wind").drop("variable", axis=1)
solar_data = data_2018[["date", "utc_timestamp","solar"]].melt(id_vars="date", value_vars="solar").drop("variable", axis=1)
demand_data = data_2018[["date", "utc_timestamp","demand"]].melt(id_vars="date", value_vars="demand").drop("variable", axis=1)



from sklearn.preprocessing import StandardScaler

wind_scaler = StandardScaler()
wind_scaled = pd.DataFrame(columns=["date","value"])
wind_scaled["date"] = wind_data["date"]
wind_scaled["value"] = wind_scaler.fit_transform(np.array(wind_data["value"]).reshape(-1, 1))

solar_scaler = StandardScaler()
solar_scaled = pd.DataFrame(columns=["date","value"])
solar_scaled["date"] = solar_data["date"]
solar_scaled["value"] = solar_scaler.fit_transform(np.array(solar_data["value"]).reshape(-1, 1))

demand_scaler = StandardScaler()
demand_scaled = pd.DataFrame(columns=["date","value"])
demand_scaled["date"] = demand_data["date"]
demand_scaled["value"] = demand_scaler.fit_transform(np.array(demand_data["value"]).reshape(-1, 1))

input_data = pd.DataFrame() 

for day in wind_data["date"].drop_duplicates():
    wind = wind_scaled.loc[wind_scaled["date"]==day]["value"].values
    solar = solar_scaled.loc[solar_scaled["date"]==day]["value"].values
    demand = demand_scaled.loc[demand_scaled["date"]==day]["value"].values
    
    day_series = np.concatenate([wind, solar, demand])
    
    input_data[day] = day_series 
input_array = np.array(input_data).T

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0, n_init=10).fit(input_array)
profiles_labels = kmeans.labels_
profiles_clusters = kmeans.cluster_centers_

clusters_plot = [3,6,9]

f, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7, ax8, ax9)) = plt.subplots(3,3)

for i in range(0,364):
    if profiles_labels[i] == clusters_plot[0]:
        ax1.plot(data_2018_pivot.iloc[i]["wind"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("Blues")(np.random.rand(1)))  
        ax1.plot(wind_scaler.inverse_transform(profiles_clusters[clusters_plot[0]][0:96].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("Blues")(0.9))
        ax2.plot(data_2018_pivot.iloc[i]["solar"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("YlOrBr")(np.random.rand(1)*0.5))
        ax2.plot(solar_scaler.inverse_transform(profiles_clusters[clusters_plot[0]][96:192].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("YlOrBr")(0.5))
        ax3.plot(data_2018_pivot.iloc[i]["demand"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("Greys")(np.random.rand(1)))  
        ax3.plot(demand_scaler.inverse_transform(profiles_clusters[clusters_plot[0]][192:288].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("Greys")(0.9))
    if profiles_labels[i] == clusters_plot[1]:
        ax4.plot(data_2018_pivot.iloc[i]["wind"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("Blues")(np.random.rand(1)))  
        ax4.plot(wind_scaler.inverse_transform(profiles_clusters[clusters_plot[1]][0:96].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("Blues")(0.9))
        ax5.plot(data_2018_pivot.iloc[i]["solar"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("YlOrBr")(np.random.rand(1)*0.5))
        ax5.plot(solar_scaler.inverse_transform(profiles_clusters[clusters_plot[1]][96:192].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("YlOrBr")(0.5))
        ax6.plot(data_2018_pivot.iloc[i]["demand"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("Greys")(np.random.rand(1)))    
        ax6.plot(demand_scaler.inverse_transform(profiles_clusters[clusters_plot[1]][192:288].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("Greys")(0.9))
    if profiles_labels[i] == clusters_plot[2]:
        ax7.plot(data_2018_pivot.iloc[i]["wind"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("Blues")(np.random.rand(1)))  
        ax7.plot(wind_scaler.inverse_transform(profiles_clusters[clusters_plot[2]][0:96].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("Blues")(0.9))
        ax8.plot(data_2018_pivot.iloc[i]["solar"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("YlOrBr")(np.random.rand(1)*0.5))
        ax8.plot(solar_scaler.inverse_transform(profiles_clusters[clusters_plot[2]][96:192].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("YlOrBr")(0.5))
        ax9.plot(data_2018_pivot.iloc[i]["demand"].values, lw=0.5, alpha=0.5, color=plt.get_cmap("Greys")(np.random.rand(1)))    
        ax9.plot(demand_scaler.inverse_transform(profiles_clusters[clusters_plot[2]][192:288].reshape(-1, 1)), lw=1, alpha=1, color=plt.get_cmap("Greys")(0.9))

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xticks([0, 23, 47, 71, 95], labels=["0:00", "6:00", "12:00", "18:00", "24:00"])
    ax.set_xlim([0,95])

for ax in [ax1, ax2, ax4, ax5, ax7, ax8]:    
    ax.set_ylim([0,1])
    
for ax in [ax3, ax6, ax9]:    
    ax.set_ylim([data_2018_pivot["demand"].values.min(),data_2018_pivot["demand"].values.max()])

ax4.set_ylabel("wind capacity factor")
ax5.set_ylabel("solar capacity factor")
ax6.set_ylabel("power demand / MW")

ax2.set_title("cluster "+ str(clusters_plot[0]))
ax5.set_title("cluster "+ str(clusters_plot[1]))
ax8.set_title("cluster "+ str(clusters_plot[2]))

plt.subplots_adjust(wspace=0.4, hspace=0.3)

f.set_size_inches([10,10])
plt.show()



num_k = range(1,30)
errors = []

# This checks in error for different k values.
# The elbow method can be used to determine the optimal number of clusters by looking for a point where the rate of decrease sharply shifts (the "elbow").
# The method helps to balance the trade-off between having too few clusters (which may oversimplify the data) and too many clusters (which may overfit the data).
for k in num_k:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(input_array)
    errors.append(kmeans.inertia_)

f, ax = plt.subplots(1,1)
ax.plot(num_k, errors, marker="x" , ms = 10, color= get_colors([1][0]))
ax.set_xlabel("number of clusters k")
ax.set_ylabel("total distance")
plt.show()
