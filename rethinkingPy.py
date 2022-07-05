### Rethinking data ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
#import pymc3 as pm
import bambi
import pingouin as pg
import graphviz

dat_sim = pd.read_csv('C:/Users/daniel.feeney/OneDrive - Boa Technology Inc/Desktop/Rethinking Neuro Data Manuscript/Code/passDat.csv')

########### messy data ##########

##### CREATE MESSY DATASET #####
dat_mess = dat_sim.copy()

# subject 1 missing half the trials in both conditions
mask = dat_mess[(dat_mess["category"]=="seated") & (dat_mess["subj_id"]==1) & (dat_mess["item_id"]>1)].index
dat_mess = dat_mess.drop(mask)

mask = dat_mess[(dat_mess["category"]=="standing") & (dat_mess["subj_id"]==1) & (dat_mess["item_id"]>7)].index
dat_mess = dat_mess.drop(mask)

# subject 2 missing half the trials in both condition
mask = dat_mess[(dat_mess["category"]=="seated") & (dat_mess["subj_id"]==2) & (dat_mess["item_id"]>1)].index
dat_mess = dat_mess.drop(mask)

mask = dat_mess[(dat_mess["category"]=="standing") & (dat_mess["subj_id"]==2) & (dat_mess["item_id"]>7)].index
dat_mess = dat_mess.drop(mask)

# subject 3 missing all the trials in one condition
mask = dat_mess[(dat_mess["category"]=="standing") & (dat_mess["subj_id"]==3) & (dat_mess["item_id"]>7)].index
dat_mess = dat_mess.drop(mask)

# subject 16 missing all the trials in one condition
mask = dat_mess[(dat_mess["category"]=="standing") & (dat_mess["subj_id"]==16)].index
dat_mess = dat_mess.drop(mask)

##### CREATE UNBALANCED DATASET #####
dat_unbal = dat_mess.copy()

# remove subject 16
mask = dat_unbal[dat_unbal["subj_id"]==16].index
dat_unbal = dat_unbal.drop(mask)

dat_unbal.to_csv('C:/Users/daniel.feeney/OneDrive - Boa Technology Inc/Desktop/Rethinking Neuro Data Manuscript/Code/unbalDat.csv')
dat_mess.to_csv('C:/Users/daniel.feeney/OneDrive - Boa Technology Inc/Desktop/Rethinking Neuro Data Manuscript/Code/messDat.csv')
############ plotting ################
sns.displot(data=dat_sim, x="pwr", hue="category", col="subj_id",
            kind="kde", col_wrap=4, fill=True, alpha=0.5)



## modeling ##
model_partial = bambi.Model("pwr ~ category + (1 + category | subj_id)", data = dat_sim)
i_partial = model_partial.fit()
az.summary(i_partial)

# pooled #
model_pooled = bambi.Model("pwr ~ category", data = dat_sim)
i_pooled = model_pooled.fit()
az.summary(i_pooled)

### Complete pooling ###
model_unpooled = bambi.Model("pwr ~ (category | subj_id)", data = dat_sim)
i_unpooled = model_unpooled.fit()
az.summary(i_unpooled)

### comparing model fits ###
sns.relplot(data=dat_sim, x="category", y="pwr", hue="category", col="subj_id", 
                kind="scatter", col_wrap=4)
                
                
### ###
df = az.summary(i_pooled)

df_pooled = pd.DataFrame()
df_pooled["Subject"] = np.arange(1,17)
df_pooled["Intercept"] = [df["mean"]["Intercept"]] * 16
df_pooled["Slope_Cond"] = [df["mean"]["category[standing]"]] * 16
df_pooled["Model"] = ["Complete pooling"] * 16

df = az.summary(i_pooled)

df_pooled = pd.DataFrame()
df_pooled["Subject"] = np.arange(1,17)
df_pooled["Intercept"] = [df["mean"]["Intercept"]] * 16
df_pooled["Slope_Cond"] = [df["mean"]["category[standing]"]] * 16
df_pooled["Model"] = ["Complete pooling"] * 16

## comparing effect sizes ##
sns.set()

df = az.summary(i_unpooled)

mu_se = df["mean"][0] + df["mean"][2:18].values
mu_st = df["mean"][0] + df["mean"][2:18].values + df["mean"][19:35].values

est_slope = mu_st - mu_se

plt.scatter(x=mu_se, y=est_slope)

# print([np.mean(est_slope), np.std(est_slope)])
print("Unpooled model : Est. Effect size = ", np.mean(est_slope) / np.std(est_slope))

## means to posterior means ##
df = az.summary(i_pooled)

df_pooled = pd.DataFrame()
df_pooled["Subject"] = np.arange(1,17)
df_pooled["Intercept"] = [df["mean"]["Intercept"]] * 16
df_pooled["Slope_Cond"] = [df["mean"]["category[standing]"]] * 16
df_pooled["Model"] = ["Complete pooling"] * 16


## modeling messy data ##
model_pooled_messy = bambi.Model("pwr ~ category", data = dat_mess)
i_pooled_messy = model_pooled_messy.fit()
az.summary(i_pooled_messy)

# no pooling
model_unpooled_messy = bambi.Model("pwr ~ (category | subj_id)", data = dat_mess)
i_unpooled_messy = model_unpooled_messy.fit()
az.summary(i_unpooled_messy)

x = [12.72, -52.69, 49.87, 10.55, 8.23, 71.81, 45.52, 45.9, 58.73, 60.94, 60.95, 43.72, 104.29, -10.75, 68.88, -0.75]
print([np.mean(x), np.std(x)]) 

model_partial_messy = bambi.Model("pwr ~ category + (1 + category | subj_id)", data = dat_mess)
i_partial_messy = model_partial_messy.fit()
az.summary(i_partial_messy)

## visualizing uncertainty and predictions ##
df = az.summary(i_partial_messy)

sns.set()
sns.set_palette("colorblind")

fig, ax = plt.subplots(figsize=(8,5))

mu_subj = df["mean"][0] + df["mean"][1] + df["mean"][3:19].values + df["mean"][20:36].values

sd_subj = df["sd"][20:36].values
k = mu_subj.argsort()
x = np.arange(1,17)

plt.errorbar(x, mu_subj[k], sd_subj[k], ls="none", capsize=3, color="grey", label="Posterior ($\pm$SD)")

# mu_subj = df["mean"][0] + df["mean"][3:19].values
# sd_subj = df["sd"][3:19].values
# plt.errorbar(x, mu_subj[k], sd_subj[k], ls="none", capsize=3, color="r")

observed = dat_mess[dat_mess["category"]=="standing"].groupby("subj_id")["pwr"].mean().values
observed = np.insert(observed, [15], np.nan)
plt.scatter(x, observed[k], c="b", label="Observed")

# observed2 = dat_mess[dat_mess["category"]=="seated"].groupby("subj_id")["pwr"].mean().values
# plt.scatter(x, observed2[k], c="r", label="Observed seated")

ax.set_xticklabels(x[k]);
ax.set_xticks(x);

plt.xlabel("Subject ID")
plt.ylabel("Maximal 30 s Power Output (W)")
plt.title("Observations vs. posterior estimates for standing posture")
plt.legend()
