#version modules
#dowhy==0.11.1
#econml==0.15.1


# importing required libraries
# importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
np.int = np.int32
from econml.dml import DML, SparseLinearDML
from econml.dr import SparseLinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_split
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
import scipy.stats as stats
from zepid.graphics import EffectMeasurePlot
from itertools import product
from econml.utilities import WeightedModelWrapper
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text, geom_errorbarh, geom_vline, theme_bw, element_blank
from xgboost import XGBRegressor, XGBClassifier


# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)




#%%#
#conditionated by population density

# Set display options for pandas to show 5 decimal places
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))

# Create data frame of ATE results
df_ATE = pd.DataFrame(0.0, index=range(0, 3), columns=['ATE', '95% CI']).astype({'ATE': 'float64'})

# Convert the second column to tuples with 5 decimal places
df_ATE['95% CI'] = [((0.0, 0.0)) for _ in range(3)]  # Using list comprehension to create tuples


# Display the DataFrame
print(df_ATE)

#%%#
#import data
data_all = pd.read_csv("D:/data_final.csv", encoding='latin-1') 

#subset to 95th percentile
data = data_all[(data_all['Pop_density'] <= 385.45)]


# Calculate z-score
data.SST3 = stats.zscore(data.SST3, nan_policy='omit') 
data.SST34 = stats.zscore(data.SST34, nan_policy='omit') 
data.SST4 = stats.zscore(data.SST4, nan_policy='omit') 
data.SOI = stats.zscore(data.SOI, nan_policy='omit')
data.NATL = stats.zscore(data.NATL, nan_policy='omit')
data.TROP = stats.zscore(data.TROP, nan_policy='omit')
data.Pop_density = stats.zscore(data.Pop_density, nan_policy='omit')


# ML method
reg = lambda: XGBClassifier(n_estimators=2500, random_state=123) 

## Ignore warnings
warnings.filterwarnings('ignore') 


#%%#
#consensus4                           
#NeutralvsLa_Nina
data_NeutralvsLa_Nina = data[['excess', 'NeutralvsLa_Nina_consensus4', 'SST3', 'SST4', 'SST34',
                                         'SOI', 'NATL', 'TROP', 'Pop_density']] 

data_NeutralvsLa_Nina = data_NeutralvsLa_Nina.dropna()

Y = data_NeutralvsLa_Nina.excess.to_numpy()
T = data_NeutralvsLa_Nina.NeutralvsLa_Nina_consensus4.to_numpy()
W = data_NeutralvsLa_Nina[['SST3', 'SST4', 'SST34',
                                      'SOI', 'NATL', 'TROP']].to_numpy().reshape(-1, 6)
X = data_NeutralvsLa_Nina[['Pop_density']].to_numpy()

#Estimation of the effect 
estimate_NeutralvsLa_Nina = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False),   
                                                   cv=5, random_state=123)

estimate_NeutralvsLa_Nina = estimate_NeutralvsLa_Nina.dowhy

# fit the model
estimate_NeutralvsLa_Nina.fit(Y=Y, T=T, X=X, W=W, inference='auto') 

# predict effect for each sample X
estimate_NeutralvsLa_Nina.effect(X)

# ATE
ate_NeutralvsLa_Nina = estimate_NeutralvsLa_Nina.ate(X) 
print(ate_NeutralvsLa_Nina) 

# confidence interval of ate
ci_NeutralvsLa_Nina = estimate_NeutralvsLa_Nina.ate_interval(X) 
print(ci_NeutralvsLa_Nina) 

# Set values in the df_ATE
df_ATE.at[0, 'ATE'] = round(ate_NeutralvsLa_Nina, 5)
df_ATE.at[0, '95% CI'] = ci_NeutralvsLa_Nina
print(df_ATE)


#%%#

# constant marginal effect
#range of X
min_X = min(X)
max_X = max(X)
delta = (max_X - min_X) / 100
X_pop = np.arange(min_X, max_X + delta - 0.001, delta).reshape(-1, 1)

est2_NeutralvsLa_Nina = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False),   
                                                   cv=5, random_state=123)

est2_NeutralvsLa_Nina.fit(Y=Y, T=T, X=X, W=W, inference='auto')

treatment_cont_marg = est2_NeutralvsLa_Nina.const_marginal_effect(X_pop)
hte_lower2_cons, hte_upper2_cons = est2_NeutralvsLa_Nina.const_marginal_effect_interval(X_pop)

# Reshape X to 1-dimensional array
X_pop = X_pop.ravel()
treatment_cont_marg = treatment_cont_marg.ravel()

# Reshape te_lower2_cons and te_upper2_cons to 1-dimensional arrays
hte_lower2_cons = hte_lower2_cons.ravel()
hte_upper2_cons = hte_upper2_cons.ravel()

#Figure 4a
(
ggplot()
  + aes(x=X_pop, y=treatment_cont_marg)
  + geom_line()
  + geom_ribbon(aes(ymin = hte_lower2_cons, ymax = hte_upper2_cons), alpha=0.1)
  + labs(x='Population density (sd)', y='Effect of Neutral vs La Niña on excess dengue cases')
  + geom_hline(yintercept=0, color="red", linetype="dashed")
)

#%%#

# CATEs per municipality
estimate_NeutralvsLa_Nina.effect(X)

cates = estimate_NeutralvsLa_Nina.estimate_.cate_estimates
dataframe_cates = pd.DataFrame.from_records(cates)
dataframe_cates.rename(columns={dataframe_cates.columns[0]: 'CATEs'}, inplace=True)

data_period = data[['excess', 'NeutralvsLa_Nina_consensus4', 'SST3', 'SST4', 'SST34',
                    'SOI', 'NATL', 'TROP', 'Pop_density', 'DANE', 'DANE_Period']] 

data_period = data_period.dropna()

# Bind datasets
geo_NeutralvsLa_Nina = pd.concat([data_period.reset_index(drop=True), dataframe_cates], axis=1)

# Grouping the DataFrame by "code.DANE" and calculating the average of "ATEs"
NeutralvsLa_Nina_dane = geo_NeutralvsLa_Nina.groupby('DANE')['CATEs'].mean().reset_index()

# Save the data
NeutralvsLa_Nina_dane.to_csv('D:/Density_NeutralvsLa_Nina_dane.csv', index=False)



#%%#
#Refute tests
#with random common cause
random_NeutralvsLa_Nina = estimate_NeutralvsLa_Nina.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=20)
print(random_NeutralvsLa_Nina)

#with replace a random subset of the data
subset_NeutralvsLa_Nina = estimate_NeutralvsLa_Nina.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.10, num_simulations=20)
print(subset_NeutralvsLa_Nina)

#with replace a dummy outcome
dummy_NeutralvsLa_Nina = estimate_NeutralvsLa_Nina.refute_estimate(method_name="dummy_outcome_refuter", num_simulations=20)
print(dummy_NeutralvsLa_Nina[0])

#with placebo 
placebo_NeutralvsLa_Nina = estimate_NeutralvsLa_Nina.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=20)
print(placebo_NeutralvsLa_Nina)


#%%#
#consensus4                           
#NeutralvsEl_Nino
data_NeutralvsEl_Nino = data[['excess', 'NeutralvsEl_Nino_consensus4', 'SST3', 'SST4', 'SST34',
                                         'SOI', 'NATL', 'TROP', 'Pop_density']] 

data_NeutralvsEl_Nino = data_NeutralvsEl_Nino.dropna()

Y = data_NeutralvsEl_Nino.excess.to_numpy()
T = data_NeutralvsEl_Nino.NeutralvsEl_Nino_consensus4.to_numpy()
W = data_NeutralvsEl_Nino[['SST3', 'SST4', 'SST34',
                                      'SOI', 'NATL', 'TROP']].to_numpy().reshape(-1, 6)
X = data_NeutralvsEl_Nino[['Pop_density']].to_numpy()

#Estimation of the effect 
estimate_NeutralvsEl_Nino = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False),   
                                                   cv=5, random_state=123)

estimate_NeutralvsEl_Nino = estimate_NeutralvsEl_Nino.dowhy

# fit the model
estimate_NeutralvsEl_Nino.fit(Y=Y, T=T, X=X, W=W, inference='auto') 

# predict effect for each sample X
estimate_NeutralvsEl_Nino.effect(X)

# ATE
ate_NeutralvsEl_Nino = estimate_NeutralvsEl_Nino.ate(X) 
print(ate_NeutralvsEl_Nino) 

# confidence interval of ate
ci_NeutralvsEl_Nino = estimate_NeutralvsEl_Nino.ate_interval(X) 
print(ci_NeutralvsEl_Nino) 

# Set values in the df_ATE
df_ATE.at[1, 'ATE'] = round(ate_NeutralvsEl_Nino, 5)
df_ATE.at[1, '95% CI'] = ci_NeutralvsEl_Nino
print(df_ATE)


#%%#

# constant marginal effect
#range of X
min_X = min(X)
max_X = max(X)
delta = (max_X - min_X) / 100
X_pop = np.arange(min_X, max_X + delta - 0.001, delta).reshape(-1, 1)

est2_NeutralvsEl_Nino = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False),   
                                                   cv=5, random_state=123)

est2_NeutralvsEl_Nino.fit(Y=Y, T=T, X=X, W=W, inference='auto')

treatment_cont_marg = est2_NeutralvsEl_Nino.const_marginal_effect(X_pop)
hte_lower2_cons, hte_upper2_cons = est2_NeutralvsEl_Nino.const_marginal_effect_interval(X_pop)

# Reshape X to 1-dimensional array
X_pop = X_pop.ravel()
treatment_cont_marg = treatment_cont_marg.ravel()

# Reshape te_lower2_cons and te_upper2_cons to 1-dimensional arrays
hte_lower2_cons = hte_lower2_cons.ravel()
hte_upper2_cons = hte_upper2_cons.ravel()

#Figure 4b
(
ggplot()
  + aes(x=X_pop, y=treatment_cont_marg)
  + geom_line()
  + geom_ribbon(aes(ymin = hte_lower2_cons, ymax = hte_upper2_cons), alpha=0.1)
  + labs(x='Population density (sd)', y='Effect of Neutral vs El Niño on excess dengue cases')
  + geom_hline(yintercept=0, color="red", linetype="dashed")
)

#%%#

# CATEs per municipality
estimate_NeutralvsEl_Nino.effect(X)

cates = estimate_NeutralvsEl_Nino.estimate_.cate_estimates
dataframe_cates = pd.DataFrame.from_records(cates)
dataframe_cates.rename(columns={dataframe_cates.columns[0]: 'CATEs'}, inplace=True)

data_period = data[['excess', 'NeutralvsEl_Nino_consensus4', 'SST3', 'SST4', 'SST34',
                    'SOI', 'NATL', 'TROP', 'Pop_density', 'DANE', 'DANE_Period']] 

data_period = data_period.dropna()

# Bind datasets
geo_NeutralvsEl_Nino = pd.concat([data_period.reset_index(drop=True), dataframe_cates], axis=1)

# Grouping the DataFrame by "code.DANE" and calculating the average of "ATEs"
NeutralvsEl_Nino_dane = geo_NeutralvsEl_Nino.groupby('DANE')['CATEs'].mean().reset_index()

# Save the data
NeutralvsEl_Nino_dane.to_csv('D:/Density_NeutralvsEl_Nino_dane.csv', index=False)



#%%#
#Refute tests
#with random common cause
random_NeutralvsEl_Nino = estimate_NeutralvsEl_Nino.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=20)
print(random_NeutralvsEl_Nino)

#with replace a random subset of the data
subset_NeutralvsEl_Nino = estimate_NeutralvsEl_Nino.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.10, num_simulations=20)
print(subset_NeutralvsEl_Nino)

#with replace a dummy outcome
dummy_NeutralvsEl_Nino = estimate_NeutralvsEl_Nino.refute_estimate(method_name="dummy_outcome_refuter", num_simulations=20)
print(dummy_NeutralvsEl_Nino[0])

#with placebo 
placebo_NeutralvsEl_Nino = estimate_NeutralvsEl_Nino.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=20)
print(placebo_NeutralvsEl_Nino)



#%%#
#consensus4                           
#La_NinavsEl_Nino
data_La_NinavsEl_Nino = data[['excess', 'La_NinavsEl_Nino_consensus4', 'SST3', 'SST4', 'SST34',
                                         'SOI', 'NATL', 'TROP', 'Pop_density']] 

data_La_NinavsEl_Nino = data_La_NinavsEl_Nino.dropna()

Y = data_La_NinavsEl_Nino.excess.to_numpy()
T = data_La_NinavsEl_Nino.La_NinavsEl_Nino_consensus4.to_numpy()
W = data_La_NinavsEl_Nino[['SST3', 'SST4', 'SST34',
                                      'SOI', 'NATL', 'TROP']].to_numpy().reshape(-1, 6)
X = data_La_NinavsEl_Nino[['Pop_density']].to_numpy()

#Estimation of the effect 
estimate_La_NinavsEl_Nino = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False),   
                                                   cv=5, random_state=123)

estimate_La_NinavsEl_Nino = estimate_La_NinavsEl_Nino.dowhy

# fit the model
estimate_La_NinavsEl_Nino.fit(Y=Y, T=T, X=X, W=W, inference='auto') 

# predict effect for each sample X
estimate_La_NinavsEl_Nino.effect(X)

# ATE
ate_La_NinavsEl_Nino = estimate_La_NinavsEl_Nino.ate(X) 
print(ate_La_NinavsEl_Nino) 

# confidence interval of ate
ci_La_NinavsEl_Nino = estimate_La_NinavsEl_Nino.ate_interval(X) 
print(ci_La_NinavsEl_Nino) 

# Set values in the df_ATE
df_ATE.at[2, 'ATE'] = round(ate_La_NinavsEl_Nino, 5)
df_ATE.at[2, '95% CI'] = ci_La_NinavsEl_Nino
print(df_ATE)


#%%#

# constant marginal effect
#range of X
min_X = min(X)
max_X = max(X)
delta = (max_X - min_X) / 100
X_pop = np.arange(min_X, max_X + delta - 0.001, delta).reshape(-1, 1)

est2_La_NinavsEl_Nino = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False),   
                                                   cv=5, random_state=123)

est2_La_NinavsEl_Nino.fit(Y=Y, T=T, X=X, W=W, inference='auto')

treatment_cont_marg = est2_La_NinavsEl_Nino.const_marginal_effect(X_pop)
hte_lower2_cons, hte_upper2_cons = est2_La_NinavsEl_Nino.const_marginal_effect_interval(X_pop)

# Reshape X to 1-dimensional array
X_pop = X_pop.ravel()
treatment_cont_marg = treatment_cont_marg.ravel()

# Reshape te_lower2_cons and te_upper2_cons to 1-dimensional arrays
hte_lower2_cons = hte_lower2_cons.ravel()
hte_upper2_cons = hte_upper2_cons.ravel()

#Figure 4c
(
ggplot()
  + aes(x=X_pop, y=treatment_cont_marg)
  + geom_line()
  + geom_ribbon(aes(ymin = hte_lower2_cons, ymax = hte_upper2_cons), alpha=0.1)
  + labs(x='Population density (sd)', y='Effect of La Niña vs El Niño on excess dengue cases')
  + geom_hline(yintercept=0, color="red", linetype="dashed")
)

#%%#

# CATEs per municipality
estimate_La_NinavsEl_Nino.effect(X)

cates = estimate_La_NinavsEl_Nino.estimate_.cate_estimates
dataframe_cates = pd.DataFrame.from_records(cates)
dataframe_cates.rename(columns={dataframe_cates.columns[0]: 'CATEs'}, inplace=True)

data_period = data[['excess', 'La_NinavsEl_Nino_consensus4', 'SST3', 'SST4', 'SST34',
                    'SOI', 'NATL', 'TROP', 'Pop_density', 'DANE', 'DANE_Period']] 

data_period = data_period.dropna()

# Bind datasets
geo_La_NinavsEl_Nino = pd.concat([data_period.reset_index(drop=True), dataframe_cates], axis=1)

# Grouping the DataFrame by "code.DANE" and calculating the average of "ATEs"
La_NinavsEl_Nino_dane = geo_La_NinavsEl_Nino.groupby('DANE')['CATEs'].mean().reset_index()

# Save the data
La_NinavsEl_Nino_dane.to_csv('D:/Density_La_NinavsEl_Nino_dane.csv', index=False)



#%%#
#Refute tests
#with random common cause
random_La_NinavsEl_Nino = estimate_La_NinavsEl_Nino.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=20)
print(random_La_NinavsEl_Nino)

#with replace a random subset of the data
subset_La_NinavsEl_Nino = estimate_La_NinavsEl_Nino.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.10, num_simulations=20)
print(subset_La_NinavsEl_Nino)

#with replace a dummy outcome
dummy_La_NinavsEl_Nino = estimate_La_NinavsEl_Nino.refute_estimate(method_name="dummy_outcome_refuter", num_simulations=20)
print(dummy_La_NinavsEl_Nino[0])

#with placebo 
placebo_La_NinavsEl_Nino = estimate_La_NinavsEl_Nino.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=20)
print(placebo_La_NinavsEl_Nino)



#%%#

#Figure 3
labs = ['Neutral vs La Niña',
        'Neutral vs El Niño',
        'La Niña vs El Niño',       
]

df_labs = pd.DataFrame({'Labels': labs})

print(df_ATE)

# Convert ATEto separate DataFrame
ATE = df_ATE[['ATE']].round(5)
print(ATE)

# Convert tuples in the '95% CI' column to separate DataFrame
df_ci = df_ATE['95% CI'].apply(pd.Series)

# Rename columns in df_ci
df_ci.columns = ['Lower', 'Upper']

# Create two separate DataFrames for Lower and Upper
Lower = df_ci[['Lower']].copy()
print(Lower)
Upper = df_ci[['Upper']].copy()
print(Upper)


df_plot = pd.concat([df_labs.reset_index(drop=True), ATE, Lower, Upper], axis=1)
print(df_plot)

p = EffectMeasurePlot(label=df_plot.Labels, effect_measure=df_plot.ATE, lcl=df_plot.Lower, ucl=df_plot.Upper)
p.labels(center=0)
p.colors(pointcolor='r' , pointshape="s", linecolor='b')
p.labels(effectmeasure='ATE')  
p.plot(figsize=(10, 5), t_adjuster=0.10, max_value=0.1, min_value=-0.1)
plt.tight_layout()
plt.show()

