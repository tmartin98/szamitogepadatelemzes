#%% Modulok betöltése
import warnings;
warnings.simplefilter('ignore')
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import  seasonal_decompose
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
plt.style.use('seaborn')

#%% Adatok betöltése, átalakítása, index beállítása

df = pd.read_csv("Rainfall_1901_2016_PAK.csv",sep=',')
df.index = pd.Index(sm.tsa.datetools.dates_from_range('1901m1', '2016m12'))
df = df.rename(columns = {'Rainfall - (MM)':'Rainfall',' Year':'Year'})
df = df.drop(columns=['Year','Month'])
df.head()
#%% Adatok ploton ábrázolása

plt.figure(figsize=(18,8))
plt.plot(df['Rainfall'])
plt.xlabel('Megfigyelt évek')
plt.ylabel('Eső mennyisége')
plt.title('Pakisztáni esők eloszlása')
plt.show()

#%% Mozgóátlagokkal történő ábrázolás

plt.figure(figsize=(16,12))
df['4-month-SMA']=df['Rainfall'].rolling(window=4).mean()
df['8-month-SMA']=df['Rainfall'].rolling(window=8).mean()
df['12-month-SMA']=df['Rainfall'].rolling(window=12).mean()

plt.plot(df['Rainfall'],color='blue')
plt.plot(df['4-month-SMA'],color='red')
plt.plot(df['8-month-SMA'],color='yellow')
plt.plot(df['12-month-SMA'],color='green')
plt.xlabel('Megfigyelt évek')
plt.ylabel('Eső mennyisége')
plt.legend(["Rainfall","4-month-SMA","8-month-SMA","12-month-SMA"])
plt.title('Mozgóátlagok összehasonlítása az eredeti adathalmazzal')
plt.show()

#%% Exponenciálisan súlyozott mozgóátlag, összehasonlítása a 12 hónapos mozgóátlaggal

df['EWMA12'] = df['Rainfall'].ewm(span=12,adjust=False).mean()
plt.figure(figsize=(16,8))
plt.plot(df['Rainfall'],color='green')
plt.plot(df['EWMA12'],color='red')
plt.plot(df['12-month-SMA'],color='yellow')
plt.xlabel('Megfigyelt évek')
plt.ylabel('Eső mennyisége')
plt.legend(['Eső mennyisége','Exponenciálisan súlyozott mozgóátlaggal','12 hónapos mozgóátlag'])
plt.title('Exponenciális mozgóátlag összehasonlítása az eredeti adathalmazzal és a 12 hónapos mozgóátlaggal')
plt.show()

#%% Háromszoros exponenciális simítás

df['TESadd12'] = ExponentialSmoothing(df['Rainfall'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
plt.figure(figsize=(16,8))
plt.plot(df['Rainfall'],color='green')
plt.plot(df['TESadd12'],color='yellow')
plt.xlabel('Megfigyelt évek')
plt.ylabel('Eső mennyisége')
plt.legend(['Eső mennyisége','Exponenciális simítással'])
plt.title('Háromszoros exponenciális simítás összehasonlítása az eredeti adathalmazzal')
plt.show()

#%% Seasonal decomposal megnézzünk, hogy követ-e valamilyen trendet, illetve, hogy van e szezonalitás

from pylab import rcParams
rcParams['figure.figsize'] = 15, 12
rcParams['axes.labelsize'] = 20
rcParams['ytick.labelsize'] = 16
rcParams['xtick.labelsize'] = 16
result = seasonal_decompose(df['Rainfall'], model='multiplicative').plot()
result

#%% Arima importálása, és stacionaritás tesztelése Dickey-Fuller tesztel

import pmdarima as pmd
result = adfuller(df['Rainfall'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

# Látható, hogy a p változó értéke kisebb, mint 0.05, ebből kifolyólag az idősorunk stacionárius.
#%% Korrelogramm alkalmazása

plot_acf(df['Rainfall'].values.squeeze(),lags=25)
plot_pacf(df['Rainfall'].values.squeeze(),lags=25)

# A korrelogrammot arra szokták használni, hogy megfigyeljék az adatok randomitását, azaz, 
# hogy van e az adatokban valamilyen trend, szezonalitás. Ha az adatok között nincs 
# semmilyen összefüggés akkor az autokorrelációs függvénynek közel kellene lennie a 0-hoz.
#Látható, a szezonalistás a korrelogrammon, folyamatosan váltakozik 
# az értékek között egyszer csökken egyszer nő.
#%% Autoarima használata, az optimális modell meghatározásása
autoarima_model = pmd.auto_arima(df['Rainfall'], 
                              start_p=1, 
                              start_q=1,
                              test="adf",
                              trace=True)

#%% Az optimális modell megvizsgálása
autoarima_model.summary()
#Látható, hogy a legjobb AIC érték a 12165.668 ami a 2,0,3 orderhez tartozó modellre igaz.
#%% Arima modell diagrammon ábrázolása

model = ARIMA(df['Rainfall'],order=(2,0,3),freq='M')
result = model.fit(disp=1)

plt.figure(figsize=(18,12))
plt.plot(df['Rainfall'])
plt.plot(result.fittedvalues,color='yellow')
plt.xlabel('Megfigyelt évek')
plt.ylabel('Eső mennyisége')
plt.legend(['Eredeti értékek','Prediktált értékek'])
plt.title('Prediktált értékek összehasonlítása az eredetivel')
plt.show()

#%%
arima_pred = result.predict(start = '1901-01-31', end = len(df)-1, typ="levels").rename("ARIMA Predictions")
df['ARIMA_Predictions'] = arima_pred
mse = mean_squared_error(df['Rainfall'],arima_pred)

print("Rainfall mean",df['Rainfall'].mean())
print("RMSE value:",rmse(df['Rainfall'],arima_pred))

# Esetünkben véleményem szerint, az rmse értéke a kiugró adatok mellett teljesen 
# érthető, mivel az átlag is 25 körül van.
#%% Előrejelzés a kapott modellel

start_index = '2017-01-31'
end_index = '2020-03-31'
a = result.predict(start=start_index, end=end_index)
a

#%% Előrejelzett értékek diagrammon való ábrázolása

plt.figure(figsize=(18,12))
plt.plot(df['Rainfall'])
plt.plot(result.fittedvalues,color='green')
plt.plot(a,color='red')
plt.xlabel('Megfigyelt évek')
plt.ylabel('Eső mennyisége')
plt.legend(['Eredeti értékek','Prediktált értékek','Előrejelzett eső mennyisége'])
plt.title('Prediktált értékek, illetve előrejelzett értékek összehasonlítása az eredeti adatokkal')
plt.show()

# %% Adatok átalakítása, hónapokban történő ábrázolásra

months = df.index.month
monthly_sum=df.groupby(months).sum()
plt.figure(figsize=(18,6))
plt.plot(monthly_sum['Rainfall'])
plt.plot(monthly_sum['ARIMA_Predictions'])
plt.xlabel('Hónapok')
plt.ylabel('Összegzett esőmennyisége')
plt.legend(['Eredeti eső mennyisége','Számított mennyisége'])
plt.title('Az eső eloszlása a megfigyelt években, hóhapokra lebontva.')
plt.show()

#%% Bar charton ábrázoljuk a prediktált illetve az eredeti adatok különbségét.

monthly_sum[['Rainfall','ARIMA_Predictions']].plot(figsize=(18,8),kind='bar',title=('Eredeti adatok, illetve prediktált adatok összehasonlítása')).autoscale(axis='x',tight=True)


# %%
