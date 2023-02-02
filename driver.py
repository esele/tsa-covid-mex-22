import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os.path
import statsmodels.tsa.ar_model as ar_model
import statsmodels.tsa.arima.model as arima_model
import itertools
from matplotlib import cycler

def graficar(serie,title,xax,yax,etiqueta,filename):
    """
       graficar(serie,titulo,xax,yax,filename): genera una gráfica
       Entrada: serie: datos a graficar
                titulo: título de la gráfica (opcional)
                xax: nombre del eje x (opcional)
                yax: nombre del eje y (opcional)
                filename: nombre para guardar la gráfica
    """
    if title is None:
        title=""
    if xax is None:
        xax="equis"
    if yax is None:
        yax="i griega"
    plt.figure(0,figsize=(12.8,7.2))
    plt.plot(serie,label=etiqueta)
    plt.title(title)
    plt.xlabel(xax)
    plt.ylabel(yax)
    plt.legend()
    plt.savefig(filename,bbox_inches="tight")
    plt.close("all")

def graficar_acf_pacf(serie,lag,acf_title,pacf_title,acf_filename,pacf_filename):
    fig,ax=plt.subplots(figsize=(12.8,7.2))
    sm.graphics.tsa.plot_acf(serie.to_numpy().squeeze(),lags=lag,fft=True,alpha=.05,
                             title=acf_title,ax=ax)
    plt.xlabel("Retraso")
    plt.ylabel("Correlación")
    plt.savefig(acf_filename,bbox_inches="tight")
    fig,ax=plt.subplots(figsize=(12.8,7.2))
    sm.graphics.tsa.plot_pacf(serie.to_numpy().squeeze(),lags=lag,alpha=.05,method="ywm",
                             title=pacf_title,ax=ax)
    plt.xlabel("Retraso")
    plt.ylabel("Correlación parcial")
    plt.savefig(pacf_filename,bbox_inches="tight")
    plt.close("all")

def graficar_diag(modelo,filename):
    plt.figure(0)
    modelo.plot_diagnostics(figsize=(19.2,10.8))
    plt.savefig(filename,bbox_inches="tight")
    plt.close("all")

def graficar_pred(pred,serie,title,xax,yax,filename,ci=True):
    if ci:
        pred_ci = pred.conf_int()
    plt.figure(0,figsize=(12.8,7.2))
    plt.plot(serie,label="Valores observados")
    pred.predicted_mean.plot(label="Valores predecidos", alpha=.7)
    if ci:
        plt.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color="#E6E6E6", alpha=.5)
    plt.xlabel(xax)
    plt.ylabel(yax)
    plt.title(title)
    plt.legend()
    plt.savefig(filename,bbox_inches="tight")
    plt.close("all")

def prueba_adf_kpss(serie):
    test1=sm.tsa.stattools.adfuller(serie)
    test2=sm.tsa.stattools.kpss(serie)
    return test1[1],test2[1]

def min_aic(serie,pdq,aic):
    j=0
    for i in pdq:
        try:
            res=arima_model.ARIMA(serie,order=i,enforce_stationarity=False,enforce_invertibility=False).fit()
            print(res.summary())
            aic[j]=res.aic
            print(aic[j])
            j+=1
        except:
            continue
    return pdq[np.argmin(aic)]

if __name__=="__main__":
    alpha=0.05

    # esto se encarga de inicializar las gráficas
    plt.close("all")
    colors = cycler("color",["#1234BB","#319F5B","#C70664",
                                "#C40F0F","#B74B03","#92C101",
                                "#620CC7","#2EA183","#36ACB2"])
    plt.rc("axes", facecolor='#AAAAAA', edgecolor="none",
            axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc("grid", color="w", linestyle="solid")
    plt.rc("xtick", direction="out", color="#222222")
    plt.rc("ytick", direction="out", color="#222222")
    plt.rc("patch", edgecolor="#AAAAAA")
    plt.rc("lines", linewidth=2)
    
    # lee los casos de los datos diarios y los grafica
    diarios=pd.read_csv("daily.csv",low_memory=False,index_col='FECHA',parse_dates=True,infer_datetime_format=True)
    diarios=diarios.loc[:,"Nacional"]
    graficar(diarios,
             "Historial de casos diarios de COVID-19 en México nacionales,\nFebrero de 2020 a Mayo de 2022",
             "Fecha",
             "Casos diarios",
             "Casos en el país",
             "casos_diarios_total")

    # grafica los datos pero sólo en un rango de fecha
    fecha="3/1/2022"
    diarios_off=diarios.loc[fecha:]
    graficar(diarios_off,
             "Historial de casos diarios de COVID-19 en México nacionales,\nMarzo a Mayo de 2022",
             "Fecha",
             "Casos diarios",
             "Casos en el país",
             "casos_marzo")
    graficar_acf_pacf(diarios_off,25,
                      "Función de autocorrelación de los casos diarios de COVID-19\na partir de marzo del 2022",
                      "Función de autocorrelación parcial de los casos diarios de COVID-19\na partir de marzo del 2022",
                      "acf_marzo","pacf_marzo")
    print("Valor p de la prueba DFA: ",prueba_adf_kpss(diarios_off)[0],"\nValor p de la prueba KPSS:",prueba_adf_kpss(diarios_off)[1])

    diarios_dif=(diarios_off-diarios_off.shift(1)).dropna()
    graficar(diarios_dif,
             "Serie diferenciada del historial de casos diarios de COVID-19 en México nacionales,\nMarzo a Mayo de 2022",
             "Fecha",
             "Casos diarios",
             "Casos en el país (diferenciados)",
             "casos_marzo_dif")
    graficar_acf_pacf(diarios_dif,25,
                      "Función de autocorrelación de los casos diarios de COVID-19 diferenciados\na partir de marzo del 2022",
                      "Función de autocorrelación parcial de los casos diarios de COVID-19 diferenciados\na partir de marzo del 2022",
                      "acf_marzo_dif","pacf_marzo_dif")
    print("Valor p de la prueba DFA: ",prueba_adf_kpss(diarios_dif)[0],"\nValor p de la prueba KPSS:",prueba_adf_kpss(diarios_dif)[1])

    # p=q=range(0,10)
    # pq=list(itertools.product(p,{0},q))
    # aic=np.zeros(100)
    # x=min_aic(diarios_off,pq,aic)
    # print(x)
    # (8,0,9)
    # res=arima_model.ARIMA(diarios_off,order=x,enforce_stationarity=False,enforce_invertibility=False).fit()
    res=arima_model.ARIMA(diarios_off,order=(6,3,9),enforce_stationarity=False,enforce_invertibility=False).fit()
    print(res.summary())
    graficar_diag(res,"modelo_marzo_diag")

    # genera una predicción fuera de muestra
    dias=31
    pred=res.get_forecast(steps=dias)
    graficar_pred(pred,diarios_off,
                  "Casos diarios de COVID-19 en México nacionales esperados a partir de Mayo del 2022\ncon datos de Marzo y Abril del 2022",
                  "Fecha",
                  "Casos diarios esperados",
                  "pred_mayo_marzo")

    #genera una predicción dentro de muestra
    pred=res.get_prediction(start="4/1/2022")
    graficar_pred(pred,diarios_off,
                  "Casos diarios de COVID-19 en México nacionales esperados a partir de Abril del 2022\ncon datos de Marzo y Abril del 2022",
                  "Fecha",
                  "Casos diarios esperados",
                  "pred_abril_marzo")

    # grafica los datos pero sólo en un rango de fecha
    fecha="12/1/2021"
    diarios_off=diarios.loc[fecha:]
    graficar(diarios_off,
             "Historial de casos diarios de COVID-19 en México nacionales,\nDiciembre de 2021 a Mayo de 2022",
             "Fecha",
             "Casos diarios",
             "Casos en el país",
             "casos_diciembre")
    graficar_acf_pacf(diarios_off,25,
                      "Función de autocorrelación de los casos diarios de COVID-19\na partir de diciembre del 2021",
                      "Función de autocorrelación parcial de los casos diarios de COVID-19\na partir de diciembre del 2021",
                      "acf_marzo","pacf_marzo")
    print("Valor p de la prueba DFA: ",prueba_adf_kpss(diarios_off)[0],"\nValor p de la prueba KPSS:",prueba_adf_kpss(diarios_off)[1])

    diarios_dif=(diarios_off-diarios_off.shift(1)).dropna()
    graficar(diarios_dif,
             "Serie diferenciada del historial de casos diarios de COVID-19 en México nacionales,\nDiciembre de 2021 a Mayo de 2022",
             "Fecha",
             "Casos diarios",
             "Casos en el país (diferenciados)",
             "casos_diciembre_dif")
    graficar_acf_pacf(diarios_dif,25,
                      "Función de autocorrelación de los casos diarios de COVID-19 diferenciados\na partir de diciembre del 2021",
                      "Función de autocorrelación parcial de los casos diarios de COVID-19 diferenciados\na partir de diciembre del 2021",
                      "acf_diciembre_dif","pacf_diciembre_dif")
    print("Valor p de la prueba DFA: ",prueba_adf_kpss(diarios_dif)[0],"\nValor p de la prueba KPSS: ",prueba_adf_kpss(diarios_dif)[1])

    # p=q=range(0,10)
    # pq=list(itertools.product(p,{0},q))
    # aic=np.zeros(100)
    # x=min_aic(diarios_dif,pq,aic)
    # print(x)
    # (6,3,9) (pero una diferenciación aparte)
    # res=arima_model.ARIMA(diarios_dif,order=x,enforce_stationarity=False,enforce_invertibility=False).fit()
    res=arima_model.ARIMA(diarios_dif,order=(6,3,9),enforce_stationarity=False,enforce_invertibility=False).fit()
    print(res.summary())
    graficar_diag(res,"modelo_diciembre_diag")

    # genera una predicción fuera de muestra
    dias=31
    pred=res.get_forecast(steps=dias)
    graficar_pred(pred,diarios_dif,
                  "Casos diarios de COVID-19 en México nacionales esperados a partir de Mayo del 2022\ncon datos de Diciembre del 2021, diferenciados",
                  "Fecha",
                  "Casos diarios esperados",
                  "pred_mayo_diciembre_dif")

    #genera una predicción dentro de muestra
    pred=res.get_prediction(start="3/1/2022")
    graficar_pred(pred,diarios_dif,
                  "Casos diarios de COVID-19 en México nacionales esperados a partir de Marzo del 2022\ncon datos de Diciembre del 2021, diferenciados",
                  "Fecha",
                  "Casos diarios esperados",
                  "pred_marzo_diciembre_dif")

    # total!!
    graficar_acf_pacf(diarios,25,
                      "Función de autocorrelación de los casos diarios de COVID-19",
                      "Función de autocorrelación parcial de los casos diarios de COVID-19",
                      "acf_total","pacf_total")
    print("Valor p de la prueba DFA: ",prueba_adf_kpss(diarios)[0],"\nValor p de la prueba KPSS:",prueba_adf_kpss(diarios)[1])

    diarios_dif=(diarios-diarios.shift(1)).dropna()
    graficar(diarios_dif,
             "Serie diferenciada del historial de casos diarios de COVID-19 en México nacionales",
             "Fecha",
             "Casos diarios",
             "Casos en el país (diferenciados)",
             "casos_dif")
    graficar_acf_pacf(diarios_dif,25,
                      "Función de autocorrelación de los casos diarios de COVID-19 diferenciados",
                      "Función de autocorrelación parcial de los casos diarios de COVID-19 diferenciados",
                      "acf_total_dif","pacf_total_dif")
    print("Valor p de la prueba DFA: ",prueba_adf_kpss(diarios_dif)[0],"\nValor p de la prueba KPSS: ",prueba_adf_kpss(diarios_dif)[1])

    # p=d=q=range(0,10)
    # pdq=list(itertools.product(p,d,q))
    # aic=np.zeros(1000)
    # x=min_aic(diarios_dif,pdq,aic)
    # print(x)
    # (8,1,9) (pero una diferenciación aparte)
    # res=arima_model.ARIMA(diarios_dif,order=x,enforce_stationarity=False,enforce_invertibility=False).fit()
    res=arima_model.ARIMA(diarios,order=(8,2,9),enforce_stationarity=False,enforce_invertibility=False).fit()
    print(res.summary())
    # graficar_diag(res,"modelo_total_diag")

    # genera una predicción fuera de muestra
    dias=31
    pred=res.get_forecast(steps=dias)
    graficar_pred(pred,diarios,
                  "Casos diarios de COVID-19 en México nacionales esperados a partir de Mayo del 2022\ncon todos los datos (sin IP)",
                  "Fecha",
                  "Casos diarios esperados",
                  "pred_mayo_total_noci",ci=False)
    dias=244
    pred=res.get_forecast(steps=dias)
    graficar_pred(pred,diarios,
                  "Casos diarios de COVID-19 en México nacionales esperados para el 2022\ncon todos los datos (sin IP)",
                  "Fecha",
                  "Casos diarios esperados",
                  "pred_22_total_noci",ci=False)

    #genera una predicción dentro de muestra
    pred=res.get_prediction(start="1/1/2022")
    graficar_pred(pred,diarios,
                  "Casos diarios de COVID-19 en México nacionales esperados a partir de Enero del 2022\ncon todos los datos (sin IP)",
                  "Fecha",
                  "Casos diarios esperados",
                  "pred_enero_total_noci",ci=False)