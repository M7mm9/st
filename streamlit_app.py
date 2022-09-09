import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import datetime as t
from sklearn.metrics import mean_squared_error
from croston import croston

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 6

from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

import warnings

warnings.filterwarnings("ignore")

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


UserName_list = ["A", "B", "C"]
Creation_Date = [t.datetime(2022, 8, 14,), t.datetime(2022, 8, 19,), t.datetime(2022, 8, 27,)]
st.header('Verification')
UserName = st.text_input('Verify Your Subscribtion')
if UserName in UserName_list:
    index = UserName_list.index(UserName)
    Finish_Date = Creation_Date[index] + t.timedelta(days = 30)
    FinishDate = t.datetime(Finish_Date.year, Finish_Date.month, Finish_Date.day)
    Today = t.datetime(t.date.today().year, t.date.today().month, t.date.today().day)

    if Today <= FinishDate:

        st.title("ARIMA Double Layering AI Tool (Al Haitam Data)")
        Year = st.sidebar.selectbox('Current Year', (2022, 2023, 2024))
        st.sidebar.write('--------------')
        ItemGroup = st.sidebar.selectbox('Item Group', ('All Items', '908 Group', 'CA Group', 'CK Group', 'CT Group', 'EP Group', 'Other Items', 'Test'))
        st.sidebar.write('--------------')
        Period = st.sidebar.select_slider('Forecasting Horizon', range(1, 13))
        #Path = rf'C:\Users\maham\Desktop\KFUPM\Senior Project\SP Folders\ARIMA Data\{Year}\{ItemGroup}\Orders.csv'
        #Data1 = pd.read_csv(Path)

        with st.expander('ARIMA Documentation'):
            ARIMA.__doc__

        st.write('------------------------------------------')

        Orders_file = st.file_uploader("Upload your orders data:")

        if Orders_file is not None:
            Data1 = pd.read_csv(Orders_file)

            st.subheader('First Layer')

            Data1['Month'] = pd.to_datetime(Data1['Period'], infer_datetime_format=True)
            Data1_indexed = Data1.set_index(['Month'])
            Data1_indexed.drop('Period', axis='columns', inplace=True)
            Data1_indexed['Orders'] = Data1_indexed['Orders'] + 0.99999
            st.subheader('Orders # Data Pattern:')
            st.line_chart(Data1_indexed['Orders'])

            Data_Indexed_LogScale = np.log(Data1_indexed)
            Data_Indexed_LogScale_Shifted = Data_Indexed_LogScale - Data_Indexed_LogScale.shift()

            with st.expander('Optimize & Forecast # Orders'):
                with st.spinner('In Progress...'):
                    from pmdarima.arima import auto_arima

                    auto_model = auto_arima(Data_Indexed_LogScale[1:], max_p=9, max_d=2, max_q=9)
                    parm = pd.Series(auto_model.get_params())
                    P = parm[2][0]
                    D = parm[2][1]
                    Q = parm[2][2]

                    from statsmodels.tsa.arima.model import ARIMA

                    acuracy_check2 = st.radio('Good Fit?', ('Yes', 'No, It is not'))
                    if acuracy_check2 == 'No, It is not':
                        P = st.number_input('Enter P')
                        D = st.number_input('Enter D')
                        Q = st.number_input('Enter Q')

                    model = ARIMA(Data_Indexed_LogScale, order=(P, D, Q))
                    results = model.fit()
                    #Data_Indexed_LogScale_Shifted.dropna(inplace = True)
                    #esults.fittedvalues.dropna(inplace = True)
                    DataShape = Data_Indexed_LogScale_Shifted['Orders'].values.shape
                    ResultsShape = results.fittedvalues.values.shape
                    RSS = sum((results.fittedvalues[1:] - Data_Indexed_LogScale['Orders'][1:]) ** 2)
                    DataMean = Data_Indexed_LogScale['Orders'].mean()
                st.info(f'Adjusted RSS: {round((RSS**(1/2))/DataMean, 2)}')
                CompinedData = pd.DataFrame(Data_Indexed_LogScale)
                CompinedData['Preds'] = results.fittedvalues
                st.line_chart(CompinedData)
                st.write('----------------')
                st.warning(f'Optimized Model Paramters (Based on RSS):  \n  P = {P}, D = {D}, Q = {Q}')
                st.write('----------------')
                #st.dataframe(auto_model.summary())
                st.write('Model Report & Results:')
                st.write(auto_model.summary())
                st.write('----------------')
                #results.plot_predict(1, len(Data1) + Period)
                st.write('Predictions:')

                PData = pd.DataFrame(results.forecast(Period))
                PData = np.exp(PData)
                PData.columns = ['# of Orders Predictions']
                PData['# of Orders Predictions'] = PData['# of Orders Predictions'] - 0.99999
                st.dataframe(PData)
                PData['Period'] = PData.index
                PData = PData[['Period', '# of Orders Predictions']]
                df_xlsx = to_excel(PData)
                st.download_button(label='📥 Download Forecasted # of Orders',
                                   data=df_xlsx,
                                   file_name=f'Forecasted # of Orders - {ItemGroup} - {Year}.xlsx')

        #----------------------------------------------------------------------------------------------------------------

        #Path = rf'C:\Users\maham\Desktop\KFUPM\Senior Project\SP Folders\ARIMA Data\{Year}\{ItemGroup}\Quantitya.csv'
        #Data1 = pd.read_csv(Path)

        Quantity_file = st.file_uploader("Upload your Quantity data:")

        if Quantity_file is not None:
            Data1 = pd.read_csv(Quantity_file)

            Data1['Month'] = pd.to_datetime(Data1['Period'], infer_datetime_format=True)
            Data1_indexed = Data1.set_index(['Month'])
            Data1_indexed.drop('Period', axis='columns', inplace=True)
            Data1_indexed = Data1_indexed
            st.write('---------------------')
            st.subheader('Second Layer')

            st.subheader('Quantity Data Pattern:')
            st.line_chart(Data1_indexed['Quantity'])

            with st.expander('Optimize & Forecast Quantity'):
                with st.spinner('In Progress...'):
                    Data1['Month'] = pd.to_datetime(Data1['Period'], infer_datetime_format=True)
                    Data1_indexed = Data1.set_index(['Month'])
                    Data1_indexed.drop('Period', axis='columns', inplace=True)
                    Data1_indexed = Data1_indexed
                    Data1_indexed['Quantity'] = Data1_indexed['Quantity'].replace(0, Data1_indexed['Quantity'].mean())

                    Data_Indexed_LogScale = np.log(Data1_indexed)
                    Data_Indexed_LogScale_Shifted = Data_Indexed_LogScale - Data_Indexed_LogScale.shift()
                    #Data_Indexed_LogScale_Shifted.dropna(inplace=True)

                    from pmdarima.arima import auto_arima

                    auto_model = auto_arima(Data_Indexed_LogScale[1:], max_p=9, max_d=2, max_q=9)
                    parm = pd.Series(auto_model.get_params())

                    P = parm[2][0]
                    D = parm[2][1]
                    Q = parm[2][2]


                    from statsmodels.tsa.arima.model import ARIMA

                    acuracy_check2 = st.radio('Good Fit?', ('Yes', 'No'))
                    if acuracy_check2 == 'No':
                        P = st.number_input('Enter P for Quantity')
                        D = st.number_input('Enter D for Quantity')
                        Q = st.number_input('Enter Q for Quantity')

                    model = ARIMA(Data_Indexed_LogScale, order=(P, D, Q))
                    results = model.fit()
                    #Data_Indexed_LogScale_Shifted.dropna(inplace = True)
                    #esults.fittedvalues.dropna(inplace = True)
                    DataShape = Data_Indexed_LogScale_Shifted['Quantity'].values.shape
                    ResultsShape = results.fittedvalues.values.shape
                    RSS = sum((results.fittedvalues[1:] - Data_Indexed_LogScale['Quantity'][1:]) ** 2)
                    DataMean = Data_Indexed_LogScale['Quantity'].mean()
                st.info(f'Adjusted RSS: {round((RSS**(1/2)/DataMean), 2)}')
                CompinedData = pd.DataFrame(Data_Indexed_LogScale)
                CompinedData['Preds'] = results.fittedvalues
                st.line_chart(CompinedData)

                st.write('----------------')
                st.warning(f'Optimized Model Paramters (Based on RSS):  \n  P = {P}, D = {D}, Q = {Q}')
                st.write('----------------')
                #st.dataframe(auto_model.summary())
                st.write('Model Report & Results:')
                st.write(auto_model.summary())
                st.write('----------------')
                #results.plot_predict(1, len(Data1) + Period)
                st.write('Predictions:')

                PData = pd.DataFrame(results.forecast(Period))
                PData = np.exp(PData)
                PData.columns = ['Q Predictions']
                PData['Q Predictions'] = PData['Q Predictions']
                st.dataframe(PData)
                PData['Period'] = PData.index
                PData = PData[['Period', 'Q Predictions']]
                df_xlsx = to_excel(PData)
                st.download_button(label='📥 Download Forecasted Quantity',
                                   data=df_xlsx,
                                   file_name=f'Forecasted Quantity - {ItemGroup} - {Year}.xlsx')

        st.map()

    else:
        st.warning("Sorry, Your Subscription is finished. Subscribe again to access the tool")
else:
    st.warning("Sorry, You are not verified to use this tool. Please Contact the Owner")
