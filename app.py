import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error as MSE
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import sweetviz as sv
import numpy as np

st.set_page_config(layout="wide")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# загрузка данных модели
df_train = pd.read_csv('train_df.csv')


# загрузка самой модели
try:
    with open('ridge_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    st.error(f"Ошибка при загрузке модели: {e}")
    st.stop()


def prepare_data(df):
    """
    Преобразование данных под формат модели
    (добавляем One-Hot признаки)
    """

    df = df.drop('Unnamed: 0', axis=1)
    target = df['selling_price']
    df = df.drop('selling_price', axis=1)
    
    cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'auto_mark', 'seats']
    for col in cat_columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
    
    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 
                      'torque', 'max_torque_rpm']
    
    for col in num_cols:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
    
    return df, target

def plot_hist(df, col_name, plot_title, xlabel, ylabel):
    """
    Построение гистограммы
    """

    fig_nbins = st.slider('Количество бинов гистограммы', 25, 300, 25)
    fig = px.histogram(df, col_name,
            title=plot_title,
            labels={'selling_price': 'Цена', 'count': 'Количество автомобилей'},
            nbins=fig_nbins)

    fig.update_layout(
        bargap=0.1,
        xaxis_title=xlabel,
        yaxis_title=ylabel
    )

    st.plotly_chart(fig, use_container_width=True)


with st.sidebar:
    st.title("Данные для предсказания")
    test_file = st.file_uploader('Выберите файл', type='csv')


st.title("Предсказание цен автомобилей")
st.subheader("Информация о данных")

col1, col2 = st.columns(2)
with col1:
    st.metric("Автомобилей", len(df_train))
with col2:
    st.metric("Средняя цена", f"{df_train['selling_price'].mean():,.0f} руб")

plot_hist(df_train, 'selling_price',
    plot_title='Распределение цены автомобилей',
    xlabel="Цена",
    ylabel="Количество"
)

# тут я сильно не успевала построить дашборд руками, поэтому вывела результаты из библиотеки
report = sv.analyze(df_train)
with open('report.html', 'r', encoding='utf-8') as f:
    html_content = f.read()
st.components.v1.html(html_content, height=1000, scrolling=True)

if test_file is not None:
    st.divider()
    st.header("Предсказание на тестовых данных")
    
    try:
        df_test_raw = pd.read_csv(test_file)
        X_test, y_test = prepare_data(df_test_raw)
        model_features = loaded_model.feature_names_in_
        X_test = X_test.reindex(columns=model_features, fill_value=0)
        pred_test = loaded_model.predict(X_test)

        st.subheader("Метрики качества")
        r2 = r2_score(y_test, pred_test)
        mse = MSE(y_test, pred_test)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("R2 Score", f"{r2:.4f}")
        with col2:
            st.metric("MSE", f"{mse:,.0f}")


        coef_df = pd.DataFrame({
                'feature': X_test.columns,
                'coefficient': loaded_model.coef_,
                'abs_coefficient': np.abs(loaded_model.coef_)
        })
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        st.subheader("Топ-10 признаков")
        top_features = coef_df.head(10)
        fig1 = px.bar(
                top_features,
                x='coefficient',
                y='feature',
                orientation='h',
                title='Топ-10 признаков',
                labels={'coefficient': 'Вес', 'feature': 'Признак'}
         )
        fig1.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600,
                showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    except Exception as e:
        st.error(e)
