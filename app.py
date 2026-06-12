import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

st.set_page_config(page_title="LLM Analysis", layout="wide")
st.title( 'LLM Benchmark Leaderboard Dataset (2024–2026)')

st.header("Annotation")
st.write("Этот проект исследует рынок современных языковых моделей (LLM). Основная цель - выявить закономерности ценообразования, оценить разрыв между Open Source и закрытыми решениями, а также проанализировать влияние наличия инструментов (tool use) на качество генерации кода.")

st.subheader("Сквадик")
st.write("- **Никита**:")
st.write("- **Михаил**: ")
st.write("- **Филипп**:")

try:
    res = requests.get("http://127.0.0.1:8000/models?limit=50")
    df = pd.DataFrame(res.json())
except:
    df = pd.read_csv("dataset.csv")

st.header("Dataset Overview")
st.dataframe(df, use_container_width=True)

col1,col2,col3 =st.columns(3)
col1.metric("Всего моделей", len(df))
col2.metric("Средний MMLU", round(df['mmlu_score'].mean(), 1))
col3.metric("Open Source", len(df[df['open_weights']==True]))

st.write("Датасет содержит информацию о ведущих языковых моделях: их характеристики, результаты бенчмарков (MMLU, HumanEval, GPQA) и стоимость токенов. У нас было суммарно около 20 пропусков(в parameter_count_B и GPQA)В процессе предобработки пропуски в GPQA были восстановлены через корреляцию с MMLU, а параметр b провто приравнен к 0")

st.header("Exploratory Data Analysis (EDA)")

c1, c2= st.columns(2)
with c1:
    fig1 = px.histogram(df, x='mmlu_score', nbins=10, title='MMLU Score', height=450)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Вывод: Большинство моделей имеют базовый интеллект (MMLU) в диапазоне от 80 до 90 баллов. Рынок явно смещен в сторону высокопроизводительных решений.")

with c2:
    fig2 =px.histogram(df[df['parameter_count_B']>0], x='parameter_count_B', nbins=10, title='Parameters (Billion)', height=450)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Вывод: Среди открытых моделей доминируют относительно небольшие размеры (до 150B параметров), так как их проще обучать и разворачивать локально.")

c3, c4 =st.columns(2)
with c3:
    fig3 =px.histogram(df[df['cost_per_1M_combined']<50], x='cost_per_1M_combined', nbins=15, title='Cost (under $50)', height=450)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Вывод: Огромное скопление около нуля из-за бесплатных Open Source моделей. Платные модели в основном стоят до 20 долларов за миллион токенов.")

with c4:
    fig4=px.histogram(df, x='value_for_money', nbins=20, title='Value for Money', height=450)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Вывод: Метрика выгоды показывает сильный перекос. Платные API математически сильно проигрывают открытым весам по параметру чистого соотношения цена/качество.")

c5, c6=st.columns(2)
with c5:
    fig5= px.scatter(df, x='cost_per_1M_combined', y='mmlu_score', color='open_weights', title='Cost vs MMLU by Open Weights', height=450)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Вывод: Самые дорогие модели не гарантируют наивысший MMLU. Опенсорс (True) с нулевой стоимостью показывает отличный разброс качества, конкурируя на равных с платными.")

with c6:
    fig6 = px.box(df, x='price_tier', y='arena_elo_score', color='price_tier', title='Arena ELO by Price Tier', category_orders={"price_tier": ["Free/Open", "Ultra-Low", "Budget", "Mid-Range", "Premium"]}, height=450)
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Вывод: Премиум-сегмент стабилен, но в среднем ценовом сегменте (Mid-Range) есть модели-выбросы, которые по народному рейтингу обгоняют даже самые дорогие решения.")

c7, c8 = st.columns(2)
with c7:
    df_params =df[df['parameter_count_B'] > 0]
    fig7 = px.scatter(df_params, x='parameter_count_B', y='mmlu_score', size='context_window_tokens', color='provider', title='Params vs MMLU (Known params only)', height=450)
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("Вывод: Наблюдается четкий восходящий тренд — чем больше параметров у модели, тем она умнее. Размер контекстного окна (размер пузырька) при этом распределен хаотично.")

with c8:
    fig8 =px.violin(df, x='supports_tool_use', y='humaneval_score', color='supports_tool_use', box=True, title='Coding Score by Tool Use Support', height=450)
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("Вывод: Модели, способные использовать инструменты (True), пишут код (HumanEval) кардинально лучше тех, у кого этой функции нет. Распределения почти не пересекаются.")

numeric_cols= ['parameter_count_B', 'context_window_tokens', 'cost_per_1M_combined', 'arena_elo_score', 'mmlu_score', 'humaneval_score', 'gpqa_diamond_score']
fig9=px.imshow(df[numeric_cols].corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title='Correlation Matrix', height=600)
st.plotly_chart(fig9, use_container_width=True)
st.caption("Вывод: Сильная корреляция (0.87) между MMLU и Arena ELO подтверждает, что пользователи на слепых тестах чувствуют реальный ум модели. Стоимость имеет слабую связь с качеством (0.33).")



import plotly.graph_objects as go

st.header("Продвинутая визуализация")

st.subheader("Сравнение профилей моделей (Spider Chart)")
models_list = df['model_name'].tolist()

c_sel1, c_sel2 = st.columns(2)
with c_sel1:
    m1 = st.selectbox("Модель 1", models_list, index=0)
with c_sel2:
    m2 = st.selectbox("Модель 2", models_list, index=1)

df_m1 = df[df['model_name']==m1].iloc[0]
df_m2 =df[df['model_name']== m2].iloc[0]

cats = ['MMLU (База)', 'HumanEval (Кодинг)', 'Math (Математика)', 'GPQA (Сложная логика)']

fig_radar =go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=[df_m1['mmlu_score'], df_m1['humaneval_score'], df_m1['math_score'], df_m1['gpqa_diamond_score']],
    theta=cats,
    fill='toself',
    name=m1
))
fig_radar.add_trace(go.Scatterpolar(
    r=[df_m2['mmlu_score'], df_m2['humaneval_score'], df_m2['math_score'], df_m2['gpqa_diamond_score']],
    theta=cats,
    fill='toself',
    name=m2,
    line=dict(color='red')
))

fig_radar.update_layout(
  polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
  showlegend=True
)
st.plotly_chart(fig_radar, use_container_width=True)
st.caption("**Зачем это:** Позволяет оценить специализацию модели. Средний скор часто обманчив.\n**Что мы видим:** Накладывая две модели друг на друга, сразу понятно, кто из них лучше пишет код, а кто лучше решает математику. Идеально для выбора инструмента под узкую задачу.")

st.subheader("Многомерная карта рынка (3D Scatter)")

fig_3d= px.scatter_3d(df, x='parameter_count_B', y='cost_per_1M_combined', z='mmlu_score',
              color='open_weights', hover_name='model_name',
              labels={'parameter_count_B': 'Параметры (B)', 'cost_per_1M_combined': 'Цена ($/1M)', 'mmlu_score': 'MMLU'})
              
fig_3d.update_traces(marker=dict(size=6, opacity=0.8))
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600)
st.plotly_chart(fig_3d, use_container_width=True)
st.caption("**Зачем это:** Объединяет три главных измерения (размер железа, стоимость токенов и базовый ум) в одном пространстве для поиска кластеров и аномалий.\n**Что мы видим:** Если покрутить график, видно как опенсорс (синие точки) лежит строго на нулевой плоскости по цене, плавно растягиваясь по оси параметров и набирая высоту (интеллект). Закрытые модели отрываются от пола, но далеко не всегда уходят сильно выше по оси MMLU.")

st.header("Hypothesis Testing")

st.subheader("Гипотеза 1: Сокращение разрыва между Open Source и Proprietary")
st.markdown('''
**Нулевая гипотеза (H0):** Среднее значение базового интеллекта (MMLU) среди моделей с открытым исходным кодом и закрытых моделей, выпущенных за последние 400 дней, статистически не различается.

**Альтернативная гипотеза (H1):** Существует статистически значимая разница в средних значениях MMLU между этими двумя группами.
''')
recent= df[df['days_since_release'] < 400]
open_m = recent[recent['open_weights'] == True]['mmlu_score']
prop_m= recent[recent['open_weights'] == False]['mmlu_score']
stat, pval= stats.ttest_ind(prop_m, open_m, equal_var=False)

col_h1_1, col_h1_2= st.columns(2)
col_h1_1.metric("T-statistic", round(stat, 4))
col_h1_2.metric("P-value", round(pval, 4))
st.success("Вывод: Так как p-value > 0.05, нулевая гипотеза не отвергается. Разрыв между Open Source и закрытыми решениями в последних поколениях статистически незначим.")

st.subheader("Гипотеза 2: Иллюзия премиальности (Множественная линейная регрессия)")
st.markdown('''
Для проверки влияния цены на пользовательский рейтинг при фиксированном базовом интеллекте мы строим модель множественной линейной регрессии (Multiple Linear Regression).
''')

st.latex(r"Arena\_ELO = \beta_0 + \beta_1 \cdot MMLU\_Score + \beta_2 \cdot Cost\_per\_1M + \epsilon")

st.markdown('''
Где $\\beta_1$ оценивает вклад чистого интеллекта, а $\\beta_2$ — влияние стоимости токенов.

**Нулевая гипотеза (H0):** $\\beta_2 = 0$ (стоимость не влияет на ELO при контроле MMLU).

**Альтернативная гипотеза (H1):** $\\beta_2 \\neq 0$ (стоимость является значимым предиктором).
''')

X =df[['mmlu_score', 'cost_per_1M_combined']]
X=sm.add_constant(X)
Y= df['arena_elo_score']
model_ols = sm.OLS(Y, X).fit()

results_df = pd.DataFrame({
    "Коэффициент (coef)": model_ols.params,
    "P-value": model_ols.pvalues,
    "Стандартная ошибка (std err)": model_ols.bse
}).round(4)

st.dataframe(results_df, use_container_width=True)
st.metric("R-squared модели", round(model_ols.rsquared, 3))
st.success("Вывод: Для переменной cost_per_1M_combined значение p-value > 0.05. Нулевая гипотеза не отвергается. Пользователи отдают предпочтение объективному интеллекту (MMLU), игнорируя наценку за премиальность.")




st.header("Discussion")
st.write("")
with st.sidebar:
    st.header("Добавить новую модель")
    with st.form("add_model_form"):
        m_name= st.text_input("Название модели")
        m_prov=st.text_input("Провайдер")
        m_mmlu = st.number_input("MMLU Score", min_value=0.0, max_value=100.0, value=80.0)
        m_cost =st.number_input("Стоимость 1M токенов", min_value=0.0, value=1.0)
        m_open= st.checkbox("Open Weights")
        
        submitted =st.form_submit_button("Отправить")
        if submitted:
            payload = {
                "model_name": m_name,
                "provider": m_prov,
                "mmlu_score": m_mmlu,
                "cost_per_1M_combined": m_cost,
                "open_weights": m_open
            }
            try:
                resp = requests.post("http://127.0.0.1:8000/models", json=payload)
                if resp.status_code == 200:
                    st.success("Успешно добавлено через API!")
            except:
                new_row = pd.DataFrame([payload])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv("dataset.csv", index=False)
                st.success("Успешно добавлено локально!")
    st.image("Image.png", use_container_width=True)