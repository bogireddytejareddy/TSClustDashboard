from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from PIL import Image
from constant import *
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go


#plt.style.use('dark_background')
df = pd.read_csv('data/results.csv')
characteristics_df = pd.read_csv('data/characteristics.csv')
characteristics_df.reset_index()
characteristics_df.drop(columns=characteristics_df.columns[0], axis=1, inplace=True)
characteristics_df.columns = ['Name', 'NumOfTrainingSamples', 'NumOfTestingSample', 'NumOfSamples', 'SeqLength', 'NumOfClusters', 'Type']
characteristics_df = characteristics_df[['Name', 'NumOfSamples', 'SeqLength', 'NumOfClusters', 'Type']]


def plot_time_plot(measure_name):
    container_method = st.container()
    all_time_method = st.checkbox("Select all",key='all_time_method')
    if all_time_method: time_methods_family = container_method.multiselect('Select a group of methods', sorted(time_methods), sorted(time_methods), key='selector_time_methods_all')
    else: time_methods_family = container_method.multiselect('Select a group of methods', sorted(time_methods), key='selector_time_methods')


    if len(time_methods_family) > 0:
        time, time_text, x, y = get_bubble_data(time_methods_family, measure_name)

        fig = go.Figure(data=[go.Scatter(
            x=x,
            y=y,
            text=time_text,
            mode='markers',
            marker=dict(
                size=time,
                sizemode='area',
                sizeref=2.*max(time)/(80.**2),
                sizemin=4
            )
        )])
        fig.update_layout(showlegend=False, width=800, height=600, template="plotly_white", font=dict(family="Arial",
                                                        size=19,
                                                        color="black"))

        st.plotly_chart(fig, theme="streamlit", use_container_width=False)


def plot_box_plot(df, measure_name, methods_family, datasets, scale='linear'):
    if len(df.columns) > 0:
        tab1, tab2 = st.tabs(["Box Plot", "Scatter Plot"])
        with tab1:
            fig = go.Figure()
            for cols in df.columns[1:]:
                fig.add_trace(go.Box(y=df[cols], name=cols[:-len(measure_name)-1]))
            fig.update_layout(showlegend=False, width=800, height=600, template="plotly_white", font=dict(family="Arial",
                                                        size=19,
                                                        color="black"))

            st.plotly_chart(fig, theme="streamlit", use_container_width=False)

            cols_list = []
            for i, col in enumerate(df.columns):
                if i > 0:
                    cols_list.append(col[:-len(measure_name)-1])
                else:
                    cols_list.append(col)

            df.columns = cols_list
        with tab2:
            option1 = st.selectbox(
                'Method 1',
                tuple(methods_family))
            methods_family = methods_family[1:] + methods_family[:1]
            option2 = st.selectbox(
                'Method 2',
                tuple(methods_family))

            if len(methods_family) > 0 and len(datasets) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df[option1], y=df[option2], mode='markers', name='(Method 1, Method 2)'))
                fig.add_trace(go.Scatter(
                                    x=[min(min(df[option1])+1e-4, min(df[option2])+1e-4), max(max(df[option1])+1e-4, max(df[option2])+1e-4)],
                                    y=[min(min(df[option1])+1e-4, min(df[option2])+1e-4), max(max(df[option1])+1e-4, max(df[option2])+1e-4)],
                                    name="X=Y"
                                ))
                fig.update_layout(template="plotly_white", xaxis_title=option1, yaxis_title=option2, font=dict(family="Arial",
                                                            size=19,
                                                            color="black"))
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        AgGrid(df)
        

def generate_dataframe(df, datasets, methods_family, metric_name):
    df = df.loc[df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in methods_family]]
    df.insert(0, 'Datasets', datasets)
    return df
      
    
with st.sidebar:
    st.markdown('# TSClustering') 
    metric_name = st.selectbox('Pick an assessment metric', list_measures)

    container_dataset = st.container()  
    all_cluster = st.checkbox("Select all", key='all_clusters')
    if all_cluster: cluster_size = container_dataset.multiselect('Select cluster size', sorted(list(list_num_clusters)), sorted(list(list_num_clusters)))
    else: cluster_size = container_dataset.multiselect('Select cluster size', sorted(list(list_num_clusters)))

    container_dataset = st.container()  
    all_length = st.checkbox("Select all", key='all_lengths')
    if all_length: length_size = container_dataset.multiselect('Select sequence length size', sorted(list(list_seq_length)), sorted(list(list_seq_length)))
    else: length_size = container_dataset.multiselect('Select length sequence size', sorted(list(list_seq_length)))

    container_dataset = st.container()  
    all_type = st.checkbox("Select all", key='all_types')
    if all_type: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)), sorted(list(list_type)))
    else: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)))

    container_dataset = st.container()  
    all_dataset = st.checkbox("Select all", key='all_dataset')
    if all_dataset: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
    else: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types))) 
    
    container_method = st.container()
    all_class = st.checkbox("Select all",key='all_class')
    if all_class: class_family = container_method.multiselect('Select a group of classes', sorted(list(classes)), sorted(list(classes)), key='selector_class_all')
    else: class_family = container_method.multiselect('Select a group of classes', sorted(list(classes)), key='selector_class')

    container_method = st.container()
    all_method = st.checkbox("Select all",key='all_method')
    if all_method: methods_family = container_method.multiselect('Select a group of methods', sorted(find_methods(class_family)), sorted(find_methods(class_family)), key='selector_methods_all')
    else: methods_family = container_method.multiselect('Select a group of methods', sorted(find_methods(class_family)), key='selector_methods')


df = pd.read_csv('data/results.csv')
tab_desc, tab_acc, tab_time, tab_dataset, tab_method = st.tabs(["Description", "Evaluation", "Execution Time", "Datasets", "Methods"])  

with tab_desc:
    st.markdown('# TSClustering')
    st.markdown(description_intro)

with tab_acc:
    st.markdown('# Evaluation')
    st.markdown('Overall evaluation of time-series clustering algorithms used over 128 datasets. Measure used: {}'.format(metric_name))
    df_toplot = generate_dataframe(df, datasets, methods_family, metric_name)
    plot_box_plot(df_toplot, measure_name=metric_name, methods_family=methods_family, datasets=datasets)
    
with tab_time:
    st.markdown('# Execution Time')
    plot_time_plot(metric_name)

with tab_dataset:
    st.markdown('# Dataset Description')
    st.markdown(text_description_dataset)
    AgGrid(characteristics_df)

with tab_method:
    st.markdown('# Time-Series Clustering Methods')
    st.markdown(text_description_models, unsafe_allow_html=True)

