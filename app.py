from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from constant import *
from constant import compute_CD
from st_aggrid import AgGrid
import plotly.graph_objects as go
from statistical_test import graph_ranks

st. set_page_config(layout="wide") 
st.set_option('deprecation.showPyplotGlobalUse', False)

#plt.style.use('dark_background')
df = pd.read_csv('data/results.csv')
characteristics_df = pd.read_csv('data/characteristics.csv')
characteristics_df.reset_index()
characteristics_df.drop(columns=characteristics_df.columns[0], axis=1, inplace=True)
characteristics_df.columns = ['Name', 'NumOfTrainingSamples', 'NumOfTestingSample', 'NumOfSamples', 'SeqLength', 'NumOfClusters', 'Type']
characteristics_df = characteristics_df[['Name', 'NumOfSamples', 'SeqLength', 'NumOfClusters', 'Type']]


def plot_stat_plot(df, metric_name, methods_family, datasets):
    container_method = st.container()
    stat_methods_family = container_method.multiselect('Select a group of methods', sorted(methods_family), key='selector_stat_methods')
    
    df = df.loc[df['Datasets'].isin(datasets)][[method_g + '-' + metric_name for method_g in stat_methods_family]]
    df.insert(0, 'Datasets', datasets)

    if len(datasets) > 0:
        if len(stat_methods_family) > 1 and len(stat_methods_family) < 13:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df

                df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
                rank_df = rank_df.reset_index()
                rank_df.columns = ['Method Name', 'Average Rank']
                st.table(rank_df)

            stat_plots(df)
    

def plot_time_plot(measure_name):
    container_method = st.container()
    all_time_method = st.checkbox("Select all",key='all_time_method')
    if all_time_method: time_methods_family = container_method.multiselect('Select a group of methods', sorted(time_methods), sorted(time_methods), key='selector_time_methods_all')
    else: time_methods_family = container_method.multiselect('Select a group of methods', sorted(time_methods), key='selector_time_methods', default=['k-Shape', 'DTC', 'DTCR', 'SOM-VAE', 'SDCN', 'ClusterGAN'])


    if len(time_methods_family) > 0:
        time, time_text, x, y = get_bubble_data(time_methods_family, measure_name)

        fig = go.Figure(data=[go.Scatter(
            x=x,
            y=y,
            text=time_text,
            mode='markers',
            marker=dict(
                opacity=.7,
                color='red',
                line = dict(width=1, color = '#1f77b4'),
                size=time,
                sizemode='area',
                sizeref=2.*max(time)/(80.**2),
                sizemin=4
            )
        )])
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        fig.update_layout(showlegend=False, width=800, height=600, template="plotly_white", font=dict(
                                                        size=19,
                                                        color="black"))

        st.plotly_chart(fig)


def plot_misconceptions_plot(metric_name, datasets):
    tab4, tab5, tab6 = st.tabs(["Distance Measures", "Supervised vs. Unsupervised Tuning", "Deep Learning Methods"])
    with tab4:
        elastic_measures_list = ['PAM-ED', 'PAM-SBD', 'PAM-MSM', 'PAM-LCSS', 'PAM-TWED', 'PAM-SWALE', 'PAM-DTW', 'PAM-EDR', 'PAM-ERP']
        kernel_measures_list = ['KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK']

        container_method = st.container()
        all_elastic_measures = st.checkbox("Select all", key='all_elastic_measures')
        if all_elastic_measures: all_elastic_measures_family = container_method.multiselect('Select elastic measures', sorted(elastic_measures_list), sorted(elastic_measures_list), key='selector_all_elastic_measures')
        else: all_elastic_measures_family = container_method.multiselect('Select elastic measures', sorted(elastic_measures_list), key='selector_elastic_measures', default=['PAM-ED', 'PAM-MSM', 'PAM-TWED', 'PAM-DTW', 'PAM-ERP'])

        df = pd.read_csv('data/results.csv')
        df = df.loc[df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in all_elastic_measures_family]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(all_elastic_measures_family) > 1 and len(all_elastic_measures_family) < 13:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df

                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append(method[:-len(metric_name)-1])

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)
                
                stat_plots(df)
                


        container_method = st.container()
        all_kernel_measures = st.checkbox("Select all",key='all_kernel_measures')
        if all_kernel_measures: all_kernel_measures_family = container_method.multiselect('Select kernel measures', sorted(kernel_measures_list), sorted(kernel_measures_list), key='selector_all_kernel_measures')
        else: all_kernel_measures_family = container_method.multiselect('Select kernel measures', sorted(kernel_measures_list), key='selector_kernel_measures', default=['KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK'])
        
        df = pd.read_csv('data/results.csv')
        df = df.loc[df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in all_kernel_measures_family]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(all_kernel_measures_family) > 1 and len(all_kernel_measures_family) < 13:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df

                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append(method[:-len(metric_name)-1])

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)

                stat_plots(df)

    with tab5:
        unsupervised_list = ['k-Shape', 'PAM-Unsupervised-MSM', 'PAM-Unsupervised-LCSS', 'PAM-Unsupervised-TWED', 'PAM-Unsupervised-SWALE', 'PAM-Unsupervised-DTW', 'PAM-Unsupervised-EDR', 'PAM-Unsupervised-ERP']
        supervised_list = ['k-Shape', 'PAM-Supervised-MSM', 'PAM-Supervised-LCSS', 'PAM-Supervised-TWED', 'PAM-Supervised-SWALE', 'PAM-Supervised-DTW', 'PAM-Supervised-EDR', 'PAM-Supervised-ERP']

        container_method = st.container()
        all_supervised_measures = st.checkbox("Select all", key='all_supervised_measures')
        if all_supervised_measures: all_supervised_measures_family = container_method.multiselect('Select supervised elastic measures', sorted(supervised_list), sorted(supervised_list), key='selector_all_supervised_measures')
        else: all_supervised_measures_family = container_method.multiselect('Select supervised elastic measures', sorted(supervised_list), key='selector_supervised_measures', default=['k-Shape', 'PAM-Supervised-MSM', 'PAM-Supervised-TWED', 'PAM-Supervised-ERP'])
        
        df = pd.read_csv('data/supervised-unsupervised.csv')
        df = df.loc[df['Name'].isin(datasets)][[method_g + '-' + metric_name for method_g in all_supervised_measures_family]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(all_supervised_measures_family) > 1 and len(all_supervised_measures_family) < 13:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df

                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append(method[:-len(metric_name)-1])

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)

                stat_plots(df)

        container_method = st.container()
        all_unsupervised_measures = st.checkbox("Select all", key='all_unsupervised_measures')
        if all_unsupervised_measures: all_unsupervised_measures_family = container_method.multiselect('Select unsupervised elastic measures', sorted(unsupervised_list), sorted(unsupervised_list), key='selector_all_unsupervised_measures')
        else: all_unsupervised_measures_family = container_method.multiselect('Select unsupervised elastic measures', sorted(unsupervised_list), key='selector_unsupervised_measures', default=['k-Shape', 'PAM-Unsupervised-MSM', 'PAM-Unsupervised-TWED', 'PAM-Unsupervised-ERP'])
        
        df = pd.read_csv('data/supervised-unsupervised.csv')
        df = df.loc[df['Name'].isin(datasets)][[method_g + '-' + metric_name for method_g in all_unsupervised_measures_family]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(all_unsupervised_measures_family) > 1 and len(all_unsupervised_measures_family) < 13:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df

                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append(method[:-len(metric_name)-1])

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)

                stat_plots(df)
    
    with tab6:
        dl_list = ['DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE']
        clsc_list = ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape', 'KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK', 'SC_GAK', 'SC_KDTW', 'SC_RBF', 'SC_SINK', 
           'DBSCAN-ED', 'DBSCAN-MSM', 'DBSCAN-SBD', 'DP-ED', 'DP-MSM', 'DP-SBD', 'OPTICS-ED', 'OPTICS-MSM', 'OPTICS-SBD', 
           'PAM-ED', 'PAM-SBD', 'PAM-MSM', 'PAM-LCSS', 'PAM-TWED', 'PAM-SWALE', 'PAM-DTW', 'PAM-EDR', 'PAM-ERP',
           'AGG-A-ED', 'AGG-A-MSM', 'AGG-A-SBD', 'AGG-C-ED', 'AGG-C-MSM', 'AGG-C-SBD', 'AGG-S-ED', 'AGG-S-MSM', 'AGG-S-SBD', 'BIRCH',
           'SS-DTW', 'FeatTS', 'UShapelets-50', 'GMM', 'AP-ED', 'AP-MSM', 'AP-SBD',
           'AR-COEFF', 'AR-PVAL', 'CATCH22', 'ES-COEFF', 'LPCC']
        
        container_method = st.container()
        all_dl_measures = st.checkbox("Select all", key='all_dl_measures')
        if all_dl_measures: all_dl_measures_family = container_method.multiselect('Select deep learning methods', sorted(dl_list), sorted(dl_list), key='selector_all_dl_measures')
        else: all_dl_measures_family = container_method.multiselect('Select Deep Learning Methods', sorted(dl_list), key='selector_dl_measures', default=['DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE'])

        clsc_measures = st.selectbox('Select a classical method', tuple(clsc_list), index=3)

        selected_methods = all_dl_measures_family + [clsc_measures]

        df = pd.read_csv('data/results.csv')
        df = df.loc[df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in selected_methods]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(selected_methods) > 1 and len(selected_methods) < 14:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df

                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append(method[:-len(metric_name)-1])

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)

                stat_plots(df)


def plot_box_plot(df, measure_name, methods_family, datasets, scale='linear'):
    if len(df.columns) > 0:
        tab1, tab2 = st.tabs(["Box Plot", "Scatter Plot"])
        with tab1:
            fig = go.Figure()
            #c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, len(df.columns[1:]))]
            for i, cols in enumerate(df.columns[1:]):
                fig.add_trace(go.Box(y=df[cols], name=cols[:-len(measure_name)-1],
                                     marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
            fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
            
            fig.update_xaxes(tickfont_size=16)
            fig.update_yaxes(tickfont_size=16)
            #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
            st.plotly_chart(fig)

            cols_list = []
            for i, col in enumerate(df.columns):
                if i > 0:
                    cols_list.append(col[:-len(measure_name)-1])
                else:
                    cols_list.append(col)

            df.columns = cols_list
            AgGrid(df)
        with tab2:
            option1 = st.selectbox(
                'Method 1',
                tuple(methods_family))
            methods_family = methods_family[1:] + methods_family[:1]
            option2 = st.selectbox(
                'Method 2',
                tuple(methods_family))

            if len(methods_family) > 0 and len(datasets) > 0:
                fig = go.FigureWidget()
                trace1 = fig.add_scattergl(x=df[option1], y=df[option2], mode='markers', name='(Method 1, Method 2)',  text=datasets,
                                        textposition="bottom center",
                                        marker = dict(size=10,
                                                    opacity=.7,
                                                    color='red',
                                                    line = dict(width=1, color = '#1f77b4')
                                                    ))
                fig.add_trace(go.Scatter(
                                    x=[min(min(df[option1])+1e-4, min(df[option2])+1e-4), max(max(df[option1])+1e-4, max(df[option2])+1e-4)],
                                    y=[min(min(df[option1])+1e-4, min(df[option2])+1e-4), max(max(df[option1])+1e-4, max(df[option2])+1e-4)],
                                    name="X=Y"
                                ))
                trace2 = fig.add_histogram(x=df[option1], name='x density', marker=dict(color='#1f77b4', opacity=0.7),
                                    yaxis='y2'
                                    )
                trace3 = fig.add_histogram(y=df[option2], name='y density', marker=dict(color='#1f77b4', opacity=0.7), 
                                    xaxis='x2'
                                    )
                fig.layout = dict(xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                                yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                                xaxis_title=option1, yaxis_title=option2,
                                showlegend=False,
                                margin=dict(t=50),
                                hovermode='closest',
                                bargap=0,
                                xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                                yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                                height=600,
                                )

                def do_zoom(layout, xaxis_range, yaxis_range):
                    inds = ((xaxis_range[0] <= df[option1]) & (df[option1] <= xaxis_range[1]) &
                            (yaxis_range[0] <= df[option2]) & (df[option2] <= yaxis_range[1]))

                    with fig.batch_update():
                        trace2.x = df[option1][inds]
                        trace3.y = df[option2][inds]
                    
                fig.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')
                fig.update_xaxes(tickfont_size=16)
                fig.update_yaxes(tickfont_size=16)
                st.plotly_chart(fig)


def plot_classwise(all_df, metric_name, datasets):
    tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(["Partitional", "Kernel", "Density", "Hierarchical", "Distribution", "Model and Feature", "Deep Learning"])
    with tab7:
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape']]]
        df = df[df.mean().sort_values().index]
        fig = go.Figure()
        for i, cols in enumerate(df.columns[:]):
            fig.add_trace(go.Box(y=df[cols], name=cols[:-len(metric_name)-1],
                                            marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
        fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
        st.plotly_chart(fig)

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)
        
        AgGrid(df)

    with tab8:
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in ['KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK', 'SC_GAK', 'SC_KDTW', 'SC_RBF', 'SC_SINK']]]
        df = df[df.mean().sort_values().index]
        fig = go.Figure()
        for i, cols in enumerate(df.columns[:]):
            fig.add_trace(go.Box(y=df[cols], name=cols[:-len(metric_name)-1],
                                            marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
        fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
        st.plotly_chart(fig)

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)

        AgGrid(df)

    with tab9:
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in ['DBSCAN-ED', 'DBSCAN-MSM', 'DBSCAN-SBD', 'DP-ED', 'DP-MSM', 'DP-SBD', 'OPTICS-ED', 'OPTICS-MSM', 'OPTICS-SBD']]]
        df = df[df.mean().sort_values().index]
        fig = go.Figure()
        for i, cols in enumerate(df.columns[:]):
            fig.add_trace(go.Box(y=df[cols], name=cols[:-len(metric_name)-1],
                                            marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
        fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
        st.plotly_chart(fig)

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)

        AgGrid(df)

    with tab10:
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in ['AGG-A-ED', 'AGG-A-MSM', 'AGG-A-SBD', 'AGG-C-ED', 'AGG-C-MSM', 'AGG-C-SBD', 'AGG-S-ED', 'AGG-S-MSM', 'AGG-S-SBD', 'BIRCH']]]
        df = df[df.mean().sort_values().index]
        fig = go.Figure()
        for i, cols in enumerate(df.columns[:]):
            fig.add_trace(go.Box(y=df[cols], name=cols[:-len(metric_name)-1],
                                            marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
        fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
        st.plotly_chart(fig)

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)

        AgGrid(df)

    with tab11:
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in ['GMM', 'AP-ED', 'AP-MSM', 'AP-SBD']]]
        df = df[df.mean().sort_values().index]
        fig = go.Figure()
        for i, cols in enumerate(df.columns[:]):
            fig.add_trace(go.Box(y=df[cols], name=cols[:-len(metric_name)-1],
                                            marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
        fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
        st.plotly_chart(fig)

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)

        AgGrid(df)

    with tab12:
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in ['AR-COEFF', 'AR-PVAL', 'CATCH22', 'ES-COEFF', 'LPCC']]]
        df = df[df.mean().sort_values().index]
        fig = go.Figure()
        for i, cols in enumerate(df.columns[:]):
            fig.add_trace(go.Box(y=df[cols], name=cols[:-len(metric_name)-1],
                                            marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
        fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
        st.plotly_chart(fig)

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)

        AgGrid(df)

    with tab13:
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in ['DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE']]]
        df = df[df.mean().sort_values().index]
        fig = go.Figure()
        for i, cols in enumerate(df.columns[:]):
            fig.add_trace(go.Box(y=df[cols], name=cols[:-len(metric_name)-1],
                                            marker=dict(
                                            opacity=1,
                                            color='rgb(8,81,156)',
                                            outliercolor='rgba(219, 64, 82, 0.6)',
                                            line=dict(
                                                outliercolor='rgba(219, 64, 82, 0.6)',
                                                outlierwidth=2)),
                                        line_color='rgb(8,81,156)'
                                    ))
        fig.update_layout(showlegend=False, 
                              width=1290, 
                              height=600, 
                              template="plotly_white", 
                              font=dict(
                                        size=39,
                                        color="black"))
        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
        st.plotly_chart(fig)

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)

        AgGrid(df)


def plot_ablation(all_df, metric_name, datasets):
    tab14, tab15, tab16 = st.tabs(["Architectures", "Pretext Loss", "Clustering Loss"])
    with tab14:
        arch_list = ['BI_LSTM', 'BI_GRU', 'RES_CNN', 'D_RNN', 'BI_RNN', 'D_CNN', 'MLP', 'S_CNN', 'BI_RNN+ATTN']
        container_method = st.container()
        all_archs = st.checkbox("Select all", key='all_archs_measures')
        if all_archs: all_archs_family = container_method.multiselect('Select architectures', sorted(arch_list), sorted(arch_list), key='selector_all_archs')
        else: all_archs_family = container_method.multiselect('Select architectures', sorted(arch_list), key='selector_archs', default=arch_list)

        df = pd.read_csv('data/ablation.csv')
        df = df.loc[df['Name'].isin(datasets)][[method_g + '-' + metric_name for method_g in all_archs_family]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(all_archs_family) > 1 and len(all_archs_family) < 14:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df

                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append(method[:-len(metric_name)-1] + '+NONE+REC')

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)

                stat_plots(df)
    with tab15:
        arch_list1 = ['NONE_TRPLT', 'NONE_MREC', 'NONE_VAE', 'NONE_REC', 'NONE_CNRV']
        container_method = st.container()
        all_loss1 = st.checkbox("Select all", key='all_loss1_measures')
        if all_loss1: all_loss1_family = container_method.multiselect('Select pretext losses', sorted(arch_list1), sorted(arch_list1), key='selector_all_loss1')
        else: all_loss1_family = container_method.multiselect('Select pretext losses', sorted(arch_list1), key='selector_loss1', default=arch_list1)

        df = pd.read_csv('data/ablation.csv')
        df = df.loc[df['Name'].isin(datasets)][[method_g + '-' + metric_name for method_g in all_loss1_family]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(all_loss1_family) > 1 and len(all_loss1_family) < 14:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df
                    
                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append('RES-CNN+' + method[:-len(metric_name)-1].replace('_', '+'))

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)

                stat_plots(df)

        arch_list2 = ['IDEC_TRPLT',  'IDEC_REC', 'IDEC_MREC', 'IDEC_VAE', 'IDEC_CNRV']
        container_method = st.container()
        all_loss2 = st.checkbox("Select all", key='all_loss2_measures')
        if all_loss2: all_loss2_family = container_method.multiselect('Select pretext losses', sorted(arch_list2), sorted(arch_list2), key='selector_all_loss2')
        else: all_loss2_family = container_method.multiselect('Select pretext losses', sorted(arch_list2), key='selector_loss2', default=arch_list2)

        df = pd.read_csv('data/ablation.csv')
        df = df.loc[df['Name'].isin(datasets)][[method_g + '-' + metric_name for method_g in all_loss2_family]]
        df.insert(0, 'Datasets', datasets)

        if len(datasets) > 0:
            if len(all_loss2_family) > 1 and len(all_loss2_family) < 14:
                def stat_plots(df_toplot):
                    def cd_diagram_process(df, rank_ascending=False):
                        df = df.rank(ascending=rank_ascending, axis=1)
                        return df
                    
                    df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                    rank_ri_df  = cd_diagram_process(df_toplot)
                    rank_df = rank_ri_df.mean().sort_values()

                    names = []
                    for method in rank_df.index.values:
                        names.append('RES-CNN+' + method[:-len(metric_name)-1].replace('_', '+'))

                    avranks =  rank_df.values
                    cd = compute_CD(avranks, 128, "0.1")
                    graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                    fig = plt.show()
                    st.pyplot(fig)

                stat_plots(df)
    with tab16:
        dl_list = ['DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE']
        container_method = st.container()
        all_dl1 = st.checkbox("Select all", key='all_dl1_measures')
        if all_dl1: dl1_list_family = container_method.multiselect('Select clustering losses', sorted(dl_list), sorted(dl_list), key='selector_dl_list1')
        else: dl1_list_family = container_method.multiselect('Select clustering losses', sorted(dl_list), key='selector_dl1', default=dl_list)
        
        df = all_df.loc[all_df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in dl1_list_family]]
        df = df[df.mean().sort_values().index]

        if len(datasets) > 0:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df
                
                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method[:-len(metric_name)-1])

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, "0.1")
                graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                fig = plt.show()
                st.pyplot(fig)
            stat_plots(df)
                

def generate_dataframe(df, datasets, methods_family, metric_name):
    df = df.loc[df['Dataset'].isin(datasets)][[method_g + '-' + metric_name for method_g in methods_family]]
    df = df[df.mean().sort_values().index]
    df.insert(0, 'Datasets', datasets)
    return df
      
    
with st.sidebar:
    st.markdown('# OdysseyEngine') 
    metric_name = st.selectbox('Pick an assessment measure', list_measures) 
    
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
#tab_desc, tab_acc, tab_time, tab_stats, tab_analysis, tab_misconceptions, tab_ablation, tab_dataset, tab_method, tab_reference = st.tabs(["Description", "Evaluation", "Runtime", "Statistical Tests", "Comparative Analysis", "Misconceptions", "DNN Ablation Analysis", "Datasets", "Methods", "References"])  
tab_desc, tab_acc, tab_time, tab_stats, tab_analysis, tab_misconceptions, tab_ablation, tab_dataset, tab_method = st.tabs(["Description", "Evaluation", "Runtime", "Statistical Tests", "Comparative Analysis", "Misconceptions", "DNN Ablation Analysis", "Datasets", "Methods"])  

with tab_desc:
    st.markdown('# Odyssey Engine')
    st.markdown(description_intro1)
    st.columns(3)[0].image('./data/summary.png', width=1000, caption='Overview of Odyssey’s architecture.')
    st.markdown(description_intro2)

with tab_acc:
    st.markdown('# Evaluation')
    st.markdown('Overall evaluation of time-series clustering algorithms used over 128 datasets. Measure used: {}'.format(metric_name))
    df_toplot = generate_dataframe(df, datasets, methods_family, metric_name)
    plot_box_plot(df_toplot, measure_name=metric_name, methods_family=methods_family, datasets=datasets)
    
with tab_time:
    st.markdown('#  Runtime')
    plot_time_plot(metric_name)

with tab_stats:
    st.markdown('# Statistical Tests')
    df_toplot = generate_dataframe(df, datasets, methods_family, metric_name)
    plot_stat_plot(df_toplot, metric_name, methods_family, datasets)

with tab_misconceptions:
    st.markdown('# Misconceptions')
    plot_misconceptions_plot(metric_name, datasets)

with tab_analysis:
    st.markdown('# Comparative Analysis')
    plot_classwise(df, metric_name, datasets)

with tab_ablation:
    st.markdown('# DNN Ablation Analysis')
    plot_ablation(df, metric_name, datasets)

with tab_dataset:
    st.markdown('# Dataset Description')
    st.markdown(text_description_dataset)
    AgGrid(characteristics_df)

with tab_method:
    st.markdown('# Time-Series Clustering Methods')
    st.markdown(text_description_models1, unsafe_allow_html=True)
    
    background = Image.open('./data/taxonomy.png')
    col1, col2, col3 = st.columns([2, 5, 0.2])
    col2.image(background, width=1000, caption='Taxonomy of time-series clustering methods in Odyssey.')

    st.columns(3)[0].image('./data/taxonomy.png', width=1000, caption='Taxonomy of time-series clustering methods in Odyssey.')
    st.markdown(text_description_models2, unsafe_allow_html=True)

#with tab_reference:
#    st.markdown('# References')
#    st.markdown(list_references, unsafe_allow_html=True)
