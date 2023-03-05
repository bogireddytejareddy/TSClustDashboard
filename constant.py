import pandas as pd


method_class_dict = {'Partitional': ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape'],
                     'Kernel': ['KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK', 'SC_GAK', 'SC_KDTW', 'SC_RBF', 'SC_SINK'],
                     'Density': ['DBSCAN-ED', 'DBSCAN-MSM', 'DBSCAN-SBD', 'DP-ED', 'DP-MSM', 'DP-SBD', 'OPTICS-ED', 'OPTICS-MSM', 'OPTICS-SBD'],
                     'Hierarchical': ['AGG-A-ED', 'AGG-A-MSM', 'AGG-A-SBD', 'AGG-C-ED', 'AGG-C-MSM', 'AGG-C-SBD', 'AGG-S-ED', 'AGG-S-MSM', 'AGG-S-SBD', 'BIRCH'],
                     'Semi-Supervised': ['SS-DTW', 'FeatTS'],
                     'Shapelet': ['UShapelets-50'],
                     'Distribution': ['GMM', 'AP-ED', 'AP-MSM', 'AP-SBD'],
                     'Model-and-Feature': ['AR-COEFF', 'AR-PVAL', 'CATCH22', 'ES-COEFF', 'LPCC'],
                     'Deep-Learning': ['DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'RES_CNN+CNRV+IDEC', 'RES_CNN+CNRV+NONE', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE']
                     }

classes = ['Partitional', 'Kernel', 'Density', 'Hierarchical', 'Semi-Supervised', 'Shapelet', 'Distribution', 'Model-and-Feature', 'Deep-Learning']

methods = ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape', 'KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK', 'SC_GAK', 'SC_KDTW', 'SC_RBF', 'SC_SINK', 
           'DBSCAN-ED', 'DBSCAN-MSM', 'DBSCAN-SBD', 'DP-ED', 'DP-MSM', 'DP-SBD', 'OPTICS-ED', 'OPTICS-MSM', 'OPTICS-SBD', 
           'PAM-ED', 'PAM-SBD', 'PAM-MSM', 'PAM-LCSS', 'PAM-TWED', 'PAM-SWALE', 'PAM-DTW', 'PAM-EDR', 'PAM-ERP',
           'AGG-A-ED', 'AGG-A-MSM', 'AGG-A-SBD', 'AGG-C-ED', 'AGG-C-MSM', 'AGG-C-SBD', 'AGG-S-ED', 'AGG-S-MSM', 'AGG-S-SBD', 'BIRCH',
           'SS-DTW', 'FeatTS', 'UShapelets-50', 'GMM', 'AP-ED', 'AP-MSM', 'AP-SBD',
           'AR-COEFF', 'AR-PVAL', 'CATCH22', 'ES-COEFF', 'LPCC',
           'DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'RES_CNN+CNRV+IDEC', 'RES_CNN+CNRV+NONE', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE']

time_methods = ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape', 'KKM_SINK', 'SC_SINK', 'BIRCH','AP_MSM', 'AGG-C_MSM', 
'DBSCAN_MSM', 'OPTICS_MSM','DensityPeaks_MSM', 'UShapelet_50%', 'FeatTS', 'SS-DTW', 'GMM', 'AR-COEFF', 'DEC', 'IDEC','DTC', 'DTCR', 'SOM-VAE', 
'DCN', 'DEPICT', 'SDCN', 'ClusterGAN', 'VADE', 'RES_CNN+CNRV+IDEC', 'RES_CNN+CNRV+NONE']

time_methods_dict = {'k-AVG': 4963.54138, 
           'k-Shape': 25509.62824, 
           'k-SC': 237536.667, 
           'k-DBA': 3410782.915, 
           'KKM_SINK': 243855.2651, 
           'SC_SINK': 243064.3738, 
           'BIRCH': 227.2080344,
           'AP_MSM': 431.6869359+1000000, 
           'AGG-C_MSM': 7.314567089+1000000, 
           'DBSCAN_MSM': 4.462136745+1000000, 
           'OPTICS_MSM': 994.0497878+1000000,
           'DensityPeaks_MSM': 596.108005+1000000, 
           'UShapelet_50%': 304902.1349, 
           'FeatTS': 87117.32219, 
           'SS-DTW': 493158.7731, 
           'GMM': 588.6741066, 
           'AR-COEFF': 5501.5588374, 
           'DEC': 6362.423854, 
           'IDEC': 7437.43861,
           'DTC': 86722.03318, 
           'DTCR': 107108.6329, 
           'SOM-VAE': 78389.82785, 
           'DCN': 23651.79199, 
           'DEPICT': 1.39E+04,
           'SDCN': 4.92E+04,
           'ClusterGAN': 8.05E+04, 
           'VADE': 1.42E+04, 
           'RES_CNN+CNRV+IDEC': 251273.1364, 
           'RES_CNN+CNRV+NONE': 243835.6978}

performance_time_methods_dict_ri = {'k-AVG': 0.7001, 
                                    'k-Shape': 0.7268, 
                                    'k-SC': 0.6282, 
                                    'k-DBA': 0.6791, 
                                    'KKM_SINK': 0.7287, 
                                    'SC_SINK': 0.7321, 
                                    'BIRCH': 0.7123,
                                    'AP_MSM': 0.7289, 
                                    'AGG-C_MSM': 0.7058, 
                                    'DBSCAN_MSM': 0.4310, 
                                    'OPTICS_MSM': 0.6037,
                                    'DensityPeaks_MSM': 0.5640, 
                                    'UShapelet_50%': 0.5718, 
                                    'FeatTS': 0.7203, 
                                    'SS-DTW': 0.6307, 
                                    'GMM': 0.7165, 
                                    'AR-COEFF': 0.6885, 
                                    'DEC': 0.7090, 
                                    'IDEC': 0.7159,
                                    'DTC': 0.7085, 
                                    'DTCR': 0.6832, 
                                    'SOM-VAE': 0.6457, 
                                    'DCN': 0.5716, 
                                    'DEPICT': 0.7111, 
                                    'SDCN': 0.7104,
                                    'ClusterGAN': 0.7082, 
                                    'VADE': 0.7027, 
                                    'RES_CNN+CNRV+IDEC': 0.7337, 
                                    'RES_CNN+CNRV+NONE': 0.7393}

performance_time_methods_dict_ari = {'k-AVG': 0.1811, 
                                    'k-Shape': 0.2528, 
                                    'k-SC': 0.1788, 
                                    'k-DBA': 0.2021, 
                                    'KKM_SINK': 0.2553, 
                                    'SC_SINK': 0.2661, 
                                    'BIRCH': 0.2305,
                                    'AP_MSM': 0.2204, 
                                    'AGG-C_MSM': 0.2415, 
                                    'DBSCAN_MSM': 0.1019, 
                                    'OPTICS_MSM': 0.0705,
                                    'DensityPeaks_MSM': 0.0751, 
                                    'UShapelet_50%': 0.1510, 
                                    'FeatTS': 0.2823, 
                                    'SS-DTW': 0.1383, 
                                    'GMM': 0.2193, 
                                    'AR-COEFF': 0.1159, 
                                    'DEC': 0.1935, 
                                    'IDEC': 0.2150,
                                    'DTC': 0.2132, 
                                    'DTCR': 0.1392, 
                                    'SOM-VAE': 0.0976, 
                                    'DCN': 0.0444, 
                                    'DEPICT': 0.1900, 
                                    'SDCN': 0.2000,
                                    'ClusterGAN': 0.2100, 
                                    'VADE': 0.1734, 
                                    'RES_CNN+CNRV+IDEC': 0.2565, 
                                    'RES_CNN+CNRV+NONE': 0.2702}

performance_time_methods_dict_nmi = {'k-AVG': 0.2724, 
                                    'k-Shape': 0.3362, 
                                    'k-SC': 0.2492, 
                                    'k-DBA': 0.2776, 
                                    'KKM_SINK': 0.3461, 
                                    'SC_SINK': 0.3513, 
                                    'BIRCH': 0.3483,
                                    'AP_MSM': 0.4269, 
                                    'AGG-C_MSM': 0.3712, 
                                    'DBSCAN_MSM': 0.3293, 
                                    'OPTICS_MSM': 0.3505,
                                    'DensityPeaks_MSM': 0.3293, 
                                    'UShapelet_50%': 0.2385, 
                                    'FeatTS': 0.3229, 
                                    'SS-DTW': 0.2427, 
                                    'GMM': 0.3067, 
                                    'AR-COEFF': 0.1881, 
                                    'DEC': 0.2790, 
                                    'IDEC': 0.2967,
                                    'DTC': 0.2985, 
                                    'DTCR': 0.2184, 
                                    'SOM-VAE': 0.1804, 
                                    'DCN': 0.1097, 
                                    'DEPICT': 0.2743, 
                                    'SDCN': 0.2884,
                                    'ClusterGAN': 0.2965, 
                                    'VADE': 0.2605, 
                                    'RES_CNN+CNRV+IDEC': 0.3709, 
                                    'RES_CNN+CNRV+NONE': 0.3832}


def find_methods(selected_classes):
    if len(selected_classes) == 0:
        return methods
    else:
        methods_list = []
        for class_name in selected_classes:
            for m in method_class_dict[class_name]:
                methods_list.append(m)

        return methods_list


def get_bubble_data(methods, measure_name):
    time = []
    for m in methods:
        time.append(time_methods_dict[m])
    
    time_text = []
    for t in time:
        time_text.append('Runtime: ' + str(t))

    y = []
    for m in methods:
        if measure_name == 'RI':
            y.append(performance_time_methods_dict_ri[m])
        elif measure_name == 'ARI':
            y.append(performance_time_methods_dict_ari[m])
        elif measure_name == 'NMI':
            y.append(performance_time_methods_dict_nmi[m])

    return time, time_text, methods, y



def find_datasets(clusters_size, lengths_size, types):
    df = pd.read_csv('./data/characteristics.csv')

    def f(row):
        if row['SeqLength'] < 100:
            val = 'VERY-SMALL(<100)'
        elif row['SeqLength'] < 300:
            val = 'SMALL(<300)'
        elif row['SeqLength'] < 500:
            val = 'MEDIUM(<500)'
        elif row['SeqLength'] < 1000:
            val = 'LARGE(<1000)'
        else:
            val = 'VERY-LARGE(>1000)'
        return val

    df['LengthLabel'] = df.apply(f, axis=1)

    def f(row):
        if row['NumClasses'] < 10:
            val = 'VERY-SMALL(<10)'
        elif row['NumClasses'] < 20:
            val = 'SMALL(<20)'
        elif row['NumClasses'] < 40:
            val = 'MEDIUM(<40)'
        else:
            val = 'LARGE(>40)'
        return val

    df['ClustersLabel'] = df.apply(f, axis=1)

    if len(clusters_size) > 0:
        df = df.loc[df['ClustersLabel'].isin(clusters_size)] 
    if len(lengths_size) > 0:
        df = df.loc[df['LengthLabel'].isin(lengths_size)]
    if len(types) > 0:
        df = df.loc[df['TYPE'].isin(types)]

    return list(df['Name'].values)



list_type = ['AUDIO','DEVICE','ECG','EOG','EPG','HEMODYNAMICS','IMAGE','MOTION','OTHER','SENSOR','SIMULATED','SOUND','SPECTRO','TRAFFIC']

list_seq_length = ['VERY-SMALL(<100)', 'SMALL(<300)', 'MEDIUM(<500)', 'LARGE(<1000)', 'VERY-LARGE(>1000)']

list_num_clusters = ['VERY-SMALL(<10)', 'SMALL(<20)', 'MEDIUM(<40)', 'LARGE(>40)']

list_measures = ['RI','ARI', 'NMI']

list_length = [16,32,64,128,256,512,768,1024]

oracle = ['GENIE','MORTY']


description_intro = f"""
Clustering is one of the most popular time-series tasks because it enables unsupervised data exploration and often serves as a subroutine or 
preprocessing step for other tasks. Despite being the subject of active research for decades across disciplines, only limited efforts focused on 
benchmarking clustering methods for time series. Therefore, we comprehensively evaluate 59 time-series clustering methods spanning 9 different 
classes from the data mining, machine learning, and deep learning literature.

## Contributors

* [Teja Reddy](https://github.com/bogireddytejareddy) (Exelon Utilities)
* [John Paparrizos](https://www.paparrizos.org/) (Ohio State University)

## Datasets

To ease reproducibility, we share our results over an established benchmarks:

* The UCR Univariate Archive, which contains 128 univariate time-series datasets.
    * Download all 128 preprocessed datasets [here](https://www.thedatum.org/datasets/UCR2022_DATASETS.zip).

For the preprocessing steps check [here](https://github.com/thedatumorg/UCRArchiveFixes).

## Models

We complied a Python [library](https://github.com/johnpaparrizos/TSClusteringEval) with state-of-the-art time-series clustering models 
so that all the comparisons are performed under the same framework for a consistent evaluation in terms of both performance and efficiency.

"""


text_description_dataset = f"""
We conduct our evaluation using the UCR Time-Series Archive, the largest collection of class labeled time series datasets. 
The archive consists of a collection of 128 datasets sourced from different sensor readings while performing diverse tasks from multiple 
domains. All datasets in the archive span between 40 to 24000 time-series and have lengths varying from 15 to 2844. Datasets are z-normalized, 
and each time-series in the dataset belongs to only one class. There is a small subset of datasets in the archive containing missing values and 
varying lengths. We employ linear interpolation to fill the missing values and resample shorter time series to reach the longest time series 
in each dataset.
"""



text_description_models=f"""
We have implemented x methods from 9 classes of time-series clustering methods proposed for univariate time series. The following table 
lists the methods considered:


### <span style='color:Tomato'>Partitional Clustering</span>

| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
|𝑘-AVG              |ED                                 |
|𝑘-Shape              |SBD                                 |
|𝑘-SC              |STID                                 |
|𝑘-DBA              |DTW                                 |
|PAM              |MSM                                 |
|PAM              |TWED                                 |
|PAM              |ERP                                 |
|PAM              |SBD                                 |
|PAM              |SWALE                                 |
|PAM              |DTW                                 |
|PAM              |EDR                                 |
|PAM              |LCSS                                 |
|PAM              |ED                                 |


### <span style='color:Tomato'>Kernel Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| KKM              | SINK                                 |
| KKM              | GAK                                 |
| KKM              | KDTW                                 |
| KKM              | RBF                                 |
| SC              | SINK                                 |
| SC              | GAK                                 |
| SC              | KDTW                                 |
| SC              | RBF                                 |


### <span style='color:Tomato'>Density Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| DBSCAN              | ED                                 |
| DBSCAN              | SBD                                 |
| DBSCAN              | MSM                                 |
| DP              | ED                                 |
| DP              | SBD                                 |
| DP              | MSM                                 |
| OPTICS              | ED                                 |
| OPTICS              | SBD                                 |
| OPTICS              | MSM                                 |


### <span style='color:Tomato'>Hierarchical Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| AGG              | ED                                 |
| AGG              | SBD                                 |
| AGG              | MSM                                 |
| BIRCH              | -                                 |


### <span style='color:Tomato'>Distribution Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| AP              | ED                                 |
| AP              | SBD                                 |
| AP              | MSM                                 |
| GMM              | -                                 |


### <span style='color:Tomato'>Shapelet Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| UShapelet              | -                                 |
| LDPS              | -                                 |
| USLM              | -                                 |


### <span style='color:Tomato'>Semi-Supervised Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| FeatTS              | -                                 |
| SS-DTW              | -                                 |


### <span style='color:Tomato'>Model and Feature Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| 𝑘-AVG             | AR-COEFF                                 |
| 𝑘-AVG             | CATCH22                                 |
| 𝑘-AVG             | LPCC                                 |
| 𝑘-AVG             | AR-PVAL                                 |
| 𝑘-AVG             | ES-COEFF                                 |


### <span style='color:Tomato'>Deep Learning Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> |
|:------------------|:----------------------------------|
| IDEC            | -                                 |
| DEC            | -                                 |
| DTC            | -                                 |
| DTCR            | -                                 |
| SOM-VAE            | -                                 |
| DEPICT            | -                                 |
| SDCN            | -                                 |
| ClusterGAN            | -                                 |
| VADE            | -                                 |
| DCN            | -                                 |


** Note: Results for LDPS and USLM methods are not mentioned because of thier infeasible runtimes on large datasets.

"""

text_description_MS = f"""
We consider 16 time series classification (TSC) algorithms used as model selection. The following table lists and describes the methods considered:

| TSC  (as model seleciton)  | Description|
|:--|:---------:|
| SVC | maps training examples to points in space to maximize the gap between the two categories. |
| Bayes | uses Bayes’ theorem to predict the class of a new data point using the posterior probabilities for each class. |
| MLP | consists of multiple layers of interconnected neurons. |
| QDA | is a discriminant analysis algorithm for classification problems. |
| Adaboost | is a meta-algorithm using boosting technique with weak classifiers. |
| Decision Tree | is a tree-based approach that splits data points into different leaves based on features. |
| Random Forest  | is an ensemble Decision Trees fed with random samples (with replacement) of the training set and random set of features. |
| kNN | assigns the most common class among its k nearest neighbors. |
| Rocket | transforms input time series using a small set of convolutional kernels, and uses the transformed features to train a linear classifier. |
| ConvNet  | uses convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. |
| ResNet | is a ConvNet with residual connections between convolutional block. |
| InceptionTime | is a combination of ResNets with kernels of multiple sizes. |
| SIT-conv | is a transformer architecture with a convolutional layer as input. |
| SIT-linear | is a transformer architecture for which the time series is divided into non-overlapping patches and linearly projected into the embedding space. |
| SIT-stem | is a transformer architecture with convolutional layers with increasing dimensionality as input. |
| SIT-stem-ReLU | is similar to SIT-stem but with Scaled ReLU. |
"""

