import pandas as pd




def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd



method_class_dict = {'Partitional': ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape', 'PAM-ED', 'PAM-SBD', 'PAM-MSM', 'PAM-LCSS', 'PAM-TWED', 'PAM-SWALE', 'PAM-DTW', 'PAM-EDR', 'PAM-ERP'],
                     'Kernel': ['KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK', 'SC_GAK', 'SC_KDTW', 'SC_RBF', 'SC_SINK'],
                     'Density': ['DBSCAN-ED', 'DBSCAN-MSM', 'DBSCAN-SBD', 'DP-ED', 'DP-MSM', 'DP-SBD', 'OPTICS-ED', 'OPTICS-MSM', 'OPTICS-SBD'],
                     'Hierarchical': ['AGG-A-ED', 'AGG-A-MSM', 'AGG-A-SBD', 'AGG-C-ED', 'AGG-C-MSM', 'AGG-C-SBD', 'AGG-S-ED', 'AGG-S-MSM', 'AGG-S-SBD', 'BIRCH'],
                     'Semi-Supervised': ['SS-DTW', 'FeatTS'],
                     'Shapelet': ['UShapelets-50'],
                     'Distribution': ['GMM', 'AP-ED', 'AP-MSM', 'AP-SBD'],
                     'Model-and-Feature': ['AR-COEFF', 'AR-PVAL', 'CATCH22', 'ES-COEFF', 'LPCC'],
                     'Deep-Learning': ['DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE']
                     }

classes = ['Partitional', 'Kernel', 'Density', 'Hierarchical', 'Semi-Supervised', 'Shapelet', 'Distribution', 'Model-and-Feature', 'Deep-Learning']

methods = ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape', 'KKM_GAK', 'KKM_KDTW', 'KKM_RBF', 'KKM_SINK', 'SC_GAK', 'SC_KDTW', 'SC_RBF', 'SC_SINK', 
           'DBSCAN-ED', 'DBSCAN-MSM', 'DBSCAN-SBD', 'DP-ED', 'DP-MSM', 'DP-SBD', 'OPTICS-ED', 'OPTICS-MSM', 'OPTICS-SBD', 
           'PAM-ED', 'PAM-SBD', 'PAM-MSM', 'PAM-LCSS', 'PAM-TWED', 'PAM-SWALE', 'PAM-DTW', 'PAM-EDR', 'PAM-ERP',
           'AGG-A-ED', 'AGG-A-MSM', 'AGG-A-SBD', 'AGG-C-ED', 'AGG-C-MSM', 'AGG-C-SBD', 'AGG-S-ED', 'AGG-S-MSM', 'AGG-S-SBD', 'BIRCH',
           'SS-DTW', 'FeatTS', 'UShapelets-50', 'GMM', 'AP-ED', 'AP-MSM', 'AP-SBD',
           'AR-COEFF', 'AR-PVAL', 'CATCH22', 'ES-COEFF', 'LPCC',
           'DCN', 'DEC', 'IDEC', 'DEPICT', 'DTC', 'DTCR', 'SDCN', 'SOM_VAE', 'ClusterGAN', 'VADE']

time_methods = ['k-AVG', 'k-DBA', 'k-SC', 'k-Shape', 'KKM_SINK', 'SC_SINK', 'BIRCH','AP_MSM', 'AGG-C_MSM', 
'DBSCAN_MSM', 'OPTICS_MSM','DensityPeaks_MSM', 'UShapelet_50%', 'FeatTS', 'SS-DTW', 'GMM', 'AR-COEFF', 'DEC', 'IDEC','DTC', 'DTCR', 'SOM-VAE', 
'DCN', 'DEPICT', 'SDCN', 'ClusterGAN', 'VADE']

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
    
    import numpy as np
    
    idxs = np.argsort(time)
    time = np.array(time)[idxs]
    time_text = np.array(time_text)[idxs]
    methods = np.array(methods)[idxs]
    y = np.array(y)[idxs]

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
benchmarking clustering methods for time series. Therefore, we comprehensively evaluate 80 time-series clustering methods spanning 9 different 
classes from the data mining, machine learning, and deep learning literature.

## Contributors

* [Teja Reddy](https://github.com/bogireddytejareddy) (Exelon Utilities)
* [John Paparrizos](https://www.paparrizos.org/) (Ohio State University)

## Datasets

To ease reproducibility, we share our results over an established benchmark:

* The UCR Univariate Archive, which contains 128 univariate time-series datasets.
    * Download all 128 preprocessed datasets [here](https://www.thedatum.org/datasets/UCR2022_DATASETS.zip).

For the preprocessing steps check [here](https://github.com/thedatumorg/UCRArchiveFixes).

## Models

We compiled a Python library with state-of-the-art time-series clustering models so that all the comparisons are performed under the 
same framework for a consistent evaluation in terms of both performance and efficiency.

"""


text_description_dataset = f"""
We conduct our evaluation using the UCR Time-Series Archive, the largest collection of class-labeled time series datasets. 
The archive consists of a collection of 128 datasets sourced from different sensor readings while performing diverse tasks from multiple 
domains. All datasets in the archive span between 40 to 24000 time-series and have lengths varying from 15 to 2844. Datasets are z-normalized, 
and each time-series in the dataset belongs to only one class. There is a small subset of datasets in the archive containing missing values and 
varying lengths. We employ linear interpolation to fill the missing values and resample shorter time series to reach the longest time series 
in each dataset.
"""



text_description_models=f"""
We have implemented 80 methods from 9 classes of time-series clustering methods proposed for univariate time series. The following table 
lists the methods considered:


### <span style='color:Tomato'>Partitional Clustering</span>

| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:------------------------------------|
|𝑘-AVG              |ED                                 |                 [1]          |
|𝑘-Shape              |SBD                                 |          [3]                 |
|𝑘-SC              |STID                                 |            [5]               |
|𝑘-DBA              |DTW                                 |           [4]                |
|PAM             |MSM                                 |           [2]                 |
|PAM              |TWED                                 |       [2]                     |
|PAM              |ERP                                 |        [2]                    |
|PAM              |SBD                                 |         [2]                   |
|PAM              |SWALE                                 |       [2]                     |
|PAM              |DTW                                 |           [2]                 |
|PAM              |EDR                                 |        [2]                    |
|PAM              |LCSS                                 |       [2]                     |
|PAM              |ED                                 |          [2]                  |


### <span style='color:Tomato'>Kernel Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:------------------|
| KKM              | SINK                                 |     [6]              |
| KKM              | GAK                                 |       [6]            |
| KKM              | KDTW                                 |      [6]             |
| KKM              | RBF                                 |      [6]             |
| SC              | SINK                                 |        [7]           |
| SC              | GAK                                 |        [7]           |
| SC              | KDTW                                 |      [7]             |
| SC              | RBF                                 |       [7]            |


### <span style='color:Tomato'>Density Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:------------------------|
| DBSCAN              | ED                                 |         [8]             |
| DBSCAN              | SBD                                 |       [8]               |
| DBSCAN              | MSM                                 |       [8]               |
| DP              | ED                                 |        [10]              |
| DP              | SBD                                 |       [10]               |
| DP              | MSM                                 |       [10]               |
| OPTICS              | ED                                 |         [9]             |
| OPTICS              | SBD                                 |         [9]             |
| OPTICS              | MSM                                 |         [9]             |


### <span style='color:Tomato'>Hierarchical Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-------------------|
| AGG              | ED                                 |         [11]           |
| AGG              | SBD                                 |       [11]             |
| AGG              | MSM                                 |       [11]             |
| BIRCH              | -                                 |        [12]            |


### <span style='color:Tomato'>Distribution Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-----------------------|
| AP              | ED                                 |      [13]         |
| AP              | SBD                                 |      [13]         |
| AP              | MSM                                 |      [13]         |
| GMM              | -                                 |      [14]         |


### <span style='color:Tomato'>Shapelet Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:----------------------|
| UShapelet              | -                                 |       [15]                  |
| LDPS              | -                                 |              [16]           |
| USLM             | -                                 |           [17]               |


### <span style='color:Tomato'>Semi-Supervised Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-------------------|
| FeatTS              | -                                 |        [18]             |
| SS-DTW             | -                                 |         [19]             |


### <span style='color:Tomato'>Model and Feature Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:----------------------------|
| 𝑘-AVG             | AR-COEFF                                 |                               |
| 𝑘-AVG             | CATCH22                                 |                               |
| 𝑘-AVG             | LPCC                                 |                               |
| 𝑘-AVG             | AR-PVAL                                 |                               |
| 𝑘-AVG             | ES-COEFF                                 |                               |


### <span style='color:Tomato'>Deep Learning Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-------------------------------------|
| IDEC            | -                                 |               [27]                |
| DEC            | -                                 |             [26]                  |
| DTC           | -                                 |              [29]                  |
| DTCR            | -                                 |                [28]               |
| SOM-VAE            | -                                 |               [30]                |
| DEPICT           | -                                 |             [31]                   |
| SDCN            | -                                 |                 [32]              |
| ClusterGAN            | -                                 |           [34]                    |
| VADE            | -                                 |              [33]                 |
| DCN            | -                                 |              [25]                 |


** Note: Results for LDPS and USLM methods are not mentioned because of thier infeasible runtimes on large datasets.

# References

[1] MacQueen, J. "Some methods for classiﬁcation and analysis of multivariate observations." In Proc. 5th Berkeley Symposium on Math., Stat., and Prob, p. 281. 1965.
<br>
[2] Kaufman, Leonard, and Peter J. Rousseeuw. Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
<br>
[3] Paparrizos, John, and Luis Gravano. "k-shape: Efficient and accurate clustering of time series." In Proceedings of the 2015 ACM SIGMOD international conference on management of data, pp. 1855-1870. 2015.
<br>
(4) Petitjean, François, Alain Ketterlin, and Pierre Gançarski. "A global averaging method for dynamic time warping, with applications to clustering." Pattern recognition 44, no. 3 (2011): 678-693.
<br>
[5] Yang, Jaewon, and Jure Leskovec. "Patterns of temporal variation in online media." In Proceedings of the fourth ACM international conference on Web search and data mining, pp. 177-186. 2011.
<br>
[6] Dhillon, Inderjit S., Yuqiang Guan, and Brian Kulis. "Kernel k-means: spectral clustering and normalized cuts." In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 551-556. 2004.
<br>
[7] Ng, Andrew, Michael Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems 14 (2001).
<br>
[8] Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. "A density-based algorithm for discovering clusters in large spatial databases with noise." In kdd, vol. 96, no. 34, pp. 226-231. 1996.
<br>
[9] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander. "OPTICS: Ordering points to identify the clustering structure." ACM Sigmod record 28, no. 2 (1999): 49-60.
<br>
[10] Rodriguez, Alex, and Alessandro Laio. "Clustering by fast search and find of density peaks." science 344, no. 6191 (2014): 1492-1496.
<br>
[11] Kaufman, Leonard, and Peter J. Rousseeuw. Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
<br>
[12] Zhang, Tian, Raghu Ramakrishnan, and Miron Livny. "BIRCH: an efficient data clustering method for very large databases." ACM sigmod record 25, no. 2 (1996): 103-114.
<br>
[13] Frey, Brendan J., and Delbert Dueck. "Clustering by passing messages between data points." science 315, no. 5814 (2007): 972-976.
<br>
[14] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the royal statistical society: series B (methodological) 39, no. 1 (1977): 1-22.
<br>
[15] Zakaria, Jesin, Abdullah Mueen, and Eamonn Keogh. "Clustering time series using unsupervised-shapelets." In 2012 IEEE 12th International Conference on Data Mining, pp. 785-794. IEEE, 2012.
<br>
[16] Lods, Arnaud, Simon Malinowski, Romain Tavenard, and Laurent Amsaleg. "Learning DTW-preserving shapelets." In Advances in Intelligent Data Analysis XVI: 16th International Symposium, IDA 2017, London, UK, October 26–28, 2017, Proceedings 16, pp. 198-209. springer International Publishing, 2017.
<br>
[17] Zhang, Qin, Jia Wu, Hong Yang, Yingjie Tian, and Chengqi Zhang. "Unsupervised feature learning from time series." In IJCAI, pp. 2322-2328. 2016.
<br>
[18] Tiano, Donato, Angela Bonifati, and Raymond Ng. "FeatTS: Feature-based Time Series Clustering." In Proceedings of the 2021 International Conference on Management of Data, pp. 2784-2788. 2021.
<br>
[19] Dau, Hoang Anh, Nurjahan Begum, and Eamonn Keogh. "Semi-supervision dramatically improves time series clustering under dynamic time warping." In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, pp. 999-1008. 2016.
<br>
[20] Piccolo, Domenico. "A distance measure for classifying ARIMA models." Journal of time series analysis 11, no. 2 (1990): 153-164.
<br>
[21] Kalpakis, Konstantinos, Dhiral Gada, and Vasundhara Puttagunta. "Distance measures for effective clustering of ARIMA time-series." In Proceedings 2001 IEEE international conference on data mining, pp. 273-280. IEEE, 2001.
<br>
[22] Maharaj, Elizabeth Ann. "Cluster of Time Series." Journal of Classification 17, no. 2 (2000).
<br>
[23] Lubba, Carl H., Sarab S. Sethi, Philip Knaute, Simon R. Schultz, Ben D. Fulcher, and Nick S. Jones. "catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis." Data Mining and Knowledge Discovery 33, no. 6 (2019): 1821-1852.
<br>
[24] Fulcher, Ben D., and Nick S. Jones. "hctsa: A computational framework for automated time-series phenotyping using massive feature extraction." Cell systems 5, no. 5 (2017): 527-531.
<br>
[25] Yang, Bo, Xiao Fu, Nicholas D. Sidiropoulos, and Mingyi Hong. "Towards k-means-friendly spaces: Simultaneous deep learning and clustering." In international conference on machine learning, pp. 3861-3870. PMLR, 2017.
<br>
[26] Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." In International conference on machine learning, pp. 478-487. PMLR, 2016.
<br>
[27] Guo, Xifeng, Long Gao, Xinwang Liu, and Jianping Yin. "Improved deep embedded clustering with local structure preservation." In Ijcai, pp. 1753-1759. 2017.
<br>
[28] Ma, Qianli, Jiawei Zheng, Sen Li, and Gary W. Cottrell. "Learning representations for time series clustering." Advances in neural information processing systems 32 (2019).
<br>
[29] Madiraju, Naveen Sai. "Deep temporal clustering: Fully unsupervised learning of time-domain features." PhD diss., Arizona State University, 2018.
<br>
[30] Fortuin, Vincent, Matthias Hüser, Francesco Locatello, Heiko Strathmann, and Gunnar Rätsch. "Som-vae: Interpretable discrete representation learning on time series." arXiv preprint arXiv:1806.02199 (2018).
<br>
[31] Ghasedi Dizaji, Kamran, Amirhossein Herandi, Cheng Deng, Weidong Cai, and Heng Huang. "Deep clustering via joint convolutional autoencoder embedding and relative entropy minimization." In Proceedings of the IEEE international conference on computer vision, pp. 5736-5745. 2017.
<br>
[32] Bo, Deyu, Xiao Wang, Chuan Shi, Meiqi Zhu, Emiao Lu, and Peng Cui. "Structural deep clustering network." In Proceedings of the web conference 2020, pp. 1400-1410. 2020.
<br>
[33] Jiang, Zhuxi, Yin Zheng, Huachun Tan, Bangsheng Tang, and Hanning Zhou. "Variational deep embedding: A generative approach to clustering." CoRR, abs/1611.05148 1 (2016).
<br>
[34] Ghasedi, Kamran, Xiaoqian Wang, Cheng Deng, and Heng Huang. "Balanced self-paced learning for generative adversarial clustering network." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4391-4400. 2019.


"""

list_references = f"""
(1) MacQueen, J. "Some methods for classiﬁcation and analysis of multivariate observations." In Proc. 5th Berkeley Symposium on Math., Stat., and Prob, p. 281. 1965.
<br>
(2) Kaufman, Leonard, and Peter J. Rousseeuw. Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
<br>
(3) Paparrizos, John, and Luis Gravano. "k-shape: Efficient and accurate clustering of time series." In Proceedings of the 2015 ACM SIGMOD international conference on management of data, pp. 1855-1870. 2015.
<br>
(4) Petitjean, François, Alain Ketterlin, and Pierre Gançarski. "A global averaging method for dynamic time warping, with applications to clustering." Pattern recognition 44, no. 3 (2011): 678-693.
<br>
(5) Yang, Jaewon, and Jure Leskovec. "Patterns of temporal variation in online media." In Proceedings of the fourth ACM international conference on Web search and data mining, pp. 177-186. 2011.
<br>
(6) Dhillon, Inderjit S., Yuqiang Guan, and Brian Kulis. "Kernel k-means: spectral clustering and normalized cuts." In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 551-556. 2004.
<br>
(7) Ng, Andrew, Michael Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems 14 (2001).
<br>
(8) Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. "A density-based algorithm for discovering clusters in large spatial databases with noise." In kdd, vol. 96, no. 34, pp. 226-231. 1996.
<br>
(9) Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander. "OPTICS: Ordering points to identify the clustering structure." ACM Sigmod record 28, no. 2 (1999): 49-60.
<br>
(10) Rodriguez, Alex, and Alessandro Laio. "Clustering by fast search and find of density peaks." science 344, no. 6191 (2014): 1492-1496.
<br>
(11) Kaufman, Leonard, and Peter J. Rousseeuw. Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
<br>
(12) Zhang, Tian, Raghu Ramakrishnan, and Miron Livny. "BIRCH: an efficient data clustering method for very large databases." ACM sigmod record 25, no. 2 (1996): 103-114.
<br>
(13) Frey, Brendan J., and Delbert Dueck. "Clustering by passing messages between data points." science 315, no. 5814 (2007): 972-976.
<br>
(14) Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the royal statistical society: series B (methodological) 39, no. 1 (1977): 1-22.
<br>
(15) Zakaria, Jesin, Abdullah Mueen, and Eamonn Keogh. "Clustering time series using unsupervised-shapelets." In 2012 IEEE 12th International Conference on Data Mining, pp. 785-794. IEEE, 2012.
<br>
(16) Lods, Arnaud, Simon Malinowski, Romain Tavenard, and Laurent Amsaleg. "Learning DTW-preserving shapelets." In Advances in Intelligent Data Analysis XVI: 16th International Symposium, IDA 2017, London, UK, October 26–28, 2017, Proceedings 16, pp. 198-209. springer International Publishing, 2017.
<br>
(17) Zhang, Qin, Jia Wu, Hong Yang, Yingjie Tian, and Chengqi Zhang. "Unsupervised feature learning from time series." In IJCAI, pp. 2322-2328. 2016.
<br>
(18) Tiano, Donato, Angela Bonifati, and Raymond Ng. "FeatTS: Feature-based Time Series Clustering." In Proceedings of the 2021 International Conference on Management of Data, pp. 2784-2788. 2021.
<br>
(19) Dau, Hoang Anh, Nurjahan Begum, and Eamonn Keogh. "Semi-supervision dramatically improves time series clustering under dynamic time warping." In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, pp. 999-1008. 2016.
<br>
(20) Piccolo, Domenico. "A distance measure for classifying ARIMA models." Journal of time series analysis 11, no. 2 (1990): 153-164.
<br>
(21) Kalpakis, Konstantinos, Dhiral Gada, and Vasundhara Puttagunta. "Distance measures for effective clustering of ARIMA time-series." In Proceedings 2001 IEEE international conference on data mining, pp. 273-280. IEEE, 2001.
<br>
(22) Maharaj, Elizabeth Ann. "Cluster of Time Series." Journal of Classification 17, no. 2 (2000).
<br>
(23) Lubba, Carl H., Sarab S. Sethi, Philip Knaute, Simon R. Schultz, Ben D. Fulcher, and Nick S. Jones. "catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis." Data Mining and Knowledge Discovery 33, no. 6 (2019): 1821-1852.
<br>
(24) Fulcher, Ben D., and Nick S. Jones. "hctsa: A computational framework for automated time-series phenotyping using massive feature extraction." Cell systems 5, no. 5 (2017): 527-531.
<br>
(25) Yang, Bo, Xiao Fu, Nicholas D. Sidiropoulos, and Mingyi Hong. "Towards k-means-friendly spaces: Simultaneous deep learning and clustering." In international conference on machine learning, pp. 3861-3870. PMLR, 2017.
<br>
(26) Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." In International conference on machine learning, pp. 478-487. PMLR, 2016.
<br>
(27) Guo, Xifeng, Long Gao, Xinwang Liu, and Jianping Yin. "Improved deep embedded clustering with local structure preservation." In Ijcai, pp. 1753-1759. 2017.
<br>
(28) Ma, Qianli, Jiawei Zheng, Sen Li, and Gary W. Cottrell. "Learning representations for time series clustering." Advances in neural information processing systems 32 (2019).
<br>
(29) Madiraju, Naveen Sai. "Deep temporal clustering: Fully unsupervised learning of time-domain features." PhD diss., Arizona State University, 2018.
<br>
(30) Fortuin, Vincent, Matthias Hüser, Francesco Locatello, Heiko Strathmann, and Gunnar Rätsch. "Som-vae: Interpretable discrete representation learning on time series." arXiv preprint arXiv:1806.02199 (2018).
<br>
(31) Ghasedi Dizaji, Kamran, Amirhossein Herandi, Cheng Deng, Weidong Cai, and Heng Huang. "Deep clustering via joint convolutional autoencoder embedding and relative entropy minimization." In Proceedings of the IEEE international conference on computer vision, pp. 5736-5745. 2017.
<br>
(32) Bo, Deyu, Xiao Wang, Chuan Shi, Meiqi Zhu, Emiao Lu, and Peng Cui. "Structural deep clustering network." In Proceedings of the web conference 2020, pp. 1400-1410. 2020.
<br>
(33) Jiang, Zhuxi, Yin Zheng, Huachun Tan, Bangsheng Tang, and Hanning Zhou. "Variational deep embedding: A generative approach to clustering." CoRR, abs/1611.05148 1 (2016).
<br>
(34) Ghasedi, Kamran, Xiaoqian Wang, Cheng Deng, and Heng Huang. "Balanced self-paced learning for generative adversarial clustering network." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4391-4400. 2019.
"""

