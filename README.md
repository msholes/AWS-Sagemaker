# AWS-Sagemaker with Jupyter Notebooks

### Step 1: Loading the data from Amazon S3


```python
##Import Libraries

import os
import boto3
import io
import sagemaker

%matplotlib inline 

import pandas as pd
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.style.use('ggplot')
import pickle, gzip, urllib, json
import csv
```


```python
## Get IAM role created for notebook instance --'AmazonSageMakerFullAccess' role needed

from sagemaker import get_execution_role
role = get_execution_role()

role
```

## Starting Boto3 client to interact with AWS Bucket


```python
s3_client = boto3.client('s3')
data_bucket_name='aws-ml-blog-sagemaker-census-segmentation'
```

## List of Objects within Bucket


```python
obj_list=s3_client.list_objects(Bucket=data_bucket_name)
file=[]
for contents in obj_list['Contents']:
    file.append(contents['Key'])
print(file)
['acs2015_county_data.csv', 'counties/']
file_data=file[0]
```

    ['Census_Data_for_SageMaker.csv']


## Get data from CSV file in Bucket


```python
response = s3_client.get_object(Bucket=data_bucket_name, Key=file_data)
response_body = response["Body"].read()
counties = pd.read_csv(io.BytesIO(response_body), header=0, delimiter=",", low_memory=False) 
```

## First 5 rows of data


```python
counties.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CensusId</th>
      <th>State</th>
      <th>County</th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>Alabama</td>
      <td>Autauga</td>
      <td>55221</td>
      <td>26745</td>
      <td>28476</td>
      <td>2.6</td>
      <td>75.8</td>
      <td>18.5</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>26.5</td>
      <td>23986</td>
      <td>73.6</td>
      <td>20.9</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1003</td>
      <td>Alabama</td>
      <td>Baldwin</td>
      <td>195121</td>
      <td>95314</td>
      <td>99807</td>
      <td>4.5</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.6</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>3.9</td>
      <td>26.4</td>
      <td>85953</td>
      <td>81.5</td>
      <td>12.3</td>
      <td>5.8</td>
      <td>0.4</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>Alabama</td>
      <td>Barbour</td>
      <td>26932</td>
      <td>14497</td>
      <td>12435</td>
      <td>4.6</td>
      <td>46.2</td>
      <td>46.7</td>
      <td>0.2</td>
      <td>...</td>
      <td>1.8</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>24.1</td>
      <td>8597</td>
      <td>71.8</td>
      <td>20.8</td>
      <td>7.3</td>
      <td>0.1</td>
      <td>17.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1007</td>
      <td>Alabama</td>
      <td>Bibb</td>
      <td>22604</td>
      <td>12073</td>
      <td>10531</td>
      <td>2.2</td>
      <td>74.5</td>
      <td>21.4</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.6</td>
      <td>1.5</td>
      <td>0.7</td>
      <td>28.8</td>
      <td>8294</td>
      <td>76.8</td>
      <td>16.1</td>
      <td>6.7</td>
      <td>0.4</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1009</td>
      <td>Alabama</td>
      <td>Blount</td>
      <td>57710</td>
      <td>28512</td>
      <td>29198</td>
      <td>8.6</td>
      <td>87.9</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>...</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>2.3</td>
      <td>34.9</td>
      <td>22189</td>
      <td>82.0</td>
      <td>13.5</td>
      <td>4.2</td>
      <td>0.4</td>
      <td>7.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



### Step 2: Exploratory Data Analysis (EDA)- Data cleaning and Exploration

## Cleaning the data


```python
counties.shape
```




    (3220, 37)




```python
## Drop missing data

counties.dropna(inplace=True)
counties.shape
```




    (3218, 37)




```python
## Combining descriptive columns for Index: 'state-county'

counties.index=counties['State'] + "-" + counties['County']
counties.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CensusId</th>
      <th>State</th>
      <th>County</th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama-Autauga</th>
      <td>1001</td>
      <td>Alabama</td>
      <td>Autauga</td>
      <td>55221</td>
      <td>26745</td>
      <td>28476</td>
      <td>2.6</td>
      <td>75.8</td>
      <td>18.5</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>26.5</td>
      <td>23986</td>
      <td>73.6</td>
      <td>20.9</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>Alabama-Baldwin</th>
      <td>1003</td>
      <td>Alabama</td>
      <td>Baldwin</td>
      <td>195121</td>
      <td>95314</td>
      <td>99807</td>
      <td>4.5</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.6</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>3.9</td>
      <td>26.4</td>
      <td>85953</td>
      <td>81.5</td>
      <td>12.3</td>
      <td>5.8</td>
      <td>0.4</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>Alabama-Barbour</th>
      <td>1005</td>
      <td>Alabama</td>
      <td>Barbour</td>
      <td>26932</td>
      <td>14497</td>
      <td>12435</td>
      <td>4.6</td>
      <td>46.2</td>
      <td>46.7</td>
      <td>0.2</td>
      <td>...</td>
      <td>1.8</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>24.1</td>
      <td>8597</td>
      <td>71.8</td>
      <td>20.8</td>
      <td>7.3</td>
      <td>0.1</td>
      <td>17.6</td>
    </tr>
    <tr>
      <th>Alabama-Bibb</th>
      <td>1007</td>
      <td>Alabama</td>
      <td>Bibb</td>
      <td>22604</td>
      <td>12073</td>
      <td>10531</td>
      <td>2.2</td>
      <td>74.5</td>
      <td>21.4</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.6</td>
      <td>1.5</td>
      <td>0.7</td>
      <td>28.8</td>
      <td>8294</td>
      <td>76.8</td>
      <td>16.1</td>
      <td>6.7</td>
      <td>0.4</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>Alabama-Blount</th>
      <td>1009</td>
      <td>Alabama</td>
      <td>Blount</td>
      <td>57710</td>
      <td>28512</td>
      <td>29198</td>
      <td>8.6</td>
      <td>87.9</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>...</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>2.3</td>
      <td>34.9</td>
      <td>22189</td>
      <td>82.0</td>
      <td>13.5</td>
      <td>4.2</td>
      <td>0.4</td>
      <td>7.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
## Drop Columns

drop=["CensusId", "State", "County"]
counties.drop(drop, axis=1, inplace=True)
counties.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>Citizen</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama-Autauga</th>
      <td>55221</td>
      <td>26745</td>
      <td>28476</td>
      <td>2.6</td>
      <td>75.8</td>
      <td>18.5</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40725</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>26.5</td>
      <td>23986</td>
      <td>73.6</td>
      <td>20.9</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>Alabama-Baldwin</th>
      <td>195121</td>
      <td>95314</td>
      <td>99807</td>
      <td>4.5</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>147695</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>3.9</td>
      <td>26.4</td>
      <td>85953</td>
      <td>81.5</td>
      <td>12.3</td>
      <td>5.8</td>
      <td>0.4</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>Alabama-Barbour</th>
      <td>26932</td>
      <td>14497</td>
      <td>12435</td>
      <td>4.6</td>
      <td>46.2</td>
      <td>46.7</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>20714</td>
      <td>...</td>
      <td>1.8</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>24.1</td>
      <td>8597</td>
      <td>71.8</td>
      <td>20.8</td>
      <td>7.3</td>
      <td>0.1</td>
      <td>17.6</td>
    </tr>
    <tr>
      <th>Alabama-Bibb</th>
      <td>22604</td>
      <td>12073</td>
      <td>10531</td>
      <td>2.2</td>
      <td>74.5</td>
      <td>21.4</td>
      <td>0.4</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>17495</td>
      <td>...</td>
      <td>0.6</td>
      <td>1.5</td>
      <td>0.7</td>
      <td>28.8</td>
      <td>8294</td>
      <td>76.8</td>
      <td>16.1</td>
      <td>6.7</td>
      <td>0.4</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>Alabama-Blount</th>
      <td>57710</td>
      <td>28512</td>
      <td>29198</td>
      <td>8.6</td>
      <td>87.9</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>42345</td>
      <td>...</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>2.3</td>
      <td>34.9</td>
      <td>22189</td>
      <td>82.0</td>
      <td>13.5</td>
      <td>4.2</td>
      <td>0.4</td>
      <td>7.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



## Visualizing the Data


```python
import seaborn as sns

for a in ['Professional', 'Service', 'Office']:
    ax=plt.subplots(figsize=(6,3))
    ax=sns.distplot(counties[a])
    title="Histogram of " + a
    ax.set_title(title, fontsize=12)
    plt.show()
```

    /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](output_18_1.png)


    /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](output_18_3.png)


    /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](output_18_5.png)



```python
## Indicate Mean and Skew-- A typical county has ~ 25-30% Professional workers, with a right skew,
## long tail and a Prof. worker % max ~ 80% in some counties
```


```python
## Feature engineering

## minmaxscaler to transform numerical columns to fall between 0 and 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
counties_scaled = pd.DataFrame(scaler.fit_transform(counties))
counties_scaled.columns = counties.columns
counties_scaled.index = counties.index

```

    /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/sklearn/preprocessing/data.py:334: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)



```python
counties_scaled.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>Citizen</th>
      <th>...</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>...</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
      <td>3218.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.009883</td>
      <td>0.009866</td>
      <td>0.009899</td>
      <td>0.110170</td>
      <td>0.756024</td>
      <td>0.100942</td>
      <td>0.018682</td>
      <td>0.029405</td>
      <td>0.006470</td>
      <td>0.011540</td>
      <td>...</td>
      <td>0.046496</td>
      <td>0.041154</td>
      <td>0.124428</td>
      <td>0.470140</td>
      <td>0.009806</td>
      <td>0.760810</td>
      <td>0.194426</td>
      <td>0.216744</td>
      <td>0.029417</td>
      <td>0.221775</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.031818</td>
      <td>0.031692</td>
      <td>0.031948</td>
      <td>0.192617</td>
      <td>0.229682</td>
      <td>0.166262</td>
      <td>0.078748</td>
      <td>0.062744</td>
      <td>0.035446</td>
      <td>0.033933</td>
      <td>...</td>
      <td>0.051956</td>
      <td>0.042321</td>
      <td>0.085301</td>
      <td>0.143135</td>
      <td>0.032305</td>
      <td>0.132949</td>
      <td>0.106923</td>
      <td>0.106947</td>
      <td>0.046451</td>
      <td>0.112138</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.001092</td>
      <td>0.001117</td>
      <td>0.001069</td>
      <td>0.019019</td>
      <td>0.642285</td>
      <td>0.005821</td>
      <td>0.001086</td>
      <td>0.004808</td>
      <td>0.000000</td>
      <td>0.001371</td>
      <td>...</td>
      <td>0.019663</td>
      <td>0.023018</td>
      <td>0.072581</td>
      <td>0.373402</td>
      <td>0.000948</td>
      <td>0.697279</td>
      <td>0.120861</td>
      <td>0.147541</td>
      <td>0.010204</td>
      <td>0.150685</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.002571</td>
      <td>0.002591</td>
      <td>0.002539</td>
      <td>0.039039</td>
      <td>0.842685</td>
      <td>0.022119</td>
      <td>0.003257</td>
      <td>0.012019</td>
      <td>0.000000</td>
      <td>0.003219</td>
      <td>...</td>
      <td>0.033708</td>
      <td>0.033248</td>
      <td>0.104839</td>
      <td>0.462916</td>
      <td>0.002234</td>
      <td>0.785714</td>
      <td>0.172185</td>
      <td>0.188525</td>
      <td>0.020408</td>
      <td>0.208219</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.006594</td>
      <td>0.006645</td>
      <td>0.006556</td>
      <td>0.098098</td>
      <td>0.933868</td>
      <td>0.111758</td>
      <td>0.006515</td>
      <td>0.028846</td>
      <td>0.000000</td>
      <td>0.008237</td>
      <td>...</td>
      <td>0.056180</td>
      <td>0.048593</td>
      <td>0.150538</td>
      <td>0.560102</td>
      <td>0.006144</td>
      <td>0.853741</td>
      <td>0.243377</td>
      <td>0.256831</td>
      <td>0.030612</td>
      <td>0.271233</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>



## Step 3: Data modelling


```python
## Dimensionality reduction using PCA -- http://sagemaker.readthedocs.io/en/latest/pca.html

from sagemaker import PCA
bucket_name= 'census-data-example1'
num_components=33

pca_SM = PCA(role=role,
          train_instance_count=1,
          train_instance_type='ml.c4.xlarge',
          output_path='s3://'+ bucket_name +'/counties/',
            num_components=num_components)
```


```python
## Extracting numpy array from the DataFrame and explicitly casting to float32

train_data = counties_scaled.values.astype('float32')
```


```python
## record_set function converts numpy array into record set format needed for Amazon Sagemaker

## Fit function on PCA model, passing in training data, and spinning up a training instance or cluster to perform job


%time
pca_SM.fit(pca_SM.record_set(train_data))
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 7.15 µs
    2020-04-20 00:38:05 Starting - Starting the training job...
    2020-04-20 00:38:06 Starting - Launching requested ML instances...
    2020-04-20 00:39:05 Starting - Preparing the instances for training......
    2020-04-20 00:39:47 Downloading - Downloading input data...
    2020-04-20 00:40:29 Training - Downloading the training image..[34mDocker entrypoint called with argument(s): train[0m
    [34mRunning default environment configuration script[0m
    [34m[04/20/2020 00:40:44 INFO 139992907982656] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/resources/default-conf.json: {u'_num_gpus': u'auto', u'_log_level': u'info', u'subtract_mean': u'true', u'force_dense': u'true', u'epochs': 1, u'algorithm_mode': u'regular', u'extra_components': u'-1', u'_kvstore': u'dist_sync', u'_num_kv_servers': u'auto'}[0m
    [34m[04/20/2020 00:40:44 INFO 139992907982656] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'feature_dim': u'34', u'mini_batch_size': u'500', u'num_components': u'33'}[0m
    [34m[04/20/2020 00:40:44 INFO 139992907982656] Final configuration: {u'num_components': u'33', u'_num_gpus': u'auto', u'_log_level': u'info', u'subtract_mean': u'true', u'force_dense': u'true', u'epochs': 1, u'algorithm_mode': u'regular', u'feature_dim': u'34', u'extra_components': u'-1', u'_kvstore': u'dist_sync', u'_num_kv_servers': u'auto', u'mini_batch_size': u'500'}[0m
    [34m[04/20/2020 00:40:44 WARNING 139992907982656] Loggers have already been setup.[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Launching parameter server for role scheduler[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/7d66cd76-1825-466c-9634-2c26091db721', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'AWS_REGION': 'us-east-2', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2020-04-20-00-38-05-377', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-75-200.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/502e0702-0759-4636-969d-81e9154076f5', 'PWD': '/', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:375139585421:training-job/pca-2020-04-20-00-38-05-377', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/7d66cd76-1825-466c-9634-2c26091db721', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.75.200', 'AWS_REGION': 'us-east-2', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2020-04-20-00-38-05-377', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-75-200.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/502e0702-0759-4636-969d-81e9154076f5', 'DMLC_ROLE': 'scheduler', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:375139585421:training-job/pca-2020-04-20-00-38-05-377', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Launching parameter server for role server[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/7d66cd76-1825-466c-9634-2c26091db721', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'AWS_REGION': 'us-east-2', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2020-04-20-00-38-05-377', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-75-200.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/502e0702-0759-4636-969d-81e9154076f5', 'PWD': '/', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:375139585421:training-job/pca-2020-04-20-00-38-05-377', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/7d66cd76-1825-466c-9634-2c26091db721', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.75.200', 'AWS_REGION': 'us-east-2', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2020-04-20-00-38-05-377', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-75-200.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/502e0702-0759-4636-969d-81e9154076f5', 'DMLC_ROLE': 'server', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:375139585421:training-job/pca-2020-04-20-00-38-05-377', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Environment: {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/7d66cd76-1825-466c-9634-2c26091db721', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_PS_ROOT_PORT': '9000', 'DMLC_NUM_WORKER': '1', 'SAGEMAKER_HTTP_PORT': '8080', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.75.200', 'AWS_REGION': 'us-east-2', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'pca-2020-04-20-00-38-05-377', 'HOME': '/root', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-75-200.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/502e0702-0759-4636-969d-81e9154076f5', 'DMLC_ROLE': 'worker', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:375139585421:training-job/pca-2020-04-20-00-38-05-377', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34mProcess 61 is a shell:scheduler.[0m
    [34mProcess 70 is a shell:server.[0m
    [34mProcess 1 is a worker.[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Using default worker.[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Loaded iterator creator application/x-labeled-vector-protobuf for content type ('application/x-labeled-vector-protobuf', '1.0')[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Loaded iterator creator protobuf for content type ('protobuf', '1.0')[0m
    [34m[04/20/2020 00:40:45 INFO 139992907982656] Create Store: dist_sync[0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] nvidia-smi took: 0.0251729488373 secs to identify 0 gpus[0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] Number of GPUs being used: 0[0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] The default executor is <PCAExecutor on cpu(0)>.[0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] 34 feature(s) found in 'data'.[0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] <PCAExecutor on cpu(0)> is assigned to batch slice from 0 to 499.[0m
    [34m#metrics {"Metrics": {"initialize.time": {"count": 1, "max": 600.654125213623, "sum": 600.654125213623, "min": 600.654125213623}}, "EndTime": 1587343246.106396, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1587343245.492412}
    [0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Records Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Max Records Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Reset Count": {"count": 1, "max": 0, "sum": 0.0, "min": 0}}, "EndTime": 1587343246.106614, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1587343246.106558}
    [0m
    [34m[2020-04-20 00:40:46.106] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 0, "duration": 613, "num_examples": 1, "num_bytes": 82000}[0m
    [34m[2020-04-20 00:40:46.144] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 1, "duration": 30, "num_examples": 7, "num_bytes": 527752}[0m
    [34m#metrics {"Metrics": {"epochs": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "update.time": {"count": 1, "max": 38.1929874420166, "sum": 38.1929874420166, "min": 38.1929874420166}}, "EndTime": 1587343246.145171, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1587343246.10649}
    [0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] #progress_metric: host=algo-1, completed 100 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Number of Batches Since Last Reset": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Number of Records Since Last Reset": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Total Batches Seen": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Total Records Seen": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Max Records Seen Between Resets": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Reset Count": {"count": 1, "max": 1, "sum": 1.0, "min": 1}}, "EndTime": 1587343246.145527, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "PCA", "epoch": 0}, "StartTime": 1587343246.10694}
    [0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] #throughput_metric: host=algo-1, train throughput=83088.2469267 records/second[0m
    [34m#metrics {"Metrics": {"finalize.time": {"count": 1, "max": 17.56000518798828, "sum": 17.56000518798828, "min": 17.56000518798828}}, "EndTime": 1587343246.163447, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1587343246.145271}
    [0m
    [34m[04/20/2020 00:40:46 INFO 139992907982656] Test data is not provided.[0m
    [34m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 1340.5430316925049, "sum": 1340.5430316925049, "min": 1340.5430316925049}, "setuptime": {"count": 1, "max": 552.962064743042, "sum": 552.962064743042, "min": 552.962064743042}}, "EndTime": 1587343246.168361, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1587343246.163503}
    [0m
    
    2020-04-20 00:40:53 Uploading - Uploading generated training model
    2020-04-20 00:40:53 Completed - Training job completed
    Training seconds: 66
    Billable seconds: 66



```python
## Unzipping file with trained model artifacts from S2 Bucket -- Job name found under 'Training Jobs' in Sagemaker

job_name='pca-2020-04-20-00-38-05-377'
model_key = "counties/" + job_name + "/output/model.tar.gz"

boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')
```




    2304




```python
## Load ND array using MXNet

import mxnet as mx
pca_model_params = mx.ndarray.load('model_algo-1')
```


```python
## Three groups of params in PCA model --mean if the 'subtract_mean' hyperparam is true --v: principal components --s: singular values of the comp. for PCA tranformatoin
## explained-varience-ratio ~=square(s)/ sum(square(s))

s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())
```


```python
## Variance by top 5 largest components ~72% of total variance in data

s.iloc[28:,:].apply(lambda x: x*x).sum()/s.apply(lambda x: x*x).sum()

```




    0    0.717983
    dtype: float32


