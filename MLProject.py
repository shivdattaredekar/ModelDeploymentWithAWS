#!/usr/bin/env python
# coding: utf-8

# ### Importing Important Libraries

# #### Steps To Be Followed
# 1. Importing necessary Libraries
# 2. Creating S3 bucket 
# 3. Mapping train And Test Data in S3
# 4. Mapping The path of the models in S3

# In[2]:


# You are importing sagemaker
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session


# In[3]:


# Creating a bucket name in S3 in AWS
bucket_name = 'bank-application-shivdatta-2024' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
my_region = boto3.session.Session().region_name # set the region of the instance
print(my_region)


# In[4]:


# After the below code runs sucessfully you can go and reload AWS S3 and check if the new bucket is formed or not  
import botocore

s3 = boto3.resource('s3')

try:
    if my_region == 'ap-south-1':
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'ap-south-1'})
    else:
        s3.create_bucket(Bucket=bucket_name)
    print('S3 bucket created successfully')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == 'BucketAlreadyExists':
        print('Bucket already exists.')
    else:
        print('S3 error: ', e)

    


# In[5]:


# set an output path where the trained model will be saved and can be used in furture to retrain the model with new data
prefix = 'xgboost-as-a-built-in-algo'
output_path =f's3://{bucket_name}/{prefix}/output'
print(output_path)


# #### Downloading The Dataset And Storing in S3

# In[6]:


import pandas as pd
import urllib
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ',e)

try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[ ]:





# In[7]:


### Train Test split

import numpy as np
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)


# In[21]:


print(model_data.shape)
print(train_data.shape)
print(test_data.shape)


# In[25]:


### Saving Train And Test Into Buckets
## Always remember that in AWS Dependent variable will be always be palced first followed by all IV's
## We start with Train Data

import os
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], 
                                                axis=1)], 
                                                axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')


import sagemaker

s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data=f's3://{bucket_name}/{prefix}/train', 
    content_type='csv'
)


# In[26]:


# Test Data Into Buckets


pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')

import sagemaker

# Create S3 input for testing
s3_input_test = sagemaker.inputs.TrainingInput(
    s3_data=f's3://{bucket_name}/{prefix}/test', 
    content_type='csv'
)


# In[16]:


## Now can can check in S3 'Bucket bank-application-shivdatta-2024' you can observe the prefix path and you train/test files 


# In[27]:


os.getcwd()


# ### Building Models Xgboot- Inbuilt Algorithm

# In[29]:


# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
from sagemaker.image_uris import retrieve

container = retrieve(region=boto3.Session().region_name,
                     framework='xgboost',
                     version='1.0-1')


# In[30]:


# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"binary:logistic",
        "num_round":50
        }


# In[32]:


# construct a SageMaker estimator that calls the xgboost-container
import sagemaker
from sagemaker.estimator import Estimator

# Fetch the IAM role
role = sagemaker.get_execution_role()

# Create an Estimator object
estimator = Estimator(image_uri=container,
                      role=role,
                      instance_count=1,
                      instance_type='ml.m5.2xlarge',
                      volume_size=5, # 5GB
                      output_path=output_path,
                      use_spot_instances=True,
                      max_run=300,
                      max_wait=600,
                      hyperparameters=hyperparameters)


# In[33]:


output_path


# In[38]:


print(s3_input_train)
print(s3_input_test)


# In[39]:


print("Training data S3 location:", s3_input_train.config['DataSource']['S3DataSource']['S3Uri'])
print("Test data S3 location:", s3_input_test.config['DataSource']['S3DataSource']['S3Uri'])


# In[40]:


estimator.fit({'train': s3_input_train,'validation': s3_input_test})


# ### Deploy Machine Learning Model As Endpoints

# In[41]:


# Deploy the model endpoint
xgb_predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# #### Prediction of the Test Data

# In[46]:


import csv
import io
import json
import numpy as np

class CSVSerializer:
    """
    Custom serializer for converting input data to CSV format.
    """
    def serialize(self, data):
        """
        Serialize input data to CSV format.
        """
        if isinstance(data, np.ndarray):
            # Convert numpy array to CSV string
            csv_data = io.StringIO()
            np.savetxt(csv_data, data, delimiter=",", fmt="%s")
            return csv_data.getvalue().strip()
        elif isinstance(data, list):
            # Convert list of lists to CSV string
            csv_data = io.StringIO()
            writer = csv.writer(csv_data)
            writer.writerows(data)
            return csv_data.getvalue().strip()
        else:
            raise ValueError("Unsupported input data type. Must be numpy array or list of lists.")

# The above code is to create CSVSerializer object to convert the input data of any type into CSV format  

test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = CSVSerializer()
predictions_csv = xgb_predictor.predict(test_data_array).decode('utf-8')

predictions_array = np.fromstring(predictions_csv[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)


# In[49]:


predictions_array


# In[50]:


cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# ##### Precision, Recall are not that good as bank datasets tend to be imbalnced hence with proper balncing and Hyperparamater tuning we can get better model 

# ### Deleting The Endpoints

# In[51]:


# Make sure you are deleting the end points so that you will not be charged much in AWS  


# In[52]:


sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()


# In[ ]:




