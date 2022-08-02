# Customer Churn Prediction on Amazon Sagemaker

Curn Prediction means detecting which customers are likely to leave a service or to cancel a subscription to a service. For many businesses, it is a critical thing to know in advance which customers are at risk of leaving since acquiring new customers often costs more than retaining existing ones. Once we can indentify those customers that are at the cusp of churning, we can incentivize them with special offers and discounts to increase their chances of staying.

In this project, customer churn of a mobile phone operator ([dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) is predicted using ML model called Gradient Boosted Trees (XGBoost). Once the operator knows that a customers is thinking of leaving, it can offer timely incentives - such as a phone upgrade or perhaps having a new feature activated – and the customer may stick around. Incentives are often much more cost-effective than losing and reacquiring a customer.Additionally, since ML models rarely give perfect predictions, it becomes important to know how to incorporate the relative costs of prediction mistakes when determining the financial outcome of using ML. Such a cost analysis of binary classification is also provided at the end of this project.

- `data_analysis_preparation.ipynb` : An overview of the dataset, along with distribution of various fields and correlations among them. Data cleanup and preparation for XGBoost model is also done in this notebook.

- `hyperparameter_tuning.ipynb` : Hyperparameter optimisation of XGBoost model is automated by using AWS Sagemaker "Hyperparameter Tuning Jobs" in this notebook. Model with the best evalution metrics is saved for deployment.

- `model_batch_job_&_eval.ipynb` : A "Batch Transform Job" of Sagemaker is created using the previously best trained model and evaluated using the test set. Cutoff Threshold was varied to see how the confusion matrix varies. A cost function of binary predictions was assumed to choose the optimal cutoff that has lowest financial cost.


