❤️ Predicting Heart Disease with Code
This project uses patient data to build a tool that can guess if someone has heart disease. The main goal is to help doctors find the disease early.

🎯 Why This Project is Important
Heart disease is a very serious health problem. Finding it early can save a life. This tool can help by:

Finding patients with high risk much sooner.

Giving doctors a helpful second opinion.

Making it faster to check for heart disease.

⚙️ How It Works: Our Steps
Clean the Data: We found and fixed the missing information in the dataset.

Prepare the Data: We turned all text data into numbers so the computer could understand it. We made the goal simple: 0 (No Disease) or 1 (Has Disease).

Train the Models: We used most of the data to "teach" 7 different computer models how to spot the patterns of heart disease.

🤖 The Models We Tested
We tested seven models to see which one was the best for this job:

Logistic Regression

Gaussian Naive Bayes (NB)

Random Forest

XGBoost

Decision Tree

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

🏆 Results: Which Model Won?
We used the rest of the data to test each model's accuracy. Here are the scores:

Rank	Model	Accuracy (%)
1	Logistic Regression	83.52%
2	Gaussian NB	82.42%
3	Random Forest	81.32%
4	XGBoost	79.12%
5	Decision Tree	78.02%
6	SVM	71.43%
7	KNN	68.68%

Export to Sheets
Best Model: Logistic Regression
This model won because it's simple and very good at finding straightforward patterns. It wasn't easily fooled by complex data.

Worst Model: K-Nearest Neighbors (KNN)
This model did poorly because it gets confused when numbers are on different scales (like age 20-80 vs. gender 0-1). We learned that we must scale our numbers for this model to work well.

💡 What We Learned
Simple is often best: The simplest model won.

Preparing data is key: The KNN model failed because we missed an important data preparation step.

Always test different models: You never know which one will work best for your data.

🚀 Next Steps: How to Make This Better
Fix the worst model: Add "Feature Scaling" to your code. This will make all numbers have a similar scale and should make the KNN model much more accurate.

Tune the best model: Try to adjust the settings of Logistic Regression to see if you can get an even higher score.
