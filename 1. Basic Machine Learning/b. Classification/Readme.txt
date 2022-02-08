Classification also comes under supervised learning, meaning we've got clear labels/output values for the input data.
Although the data might not be clean everytime.

The idea is to separate the data into target classes with a model.
For Random forest, the model is expressed as an ensemble of decision trees with each tree selected using bagging (search to know more).
For Naive bayes, the model uses bayes theorem with a naive assumption that the attributes are independent of each other.
For SVM, the model is a hyperplane that uses kernel-trick to find the optimal solution. (See https://www.youtube.com/watch?v=8A7L0GsBiLQ&ab_channel=StatQuestwithJoshStarmer)
For KNN, the idea is to classify based on the k-nearest neighbours. Should be careful when choosing 'k'.
For Gradient Boosting,

There are 3 types of classification tasks :-
1) Binary classification (When 2 output classes are present)
2) Multi-class classification (When more than 2 output classes are present)
3) Multi-label classification (When the output can take 2 or more classes at the same time)

Check out this article to know more: https://machinelearningmastery.com/types-of-classification-in-machine-learning/

*** Some important things to note while doing classification ***
1) Check if the data is balanced, meaning if the output classes are evenly distributed in the data.
    Or else this might introduce bias in the model. See the section 'Imbalanced Classification' in the above article.

# I'll try to use 5 different problems and fit the data for the above 5 models of classification.
# I'll try to choose the data which has minimal pre-processing.