# 66days
My 66 days of data science

## Reason
From a combination of wanting to change career and the accountability I have to my self to continually push to this repo, I have chosen the option to follow Ken Jee's #66daysofdata (https://www.kennethjee.com/66daysofdata) which builkds upn the principles of habit formation talked about by James Clear in Atomic Habits (https://jamesclear.com/atomic-habits)


## Day 1 (14/03/2022)
Today I took some time to look through th pkl files in the AlphaFold outputs to determine what was contained within them and how these could be plotted to enable a more efficient interpretation of the resulting protein structures.  
(https://github.com/mplockhart/66days/blob/main/alphafold/EDA_AF_outputs.ipynb)

## Day 2 (15/03/2022)
I moved back to my CodeCademy course where I was introduced to SciPy at the start of the Supervised Machine Learning module. It showed the manual methods of calculating different distances (Euclidean, Manhattan and Hamming) and how these were compared to using the module.  
(https://github.com/mplockhart/66days/blob/main/codecademy_learning/honeyproduction.csv)

## Day 3 (16/03/2022)
Continuing on my CodeCademy with linear regression. It introduced the concepts of gradient descent and manually writing methods to calculate this, before showing the concepts repeated more easily in Scikit-Learn.
(https://github.com/mplockhart/66days/blob/main/codecademy_learning/linear_regression.ipynb)


## Day 4 ( 17/03/2022)
Through code academy I enhanced my knowledge of linear regression with multiple linear regression where one can fit a line to multiple attributes and use it for prediction.  
(https://github.com/mplockhart/66days/blob/main/codecademy_learning/multiple_linear_regression.ipynb)
I also began the 'Yelp' data science portfolio project. So far I have improted and worked with the data before merging it. Next time I will clean the data, perform eDA and train models.
(https://github.com/mplockhart/codecademy_projects/blob/main/yelp_regression_project/.ipynb_checkpoints/yelp_regression-checkpoint.ipynb)

## Day 5 (18/03/2022)
Having only scratched the surface of yesterday's Yelp analysis I finisdhed it today. It was useful to see what is required to be dropped in the cleaning phase. All non-numerical attributes were dropped. Then it was interesting to see how a binary Vs numerical dofference could change the outcome. All in all the model prediction allows one to tune their sales in order to maximise their outputs.  
The best example was when I was looking at optimising the parameters of the new shop at the end of the project I noticed that by changing the 'number of review' and increasing them by one order of magnitude up, the results was > 5, meaning that if the model were correct (which it appeared  to be with an R^2 > 0.7) the shop was well placed to have a high rating if the reviews kept coming in.
(https://github.com/mplockhart/codecademy_projects/blob/main/yelp_regression_project/.ipynb_checkpoints/yelp_regression-checkpoint.ipynb)

## Day 6 (19/03/2022)
Today I used EDA and regression models to look at the prediction power of single, double and multiple linear regression models. IT was useful to generate a for loop to iterate through all of the individual features to see which had highest correlation. the ultimate prediction models was using all of the features, even though some contributed only a little.
(https://github.com/mplockhart/codecademy_projects/blob/main/tennis_ace_starting/tennis_ace.ipynb)

## Day 7 (20/03/2022)
This was a relatively short day but I did learn avout k-nearest neighboursand the requirement to normalise data. Initially this was a min-max normalisation to give a value between 0 and 1:  

(value - minimum)/(maximum - minimum)  
As usual, this was initially implemented manually through a function.  
Next it was to define the nearest neighbours using Pythagoras' formula for any number of dimensions and slice the resulting data list using 'k'.
Finally the classification method uses all of the above to look how an unknow film is compared to labelled data. IT was also useful so know that id there is a tie with an even number of 'k', it is often said that the first point can be used to break the tie.

## Day 8 (21/03/2022)
A continuation of k-nearest neighbours looking for a bad film. While the new James Bond film was shown to be a 'good' film (https://www.imdb.com/title/tt2382320/), The King's Daughter (https://www.imdb.com/title/tt2328678/?ref_=ttls_li_tt) was shown to be a 'bad' film, and boradly agreed with the critics responses.  
Next was a test to validate the data already curated to see if the ML model was indeed correct, which it appeared to be.  
The next step was looking at how to define 'k'. This basically comes from the following:  
### K-nearest neighbours
### Overfitting
This is where 'k' is low and not enough neighbours are considered. K is too small so outliers dominate the result.  
### Under fitting
Here 'k' is large and takes in too many neighbours. This could be problematic on a boundary of different data points. k is too big so larger trends in the data set aren't represented.  
### Validation error
This is the number of correct guesses over the total number of movies. In this case with 3 features it was 66%.
### Graph of K
This is used to graph the optimum number of K by running the method multiple times.

As per usual, the lessons conclude with the sklearn implementation (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). It goes through the same steps of fit, and then predict to get the trained results. 

## Day 9 (22/03/2022)
Finalising the KNN section of the course. It involved the usage of a project data set for breat cancer diagnosis. The data had many (~30) attributes and >500 readings of cells. The KNN was initially selected with K=3, but then a look was used to calculate the best result. This was ~22 and showed accuracy of >96% (https://github.com/mplockhart/codecademy_projects/blob/main/breast_cancer_knn/breast_cancer_knn.ipynb).

### KNN as a regressor
This was the final part of the section, where instead of classsification with KNN, it was used as a predictor.  

The use of weighted averages when calculating distance to neighbours can be used to weight the closer neighbours higher than those further away.  

For example, the numerator is the sum of every rating divided by their respective distances. The denominator is the sum of one over every distance (https://www.codecademy.com/paths/data-science/tracks/dscp-foundations-of-machine-learning-supervised-learning/modules/dscp-supervised-learning-introduction-to-classification-with-k-nearest-neighbors/lessons/ml-knn-regression/exercises/weighted-regression). This was initially calculated by manually generating methods before uasing sklearn.

The difference with sklearn is that the command now adds `KNeighborsRegressor(n_neighbors = 3, weights = "distance")` when we create the regressor.


## Day 10 (23/03/2022)[2 years since lockdown started]
## Analysing model performance
Today is a look at evaluating and improving he models that have been made. Methods to do this include, accuracy, precision, recall and F1 score.

### Accuracy
The simplest way of reporting the effectiveness of an algorithm is by calculating its accuracy. Accuracy is calculated by finding the total number of correctly classified points and dividing by the total number of points. This includes `TRUE POSITIVE + TRUE NEGATIVE as the numeator and TOTAL RESULTS` as the denominator.

### Recall
Accuracy can be an extremely misleading statistic depending on your data. Consider the example of an algorithm that is trying to predict whether or not there will be over 3 feet of snow on the ground tomorrow. We can write a pretty accurate classifier right now: always predict False. This classifier will be incredibly accurate — there are hardly ever many days with that much snow. But this classifier never finds the information we’re actually interested in.

In this situation, the statistic that would be helpful is recall. Recall measures the percentage of relevant items that your classifier found. In this example, recall is the number of snow days the algorithm correctly predicted divided by the total number of snow days. This is `TRUE POSITIVES over TRUE POSITIVES + FALSE NEGATIVES`.

Our algorithm that always predicts False might have a very high accuracy, but it never will find any True Positives, so its recall is 0. This makes sense; recall should be very low for such an absurd classifier.

### Precision
Unfortunately, recall isn’t a perfect statistic either. For example, we could create a snow day classifier that always returns True. This would have low accuracy, but its recall would be 1 because it would be able to accurately find every snow day. But this classifier is just as nonsensical as the one before! The statistic that will help demonstrate that this algorithm is flawed is precision.

In the snow day example, precision is the number of snow days the algorithm correctly predicted divided by the number of times it predicted there would be a snow day. This is `TRUE POSITIVES iver TRUE POSITIVES + FALSE POSITIVES`.

The algorithm that predicts every day is a snow day has recall of 1, but it will have very low precision. It correctly predicts every snow day, but there are tons of false positives as well. **Precision and recall are statistics that are on opposite ends of a scale. If one goes down, the other will go up.**

### F1 Score
It is useful to consider the precision and recall of an algorithm, however, we still don’t have one number that can sufficiently describe how effective our algorithm is. This is the job of the F1 score — F1 score is the harmonic mean of precision and recall. The harmonic mean of a group of numbers is a way to average them together. That is `2 * ((PRECISION * RECALL) / (PRECISION + RECALL))`.

The F1 score combines both precision and recall into a single statistic. We use the harmonic mean rather than the traditional arithmetic mean because we want the F1 score to have a low value when either precision or recall is 0. For example, consider a classifier where `recall = 1` and precision = `0.01`. We know that there is most likely a problem with this classifier since the precision is so low, and so we want the F1 score to reflect that. Using the *arithmetic mean* we would get `(1+ 0.01) / 2 = 0.505` which is too high. The *harmonic mean* is `2* (1*0.01) / (1+0.01) = 0.019` which represents this lower precision.

### Overall
- Classifying a single point can result in a true positive (truth = 1, guess = 1), a true negative (truth = 0, guess = 0), a false positive (truth = 0, guess = 1), or a false negative (truth = 1, guess = 0).
- Accuracy measures how many classifications your algorithm got correct out of every classification it made.
- Recall measures the percentage of the relevant items your classifier was able to successfully find.
- Precision measures the percentage of items your classifier found that were actually relevant.
- Precision and recall are tied to each other. As one goes up, the other will go down.
- F1 score is a combination of precision and recall.
- F1 score will be low if either precision or recall is low.

### sklearn
sklearn is able to calculate all of these metrics using `sklearn.metrics` and importing the corresponding required scoring metrics. They require a list of the correct labels and guesses. 

# Day 11 (24/03/2022)
## Data Analysts Project
This was a project from Codecademy where atatistical tests were combined to observe if the cohort studied were statistically different from each other, in terms of cholesterol levels of people with heart disease being greater than the upperhealthy limit (1 sample t-test), and seeing if the cohots was representative of the general population with regards to their fasting blood sugar levels using a binomial test (they were not). This can be found (https://github.com/mplockhart/codecademy_projects/blob/main/Heart_Disease_Research_Part_I/heart_disease_1.ipynb).
