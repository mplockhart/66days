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
