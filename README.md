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
