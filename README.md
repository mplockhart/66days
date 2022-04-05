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

I also finalised the second project which used the same data set but used the below tests (https://github.com/mplockhart/codecademy_projects/blob/main/Heart_Disease_Research_Part_I/heart_disease_2.ipynb).

## Hypothesis testing (t-test, ANOVA, or Tukey’s range test)
Before we use a two sample t-test, ANOVA, or Tukey’s range test, we need to be sure that the following things are true:

1. The observations should be independently randomly sampled from the population  
Suppose the population we are interested in is all visitors to a website. Random sampling will help ensure that our sample is representative of the population we care about. For example, if we only sample site visitors on Halloween, those visitors may behave differently from the general population. In practice, this can be a challenging assumption to meet, but it’s important to be aware of.

2. The standard deviations of the groups should be equal  
For example, if we’re comparing time spent on a website for two versions of a homepage, we first want to make sure that the standard deviation of time spent on version 1 is roughly equal to the standard deviation of time spent on version 2. To check this assumption, it is normally sufficient to divide one standard deviation by the other and see if the ratio is “close” to 1. Generally, a ratio between 0.9 and 1.1 should suffice.

That said, there is also a way to run a 2-sample t-test without assuming equal standard deviations — for example, by setting the equal_var parameter in the scipy.stats.ttest_ind() function equal to False. Running the test in this way has some disadvantages (it essentially makes it harder to reject the null hypothesis even when there is a true difference between groups), so it’s important to check for equal standard deviations before running the test.

3. The data should be normally distributed…ish  
Data analysts in the real world often still perform these tests on data that are not normally distributed. This is usually not a problem if sample size is large, but it depends on how non-normal the data is. In general, the bigger the sample size, the safer you are!

4. The groups created by the categorical variable must be independent  
Here are some examples where the groups are not independent:

the number of goals scored per soccer player before, during, and after undergoing a rigorous training regimen (not independent because the same players are measured in each category)
years of schooling completed by a group of adults compared to their parents (not independent because kids and their parents can influence one another)

## Chi squared test
Before we use a Chi-Square test, we need to be sure that the following things are true:  

1. The observations should be independently randomly sampled from the population  
This is also true of 2-sample t-tests, ANOVA, and Tukey. The purpose of this assumption is to ensure that the sample is representative of the population of interest.

2. The categories of both variables must be mutually exclusive  
In other words, individual observations should only fall into one category per variable. This means that categorical variables like “college major”, where students can have multiple different college majors, would not be appropriate for a Chi-Square test.

3. The groups should be independent  
Similar to 2-sample t-tests, ANOVA, and Tukey, a Chi-Square test also shouldn’t be used if either of the categorical variables splits observations into groups that can influence one another. For example, a Chi-Square test would not be appropriate if one of the variables represents three different time points.

# Day 12 (25/03/2022)
I completed several projects utilising hypothesis testing. These included a theoretical start up called Familiar selling blood transfusions and looking to see if there was a significant difference between the packs sold and the lifespan of the recipients.

## Familiar
(https://github.com/mplockhart/codecademy_projects/blob/main/familiar-blood_transfusion-hypothesis_testing/familiar-blood_transfusion.ipynb)
### 1 sample t-test
This was to compare the vein transfusion pack to 73 years of age using a one sample t-test due to there being only one data source compared against a fixed value. This was significant with a 0.05 cutoff.
### Independant t-test
This was to compare the lifespan of the vein pack to that of the upgraded artery pack. This was a comparing 2 different independent data sets so the independent t-test.
### Chi2 test
Determining if there is an association between the different packs and the iron levels of the patients using  chi^2 contingency test.

## Fetch Maker
https://github.com/mplockhart/codecademy_projects/blob/main/fetchmaker-hypothesis_testing/fetchmaker.ipynb
This again used various hypothesis tests to compare characteristics of dogs.

### Binomial testing
We wanted to see if our sample of whippets that were rescue dogs were representative of the global population of whippets that have a rate of 8%. Because this was a comparison of a catagorical (rescue vs non-rescue) we used a binomial test.

### ANOVA and Tukey
The next example was to compare the weight (mass) of the dogs in the 'whippet, terrier, pitbull' category. The null hypothesis is that the dogs were all the same weight on average. This was shown not to be true by ANOVA and to see which breeds were significantly differernt we used Tukey to compare.

### Chi squared
We finalised by looking at the association of dog breeds and colour. The breeds here are shihtzu and poodle.

## Analyzing Farmburg's A/B Test
This final one was interesting and has an interesting twist. It involves A/B testing with three groups based on costs of new items on the farm simulation game. 

### Chi squares test
Here 'Brian' uses a chi squared test to compare the differences of A, B and C because this would show which option would be better. 

It then turns out that while this was sound it was not business sound. We needed to see if the $0.99, $1.99 or $4.99 prices with the new purchase rates are viable. Here we thing around $1000 is required per week to make this viable. While it seems redundant that the $0.99 would have significantly higher sales, does it make enough money.

### Binomial testing
From here we needed to determine how many sales at each price point are required, how big the sample sizes are and run a binomial test on each to see if the amount sold was significantly higher than the proportion needed to break even for $1000.

Surprisingly it was the most expensive option that made most business sense. While the fewest people purchased it, the numbers were significantly higher than were needed to break $1000.

## Interview
I had a preliminary interview for a role as a DS with UCAS. Next is the numerical reasoning test.

# Day 13 (26/03/2022)
Moving back onto the machine learning aspect of DS I started with logistic regression. This was a quick introduction due to time constraints but I went through the initial information of how it was different to logistic regression. The main difference is tha tit is a classified with a binary probability. This comes from the sigmoidal function applied rather than a linear regression line which will give values that will ultimately go to +/1 infinity. One readon this is bad is that a question could be "If I study for 10 hours will I become a data scientist?". The answer is yes or no (simplistically), which is 0 or 1. Depending on other labelled values 10 hours of study could equal -50, which is in effect 0, or +50, which is in effect 1, but if there are middling values it doesn't take that into account.

# Day 14 (27/03/2022)
Today was another short day. I took some time to go back through my own data analysis if my golf shots to see if there were any methological improvements or general ones to make. As it happened I noticed sever mistakes in my data input. This was a mistake with dropping in Excel, which propagated to the CSV file. This was corrected. I also looked more into the distributions of my golf shots. My last 2 sessions with my 5i we no longer statistically different. I need to expand this work to my other clubs and dates when possible. 
https://github.com/mplockhart/golf_stats/blob/main/hooked_shots_high_legh.ipynb

# Day 15 (28/09/2022)
Building on the logisitc regression I finished the Codecademy section and tried out some project work

## Logistic regression - Breast cancer
This was loaded from the sklearn modules. The data were standardised using the standard scalar ((x - mean) / standard deviation). This is becuase the logistic regression ML models require normalisaiton. All of the data were used and the nsplit to X and y. This was then split to train test sets using sklearn's model selection module before creating the model, fitting it to the data and the npredicting the outcomes from the test set.

The confusion matrix was introduced heren and provided information of the true and false positives and negatives. This was interesting and very applicable.
https://github.com/mplockhart/66days/blob/main/codecademy_learning/logistic_regression.ipynb

## Logistic regression - Titanic project 
It was enjoyable to use a real world example of data to see how a simple system such as logistic regression could predict the outcome so well.
https://github.com/mplockhart/codecademy_projects/blob/main/titanic_logistic_regression/titanic_survival_prediction.ipynb

In addition to the breast cancer data, there were missing data here whic hrequired clearning, and the construction of new data points such as 'FirstClass' and 'SecondClass' columns.

The data were forst split to the train test data before fitting the scalar to the X_train and transforming the X_test data. The logistic regression model were fitted and the score of the train model Vs the test mode were show nto be ~ 0.77 Vs 0.84. The coefficients were then printed out and it showed that sex was the more most important identity. Thinking of the stories of the Ttianic this makes sense.

Having a prediction mode I could then add my own data for precition.  
jack = np.array([0., 20., 0., 0.])  
rose = np.array([1., 17., 1., 0.])  
mike = np.array([0., 32., 1., 0.])  
ali = np.array([1., 30., 0., 1.])  

These needed to be combines to a NumPy array and then applied to the model. Broadly, if you were a man you would die, but I did print the probabilities and saw that first class men were ~50:50.

# Day 16 (29/03/2022)
Today was a tough one. I have always struggled to get me head around Bayes theorem. Bayes probability overall looks like:

<img src="https://latex.codecogs.com/svg.image?P(A|B)&space;=&space;\frac{P(B|A).P(A)}{P(B)}">

I just need to spend time making asure I know how to calculate each of these problems.

# Day 17 (30/03/2022)
Today I spent time looking at more Bayes work and sitting to hand write a lot of examples out. 

## Testing
In additiona I completed the RANRA as part of a job application to UCAS. This was not too taxing although I was quite unprepeared for the type of question so I am unsure of what the results will hold.

## Writing equations in GitHub
After finding that standard LaTeX doesn't work I followed the advice [here](https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b) to use a URL generator such as [this](https://latex.codecogs.com/). This has made life much easier. I need to go back and adapt some notes.

# Day 18 (31/03/2022)
In my day job as a scientist, I work with all sorts of data. Today I was wanting to replot a graph from an output and use the slope and the intercept to predict resolution or numbers of particles required.  
Briefly, this consisted of:
- Importing the data 
- Transforming the x-axis by taking the log10 of each value
- Inverting the y-axis values
- Training a linear regression model on the highest resolution elements from te y-axis values
- Using the coefficients and the intercept, calculate the numberof particles, or resolution able to be obtained given the resolution required, or particles obtained.
https://github.com/mplockhart/academic_explore/blob/main/reslog/reslog_linear_regression.ipynb

# Day 19 (01/04/2022)
I missed this day travelling on holidays.

# Day 20 (02/04/2022) - Day 22 (04/04/2022)
I was doing some work to make sure I knew how to comprehend Bayes probability but I had little time on holiday.

# Day 21 (05/04/2022)
https://github.com/mplockhart/66days/blob/main/codecademy_learning/bayes_theorem.ipynb
