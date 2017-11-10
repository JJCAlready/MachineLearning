# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('Empleo_dataset.csv')
dataset = dataset[, 3:4]
Error in sample.split: 'SplitRatio' parameter has to be i [0, 1] range or [1, length(Y)] range
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Sueldo.Maximo, SplitRatio = 0.3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Sueldo.Maximo ~ ., 
               data = training_set)
# Checking the correlation
summary(regressor)

# Fixing the formula
regressor = lm(formula = Sueldo.Maximo ~ Experiencia.Minima.en.meses, 
               data = training_set)


# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)