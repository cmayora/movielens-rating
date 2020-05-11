## Carlos Mayora
## HarvardX: PH125.9x - Capstone Project
## MovieLens Project
## April 29, 2020

#################################################
# MovieLens Prediction Project 
################################################

###################################
## Data Analysis ##
###################################

###################################
##### Data Ingestion #####

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################
##### Data Exploration #####
#6 first rows of edx dataset including column names
head(edx)

# Basic summary statistics
summary(edx)

# edx rows and columns
dim(edx)

# validation rows and columns
dim(validation)

# Number of unique users and movies in the edx dataset 
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

###################################
###### Ratings ######

# Histogram: Number of ratings
edx %>% mutate(rating_group = ifelse(rating %in% c(1,2,3,4,5), "whole_rating", "half_rating")) %>%
  ggplot(aes(x=rating, fill=rating_group)) +
  geom_histogram(binwidth = 0.5) +
  scale_x_discrete(limits = c(seq(0, 5, 0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  xlab("rating") +
  ylab("count") +
  ggtitle("Histogram: Number of ratings")

###################################
###### Movies ######

# Histogram: Number of ratings per movie
edx %>% 
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins = 30) +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Histogram: Number of ratings per movie")

###################################
###### Users ######

# Histogram: Number of ratings per user
edx %>% 
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins = 30) +
  scale_x_log10() +
  xlab("Number or ratings") +
  ylab("Number of users") +
  ggtitle("Histogram: Number of ratings per user")

###################################
##### Data Preparation #####

# Let's split the edx dataset in two, train and test sets
# validation set will be used to calculate the final RMSE
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-edx_test_index,]
edx_temp <- edx[edx_test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- edx_temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
edx_removed <- anti_join(edx_temp, test_set)
train_set <- rbind(train_set, edx_removed)

rm(edx_test_index, edx_temp, edx_removed)

###################################
##  Modeling ##
###################################

###################################
##### Model Selection #####

# RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

###################################
##### Just the Average Model #####

# Average of all ratings
mu <- mean(train_set$rating)
mu

# Just average model RMSE
avg_rmse <- RMSE(test_set$rating, mu)
avg_rmse
cat("The RMSE using the Just Average Model is: ", avg_rmse)

# Save results in Data Frame, including the project goal RMSE
rmse_results = data_frame(Method = "Project Goal", RMSE = 0.86490)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Just the Average Model", RMSE = avg_rmse ))
# Check the results
rmse_results %>% knitr::kable()

###################################
##### The Movie Effect Model #####

# Most rated movies and their rating
edx %>% group_by(movieId, title) %>%
  summarize(count = n(), mean(rating)) %>%
  arrange(desc(count)) %>% head(10)

# Less rated movies and their rating
edx %>% group_by(movieId, title) %>%
  summarize(count = n(), mean(rating)) %>%
  arrange(count) %>% head(10)

# Calculating b_i per movie
movies_mean <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Histogram of movie effect distribution
movies_mean %>% ggplot(aes(b_i)) +
  geom_histogram(color = "black", bins = 10) +
  xlab("b_i") +
  ylab("Number of movies") +
  ggtitle("Histogram: Movie Effect Distribution")

# Movie Effect Model RMSE
predicted_ratings <- test_set %>% 
  left_join(movies_mean, by='movieId') %>%
  mutate(pred_rating = mu + b_i) %>%
  pull(pred_rating)

movie_rmse <- RMSE(test_set$rating, predicted_ratings)

# Add the Movie Effect Model RMSE to the results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie Effect Model", RMSE = movie_rmse ))
# Check the results
rmse_results %>% knitr::kable()

###################################
##### The Movie and User Effect Model #####

# Histogram of user effect distribution
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n() >= 100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(color = "black", bins = 30) +
  xlab("b_u") +
  ylab("Number of users") +
  ggtitle("Histogram: User Effect Distribution")

# Calculating b_u per user
users_mean <- train_set %>% 
  left_join(movies_mean, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# User Effect Model RMSE
predicted_ratings <- test_set %>% 
  left_join(movies_mean, by='movieId') %>%
  left_join(users_mean, by='userId') %>%
  mutate(pred_rating = mu + b_i + b_u) %>%
  pull(pred_rating)

user_rmse <- RMSE(test_set$rating, predicted_ratings)

# Add the Movie Effect Model RMSE to the results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie and User Effect Model", RMSE = user_rmse ))
# Check the results
rmse_results %>% knitr::kable()

###################################
##### Regularaization #####

# 10 largest mistakes from Movie Effect Model
test_set %>% 
  left_join(movies_mean, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  slice(1:10) %>% 
  select(movieId, title, residual) %>%
  knitr::kable()

# Movies ids and titles
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

# 10 best movies adding number of ratings
train_set %>% count(movieId) %>% 
  left_join(movies_mean, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(movieId, title, b_i, n) %>% 
  head(10) %>% 
  knitr::kable()

# 10 worst movies adding number of ratings
train_set %>% count(movieId) %>% 
  left_join(movies_mean, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(movieId, title, b_i, n) %>% 
  head(10) %>% 
  knitr::kable()

# lambda is the tuning parameter
# We use cross-validation to choose it.
lambdas <- seq(0, 10, 0.25)

# for each lambda we find the movie (b_i) and user (b_u) effect and predict 
rmses <- sapply(lambdas, function(l){
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# Plot rmses vs lambdas to visualize the optimal lambda
qplot(lambdas, rmses, main = "Plot: Regularization", xlab = "Lambdas", ylab = "RMSEs")

# The optimal lambda
lambda <- lambdas[which.min(rmses)]
cat("The optimal lambda is: ", lambda)

# Add the Regularization Model RMSE to the results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Regularized Movie and User Effect Model", 
                                     RMSE = min(rmses) ))
# Check the results
rmse_results %>% knitr::kable()

###################################
## Matrix Factorization

# Install recosystem package
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead

# Create the reco model object
r = Reco()

# Convert train and test sets
train_reco <- with(train_set, data_memory(user_index = userId, 
                                          item_index = movieId, rating = rating))
test_reco <- with(test_set,  data_memory(user_index = userId, 
                                         item_index = movieId, rating = rating))

# Select best tuning parameters
opts <- r$tune(train_reco, opts = list(dim = c(10, 20, 30), lrate = c(0.01, 0.1),
                                       costp_l1 = 0, costp_l2 = c(0.01, 0.1),  # user cost
                                       costq_l1 = 0, costq_l2 = c(0.01, 0.1),  # movie cost
                                       nthread  = 4, niter = 10)) # threads and iterations

# Train the model
r$train(train_reco, opts = c(opts$min, nthread = 4, niter = 40))

# Compute the prediction
predict_reco <- r$predict(test_reco, out_memory())

fact_rmse = RMSE(test_set$rating, predict_reco)

# Add the Matrix Factorization Model RMSE to the results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Matrix Factorization - Recosystem", 
                                     RMSE = fact_rmse ))
# Check the results
rmse_results %>% knitr::kable()

###################################
##  Results Analysis ##
###################################

###################################
##### Regularization with Validation Set #####

# edx rating mean
mu_edx <- mean(edx$rating)

# Movie effect using edx
b_i_edx <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# User effect using edx
b_u_edx <- edx %>% 
  left_join(b_i_edx, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

predicted_ratings <- 
  validation %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  mutate(pred = mu_edx + b_i + b_u) %>%
  .$pred

reg_validation_rmse <- RMSE(validation$rating, predicted_ratings)

# Add the Regularization with Validation Set Model RMSE to the results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Regularized Movie and User Effect Model with Validation Set", 
                                     RMSE = reg_validation_rmse))
# Check the results
rmse_results %>% knitr::kable()

###################################
##### Matrix Factorization with Validation Set #####

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead

# Create the reco model object
r = Reco()

# Convert edx and validation sets
edx_reco <- with(edx, data_memory(user_index = userId, 
                                  item_index = movieId, rating = rating))
validation_reco <- with(validation,  data_memory(user_index = userId, 
                                                 item_index = movieId, rating = rating))

# Select best tuning parameters
opts <- r$tune(edx_reco, opts = list(dim = c(10, 20, 30), lrate = c(0.01, 0.1),
                                     costp_l1 = 0, costp_l2 = c(0.01, 0.1),  # user cost
                                     costq_l1 = 0, costq_l2 = c(0.01, 0.1),  # movie cost
                                     nthread  = 4, niter = 10)) # threads and iterations

# Train the model using same tuning parameters
r$train(edx_reco, opts = c(opts$min, nthread = 4, niter = 40))

# Compute the prediction
predict_reco <- r$predict(validation_reco, out_memory())

fact_validation_rmse <- RMSE(validation$rating, predict_reco)

# Add the Matrix Factorization Model RMSE to the results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Matrix Factorization - Recosystem with Validation Set", 
                                     RMSE = fact_validation_rmse ))
# Check the results
rmse_results %>% knitr::kable()
