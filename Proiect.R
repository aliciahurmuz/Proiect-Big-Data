install.packages("modelr")
install.packages("scatterplt3d")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")
install.packages("rsample")
install.packages("dplyr")
install.packages("ggplot")
library(tidyverse)
library(modelr)
library(scatterplot3d)
library(rpart)
library(rpart.plot)
library(caret)
library(rsample)
library(dplyr)
library(ggplot2)

data_house <- read_csv("kc_house_data.csv")
data_house <- na.omit(data_house)

data_house <- data_house %>%
  select(price, sqft_living, bedrooms, bathrooms, floors, yr_built)
View(data_house)

data_house %>%
  ggplot(aes(sqft_living, price)) + geom_point()+ geom_smooth()

data_house %>%
  ggplot(aes(bedrooms, price)) + geom_point() + geom_smooth()

data_house %>%
  ggplot(aes(bathrooms, price)) + geom_point() + geom_smooth()

mod_price_sqft_living <- lm(data = data_house, price ~ sqft_living)
summary(mod_price_sqft_living)

grid_sqft_living <- data_house %>% 
  data_grid(sqft_living = seq_range(sqft_living, 100)) %>%
  add_predictions(mod_price_sqft_living, "price")

ggplot(data_house, aes(sqft_living, price)) + 
  geom_point() +  
  geom_line(data = grid_sqft_living, color = "red", size = 1) 

confint(mod_price_sqft_living)  

mod_price_bedrooms <- lm(data = data_house, price ~ bedrooms)    
summary(mod_price_bedrooms)

grid_bedrooms <- data_house %>% 
  data_grid(bedrooms = seq_range(bedrooms, 100)) %>%
  add_predictions(mod_price_bedrooms, "price")

ggplot(data_house, aes(bedrooms, price)) + 
  geom_point() +  
  geom_line(data = grid_bedrooms, color = "blue", size = 1) 

confint(mod_price_bedrooms)

mod_price_bathrooms <- lm(data = data_house, price ~ bathrooms)    
summary(mod_price_bathrooms)

grid_bathrooms <- data_house %>% 
  data_grid(bathrooms = seq_range(bathrooms, 100)) %>%
  add_predictions(mod_price_bathrooms, "price")

ggplot(data_house, aes(bathrooms, price)) + 
  geom_point() +  
  geom_line(data = grid_bathrooms, color = "yellow", size = 2) 

confint(mod_price_bathrooms)

mod_price_floors <- lm(data = data_house, price ~ floors)    
summary(mod_price_floors)

grid_floors <- data_house %>% 
  data_grid(floors = seq_range(floors, 100)) %>%
  add_predictions(mod_price_floors, "price")

ggplot(data_house, aes(floors, price)) + 
  geom_point() +  
  geom_line(data = grid_floors, color = "green", size = 1) 

confint(mod_price_floors)

mod_price_yr_built <- lm(data = data_house, price ~ yr_built)    
summary(mod_price_yr_built)

grid_yr_built <- data_house %>% 
  data_grid(yr_built = seq_range(yr_built, 100)) %>%
  add_predictions(mod_price_yr_built, "price")

ggplot(data_house, aes(yr_built, price)) + 
  geom_point() +  
  geom_line(data = grid_yr_built, color = "purple", size = 1) 

confint(mod_price_yr_built)

mod_price_all <- lm(data = data_house, price ~ sqft_living + bedrooms + bathrooms + floors + yr_built) 
summary(mod_price_all)
confint(mod_price_all)

new_house <- tibble(
  sqft_living = 2500,
  bedrooms = 4,
  bathrooms = 2.5,
  floors = 2,
  yr_built = 2005
)
predict(mod_price_all, newdata = new_house, interval = "confidence")
predict(mod_price_all, newdata = new_house, interval = "prediction")

predictions_all <- predict(mod_price_all, data_house)
rmse_all <- RMSE(predictions_all, data_house$price)
print(rmse_all)



set.seed(123)
house_split <- initial_split(data_house, prop = 0.7)
house_train <- training(house_split)
house_test <- testing(house_split)
house_train %>%
  select_if(is.numeric) %>%
  gather(metric, value) %>%
  ggplot(aes(value, fill = metric)) +
  geom_density(show.legend = FALSE) +
  facet_wrap(~metric, scales = "free")


m1 <- rpart(
  formula = price ~ ., 
  data = house_train,
  method = "anova",
  control = list(cp = 0.01)
)
m1
rpart.plot(m1) 
plotcp(m1)
m1$cptable  

m2 <- rpart(
  formula = price ~ ., 
  data = house_train,
  method = "anova",
  control = list(cp = 0, xval = 10)  
)  
plotcp(m2)
abline(v = 12, lty = "dashed") 

m3 <- rpart(
  formula = price ~ .,
  data = house_train, 
  method = "anova",
  control = list(minsplit = 10, maxdepth = 12, xval = 10)
)
m3
plotcp(m3)

hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)
head(hyper_grid)
models <- list()
for (i in 1:nrow(hyper_grid)) {
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  models[[i]] <- rpart(
    formula = price ~. ,
    data = house_train,
    method = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}
get_cp <- function(x) {
  min <- which.min(x$cptable[,"xerror"])
  cp <- x$cptable[min, "CP"]
}
get_min_error <- function(x) {
  min <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"]
}

mutated_grid <- hyper_grid %>%
  mutate(
    cp = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  )  
mutated_grid %>%
  arrange(error) %>%
  top_n(-5, wt=error)

optimal_tree <- rpart(
  formula = price ~ .,
  data = house_train,
  method = "anova",
  control = list(minsplit = 5, maxdepth = 12, cp = 0.01000000)
)


pred <- predict(m1, newdata = house_test)
rmse_tree <- RMSE(pred = pred, obs = house_test$price)
rmse_tree
optimal_tree
