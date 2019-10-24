library(lfe)
library(dplyr)
library(stringr)

data <- read.table('synthetic_data.csv', header = TRUE, sep=",")
data$X <- NULL  # remove empty column


data$Player <- factor(data$Player) #turn Player column into factor
data$Hole <- factor(data$Hole)     #turn Hole column into factor


est <- felm( data$Distance ~ data$Player + data$Hole) # using large fixed effects regression to 'predict' the average for each player and hole
summary(est)
df <- est$coefficients
df <- as.data.frame(df)
df <- tibble::rownames_to_column(df, "Factor") # dataframe has columns : hole x or player y, average increase in distance for hole x or player y

print(df)

write.csv(df, file = 'FE_regression.csv') # export back to Python to work with data
