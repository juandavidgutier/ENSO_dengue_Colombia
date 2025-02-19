library(dplyr)
library(MKdescr)
library(ggplot2)


#read dataset
data_all <- read.csv("D:/clases/UDES/articulo dengue/ltmle/ci/data_DAG.csv")
data_all <- select(data_all, DANE, excess, S3, S4, S34, SOI, NATL, TROP, 
                   UBN, Temp, Rain, invest_health, Pop_density, 
                   NeutralvsLaNina) #change it to "NeutralvsElNino" and "LaNinavsElNino"
                                    #to obtain the other figures

#subset complete cases
data <- data_all[complete.cases(data_all), ] 

#sd units
data$S3 <- zscore(data$S3, na.rm = TRUE) 
data$S34 <- zscore(data$S34, na.rm = TRUE) 
data$S4 <- zscore(data$S4, na.rm = TRUE)
data$SOI <- zscore(data$SOI, na.rm = TRUE) 
data$NATL <- zscore(data$NATL, na.rm = TRUE) 
data$TROP <- zscore(data$TROP, na.rm = TRUE)
data$UBN <- zscore(data$UBN, na.rm = TRUE) 
data$Temp <- zscore(data$Temp, na.rm = TRUE)
data$Rain <- zscore(data$Rain, na.rm = TRUE) 
data$invest_health <- zscore(data$invest_health, na.rm = TRUE) 
data$Pop_density <- zscore(data$Pop_density, na.rm = TRUE)



# Fit a logistic regression model to estimate the propensity scores
propensity_model <- glm(NeutralvsLaNina ~ S3 + S4 + S34 + SOI + 
                        NATL + TROP + 
                        UBN + Temp + Rain + invest_health + Pop_density  
                        , data = data, family = "binomial")

# Extract the propensity scores from the model
propensity_scores <- predict(propensity_model, type = "response")

# Add propensity scores to the data
data <- cbind(data, propensity_score = propensity_scores)

# Check the distribution of propensity scores
summary(data$propensity_score)

# Generate example data (replace this with your actual data)
set.seed(123)
data <- data.frame(
  NeutralvsLaNina = sample(0:1, 100, replace = TRUE),
  propensity_score = runif(100)
)

# Plot the distribution overlap
ggplot(data, aes(x = propensity_score, fill = factor(NeutralvsLaNina))) +
  geom_density(alpha = 0.5) +
  labs(x = "Propensity Score", y = "Density", 
       fill = "NeutralvsLa Niña") +  # Adjust label text size here
  ggtitle("a") +
  theme(plot.title = element_text(size = 34, hjust = 0.5),  # Adjust title font size
        axis.title = element_text(size = 26),  # Adjust axis labels font size
        axis.text = element_text(size = 24), # Adjust axis values font size 
        legend.text = element_text(size = 24),  
        legend.title = element_text(size = 24),
  )


