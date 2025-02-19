library(ggdag)
library(dagitty)
library(lavaan)
library(CondIndTests)
library(dplyr)
library(GGally)
library(tidyr)
library(MKdescr)
library(caret)


#################################################################################
#episodes
#implied Conditional Independencies
dataset <- read.csv("D:/data_DAG.csv")
str(dataset)
dataset <- select(dataset, excess, S3, S4, S34, S12, ESOI, SOI, 
                  NATL,	SATL, TROP, UBN, Temp, Rain, 
                  Pop_density,    #change it to "invest_health" to obtain the analysis for population density covariate
                  NeutralvsLaNina)  #change it to "NeutralvsElNino" and "LaNinavsElNino"to obtain the others analysis
str(dataset)

#sd units
dataset$S12 <- zscore(dataset$S12, na.rm = TRUE)  
dataset$S3 <- zscore(dataset$S3, na.rm = TRUE) 
dataset$S34 <- zscore(dataset$S34, na.rm = TRUE) 
dataset$S4 <- zscore(dataset$S4, na.rm = TRUE)
dataset$ESOI <- zscore(dataset$SOI, na.rm = TRUE) 
dataset$SOI <- zscore(dataset$SOI, na.rm = TRUE) 
dataset$NATL <- zscore(dataset$NATL, na.rm = TRUE) 
dataset$SATL <- zscore(dataset$SATL, na.rm = TRUE) 
dataset$TROP <- zscore(dataset$TROP, na.rm = TRUE)
dataset$UBN <- zscore(dataset$UBN, na.rm = TRUE) 
dataset$Temp <- zscore(dataset$Temp, na.rm = TRUE) 
dataset$Rain <- zscore(dataset$Rain, na.rm = TRUE) 
dataset$Pop_density<- zscore(dataset$Pop_density, na.rm = TRUE) 



#exclude co-variables with correlation > 0.8
df2 = cor((dataset), use = "pairwise.complete.obs")
hc = findCorrelation(df2, cutoff=0.85) # putt any value as a "cutoff" 
hc = sort(hc)
reduced_Data = dataset[,-c(hc)]
str (reduced_Data)

reduced_Data <- reduced_Data[complete.cases(reduced_Data), ] 
str(reduced_Data)


#DAG 
dag <- dagitty('dag {
excess [pos="1, 0.5"]
NeutralvsLaNina  [pos="-1, 0.5"]
S3 [pos="-1.8, 1.3"]
S34 [pos="-2, 1.5"]
S4 [pos="-1.9, 1.4"]
SOI [pos="-1.6, 1.1"]
NATL [pos="-2.2, 1.7"]
TROP [pos="-2.4, 1.9"]
Rain [pos="-1.0, -2.5"] 
Temp [pos="0, -2.0"] 
UBN [pos="0.5, -1.25"]
Pop_density [pos="0.5, 1.25"]


S3 -> S34
S3 -> S4
S3 -> SOI
S3 -> NATL
S3 -> TROP


S34 -> S4
S34 -> SOI
S34 -> NATL
S34 -> TROP


S4 -> SOI
S4 -> NATL
S4 -> TROP


SOI -> NATL
SOI -> TROP


NATL -> TROP



S3 -> NeutralvsLaNina
S34 -> NeutralvsLaNina
S4 -> NeutralvsLaNina
SOI -> NeutralvsLaNina
NATL -> NeutralvsLaNina
TROP -> NeutralvsLaNina


S3 -> excess
S34 -> excess
S4 -> excess
SOI -> excess
NATL -> excess
TROP -> excess


S3 -> Rain
S34 -> Rain
S4 -> Rain
SOI -> Rain
NATL -> Rain
TROP -> Rain


S3 -> Temp
S34 -> Temp
S4 -> Temp
SOI -> Temp
NATL -> Temp
TROP -> Temp


NeutralvsLaNina -> Temp
NeutralvsLaNina -> Rain
NeutralvsLaNina -> excess

Temp -> UBN
Temp -> excess

Rain -> UBN
Rain -> excess

Rain -> Temp

UBN -> excess

UBN -> Pop_density

Pop_density -> excess


}')  


plot(dag)


## check whether any correlations are perfect (i.e., collinearity)
myCov <- cov(reduced_Data, use = "pairwise.complete.obs")
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

## if not, check for multicollinearity (i.e., is one variable a linear combination of 2+ variables?)
det(myCov) < 0
## or
any(eigen(myCov)$values < 0)


## Conditional independences
impliedConditionalIndependencies(dag)
corr <- lavCor(reduced_Data) #, missing = "listwise")

# Plot
localTests(dag, sample.cov=corr, sample.nobs=nrow(reduced_Data))
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(reduced_Data)), xlim=c(-1,1))





#identification
simple_dag <- dagify(
  excess ~  NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP + Rain + Temp + UBN + health,
  NeutralvsLaNina ~ S3 + S34 + S4 + SOI + NATL + TROP,
  S3 ~ S34 + S4 + SOI + NATL + TROP,
  S34 ~ S4 + SOI + NATL + TROP,
  S4 ~ SOI + NATL + TROP,
  SOI ~ NATL + TROP,
  NATL ~ TROP,
  Temp ~ NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP + Rain,
  Rain ~ NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP,
  UBN ~ Rain + Temp,
  health ~ UBN,
  exposure = "NeutralvsLaNina",
  outcome = "excess",
  coords = list(x = c(excess=1, NeutralvsLaNina=-1, SOI=-1.7, S3=-1.8, S4=-1.9, S34=-2, NATL=-2.1, TROP=-2.2,
                      Rain=-1.0, Temp=0.0,
                      UBN=0.5, health=0.5),
                y = c(excess=0.5, NeutralvsLaNina=0.5, SOI=1.1, S3=1.2, S4=1.3, S34=1.4, NATL=1.5, TROP=1.6,
                      Rain=-2.5, Temp=-2.0,
                      UBN=-1.25, health=1.25))
)





# theme_dag() 
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()

# adjusting
adjustmentSets(simple_dag,  type = "minimal")
ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()


