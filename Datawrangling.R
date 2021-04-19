setwd('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj')
followup <- read.csv('followup_processed3.csv')
final <- read.csv('final_processed3.csv')

final$keep <- final$ID %in% followup$ID

split <- split(final, f = final$keep)
final2 <- split[[2]]
#sanity check to make them the same
identical(final2$ID, followup$ID)


names(final2)
names(followup)
setdiff(names(final2), names(followup))

names(followup) <- sub("_fu*", "", names(followup))
final3 <- subset(final2, select = names(final2) %in% names(followup))
followup2 <- subset(followup, select = names(followup) %in% names(final3))

outlier <- read.csv('outlier_df.csv')
final3$isoutlier <- final3$ID %in% outlier$Patient_ID
followup2$isoutlier <- followup2$ID %in% outlier$Patient_ID

# 15 outliers in both. 
table(final3$isoutlier)["TRUE"]
table(followup2$isoutlier)['TRUE']

write.csv(final3, 'C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/finalprocessed4.csv')
write.csv(followup2, 'C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/followup_processed4.csv')

## pca
finalprocessed <- read.csv('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/finalprocessed4.csv', row.names = 2)
followupprocessed <- read.csv('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/followup_processed4.csv', row.names = 2)
library(ggplot2)
library(MASS)
library(faraway)
final4 <- finalprocessed[,-c(1,24)]
#final5 <- log(final4 + 0.1) # log introduces inf and NaN so need to do a different one
#final5 <- 

pcax <- prcomp(final4, center = TRUE, scale. = TRUE)
summary(pcax)
plot(pcax$x[,1], pcax$x[,2])

followup2 <- followupprocessed[,-c(1,24)]
pcay <- prcomp(final4, center = TRUE, scale. = TRUE)
summary(pcay)
plot(pcax$x[,1], pcax$x[,2])


