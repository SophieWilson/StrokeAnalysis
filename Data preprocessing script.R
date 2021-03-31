library(missForest)
setwd('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj')
finalset <- read.csv('11AUG10_AUDITED_FinalSET1data_paths.csv', header = TRUE, skip = 1)


### Going to start by working on the final set data ####
# first split into meta and value dataframes
metadata <- finalset[,1:36]
data_values <- finalset[,37:ncol(finalset)]
rownames(data_values) <- metadata$ID
# removing non-numeric columns, not really sure why they are in here. ask pia
data_values <- subset(data_values, select = -c(S17Mtr) )
# convert all character values to NA, this includes n/a, NA, BLANK, blank etc. 
data_values2 <- data.frame(apply(data_values, 2, function(x) as.numeric(as.character(x))))

# then remove participants with 75% missing data, so remove rows with over 25% missing values (there are none in this one)
data_values2 <- data_values2[which(rowMeans(!is.na(data_values2)) >= 0.25), ]

##### this bit is just checking if there are any non-numeric bits ####
# data_values2[] <- lapply(data_values, function(x) {
#   if(is.factor(x)) as.numeric(as.character(x)) else x
# })
# sapply(data_values2, class)
# 
# is.numeric(data_values2)
# which(!grepl('^[0-9]',data_values2))

## testing missforest imputation, this is on a relatively small sample as there are lots of na values so is likely to be worse than the actual predicitons
no_natest <- data_values2[rowSums(is.na(data_values2)) == 0,]

# adding 15% of missing values
added_na <- as.data.frame(lapply(no_natest, function(cc) cc[ sample(c(TRUE, NA), prob = c(0.85, 0.15), size = length(cc), replace = TRUE) ]))
## imputing the forset, and testing against true values
full_impute <- missForest(added_na, xtrue = no_natest)
full_impute$OOBerror # has an oob of 0.27 
## calculating actual error
sum_true <- colSums(no_natest)
sum_predict <- colSums(full_impute$ximp)
difference <- 0
error <- 0
for (x in 1:length(sum_true)) {
  difference[x] <- (sum_true[x] - sum_predict[x])
  error[x] <- difference[x] / sum_true[x]
  
}
sum(error) # the error is quite high but its mainly caused by column 54 in the dataframe (which has an error of 0.43), maybe we remove it?
mean(difference)
print(error[54])

# Filling in the missing values with appropriate
processed <- missForest(data_values2)
processeddf <- processed$ximp
processed$OOBerror
## the 00b is 0.018 so its pretty good

# Rounding the numbers to 2 dp. 
processeddf[] <- lapply(processeddf[], round, 2)
# removing the metadata for participants with missing data (so the files match to merge them)
metadata2<- metadata[ metadata$ID %in% processeddf$ID, ]
# merging the datasets
final_processed <- cbind(metadata2, processeddf[,2:ncol(processeddf)])
# writing the file out
write.csv(final_processed, 'C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/final_processed.csv' )

                                ############    Now doing the same for the follow up data ###########

followupdata <- read.csv('BUCS_FollowUp Data_November2010_CT.csv', header = TRUE, skip = 2)

### Going to start by working on the final set data ####
# first split into meta and value dataframes
metadatafollowup <- followupdata[,1:31]
followup_values <- followupdata[,c(4, 32:ncol(followupdata))]
f

# convert all character values to NA, this includes n/a, NA, BLANK, blank etc. 
followup_values2 <- data.frame(apply(followup_values, 2, function(x) as.numeric(as.character(x))))

# then remove participants with 75% missing data, so remove rows with over 25% missing values (there are none in this one)
followup_values2 <- followup_values2[which(rowMeans(!is.na(followup_values2)) >= 0.25), ]

##### this bit is just checking if there are any non-numeric bits ####
# data_values2[] <- lapply(data_values, function(x) {
#   if(is.factor(x)) as.numeric(as.character(x)) else x
# })
# sapply(data_values2, class)
# 
# is.numeric(data_values2)
# which(!grepl('^[0-9]',data_values2))



# Filling in the missing values with appropriate
fuprocessed <- missForest(followup_values2)
fuprocesseddf <- fuprocessed$ximp
fuprocessed$OOBerror
## the 00b is 0.018 so its pretty good

# Rounding the numbers to 2 dp. 
fuprocesseddf[] <- lapply(fuprocesseddf[], round, 2)
# removing the metadata for participants with missing data (so the files match to merge them)
metadatafollowup2<- metadatafollowup[ metadatafollowup$ID %in% fuprocesseddf$ID, ]
# merging the datasets
followup_processed <- cbind(metadatafollowup2, fuprocesseddf[,2:ncol(fuprocesseddf)])

write.csv(followup_processed, 'C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/followup_processed.csv' )


############## Playing around with outliers ###############
# first import and format the data
header <- c('Baseline_ID', 'Baseline_val', 'FU_ID', 'FU_val')
barthel_outliers <- read.csv('barthel outliers.csv', skip=1)
colnames(barthel_outliers) <- header
# then split into follow up and baseline
fu <- data.frame(na.omit(barthel_outliers[,c(3,4)]))
furownames <- fu$FU_ID
bl <- data.frame(barthel_outliers[,c(1,2)])
# remove all baseline ids that are not in follow up
bl2 <- bl[ bl$Baseline_ID %in% furownames, ]
# remerge the data, and remove decimals to make it easier to read
barthel2 <- lapply(cbind(bl2, fu), round, 0)


## Equalling out the datasets
outlier <- read.csv('final_processedvalues_outlier.csv')
# splitting the data based on outlier class
split <- split(outlier , f = outlier$isoutlier )
# sampling of the even rows randomly to be the same length of outlier numbers
even_vals <- split[[1]][sample(nrow(split[[1]]), nrow(split[[2]])), ]

equal_data <- rbind(even_vals, split[[2]])
equal_data <- equal_data[,-1]
write.csv(equal_data, 'C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/equalled_data_vals.csv')


### Further preprocessing for the second file
library(missForest)
setwd('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj')
finalset <- read.csv('followup_processed 2.csv')
finalset <- finalset[,-1]

# convert all character values to NA, this includes n/a, NA, BLANK, blank etc. 
data_values <- data.frame(apply(finalset, 2, function(x) as.numeric(as.character(x))))

# then remove participants with 75% missing data, so remove rows with over 25% missing values (there are none in this one)
data_values <- data_values[which(rowMeans(!is.na(data_values)) >= 0.25), ]
data_values2 <- data_values[ , which(colMeans(!is.na(data_values)) >= 0.25)] # same for columns

# Filling in the missing values with appropriate
processed <- missForest(data_values2)
processeddf <- processed$ximp
processed$OOBerror
## the 00b is 0.018 so its pretty good
## Add isoutlier column
outliers <- read.csv('outlier_df.csv')
processeddf$isoutlier <- processeddf$ID %in% outliers$Patient_ID
# Rounding the numbers to 2 dp. 
processeddf[] <- lapply(processeddf[], round, 2)
write.csv(processeddf, 'C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/followup_processed3.csv')

## follow up and final had different columns kept, so i removed the differing columns by hand so that they would be the same. 


# make another dataframe with improvement data
outliers2 <- outliers
outliers2$improvement <- ifelse(outliers$X > 0, FALSE, TRUE)

dat[4:6] <- ifelse(dat[, 4:6] > 0, FALSE, TRUE)