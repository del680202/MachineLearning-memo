library("data.tree")

#https://cran.r-project.org/web/packages/data.tree/vignettes/data.tree.html
#http://codecrafthouse.jp/machine-learning/decision-tree
#http://suzuichibolgpg.blog.fc2.com/blog-entry-87.html


gini <- function(target){
  classes <- unique(target)
  numberOfData <- length(target)
  gini_value = 1
  for(cls in classes){
    gini_value <- gini_value - (length(target[target==cls]) / numberOfData) ^ 2
  }
  return(gini_value)
}

DecisionTree.fit <- function(data, target, featureIndex){
  
  if(length(unique(target)) == 1){
    return(Node$new(paste("type_", target[1]), label=target[1]))
  }
  
  numberOfData <- ncol(data)
  bestFeature <- 1
  bestGiniIndex <- 0
  bestThreshold <- 0
  parentGiniIndex <- gini(target)
  for(f in featureIndex){
    currentFeature <- f
    feature_data <- sort(unique(data[currentFeature, ]))
    length_feature_data = length(feature_data)
    threshold_points <- (feature_data[1:length_feature_data - 1] + feature_data[2:length_feature_data]) / 2
    for(threshold in threshold_points){
      l_index <- data[currentFeature, ] > threshold
      r_index <- data[currentFeature, ] <= threshold
      l_gini <- gini(target[l_index])
      r_gini <- gini(target[r_index])
      l_probability <- length(target[l_index]) / numberOfData
      r_probability <- length(target[r_index]) / numberOfData
      giniIndex <- parentGiniIndex - (l_probability * l_gini + r_gini * r_probability)
      if(bestGiniIndex < giniIndex){
        bestGiniIndex <- giniIndex
        bestFeature <- currentFeature
        bestThreshold <- threshold
      }
    }
  }
  counts <- table(target)
  currentLabel <- as.integer(names(counts)[which.max(counts)]) #Find max frequecy label
  node <- Node$new(paste("feature_", bestFeature, " > ", bestThreshold), 
                   threshold=bestThreshold, 
                   feature=bestFeature, 
                   giniIndex=bestGiniIndex,
                   label=currentLabel)
  l_index <- data[node$feature,] > node$threshold
  r_index <- data[node$feature,] <= node$threshold
  node$AddChildNode(DecisionTree.fit(data[, l_index], target[l_index], featureIndex))
  node$AddChildNode(DecisionTree.fit(data[, r_index], target[r_index], featureIndex))
  node
}


DecisionTree.predict <- function(model, features){
  #features <- c((featureName, value), (featureName, value)...)
  if(is.null(model$feature)){
    #Leaf node
    return(model$label)
  }else{
    for(feature in features){
      featureName <- feature[1]
      featureValue <- feature[2]
      if(featureName == model$feature){
        if(featureValue > model$threshold){
          return(DecisionTree.predict(model$children[[1]], features))
        }else{
          return(DecisionTree.predict(model$children[[2]], features))
        }
      }
    }
    #No feature found
    return(model$label)
  }
}


n          <- 10
Y          <- c(0,0,0,1,1,1,2,2,2,2)
red <- c(2.2, 1, 0, 1, 1, 2)
blue <- c(2, -0.5, 3, -2, 1, -4)
green <- c(-4, 2.2, -2.9, 0, -3, -3, -2, 3)
data <- matrix(c(red, blue, green),2,n)

#plot data point
plot(c(2.2,0,1),c(1,1,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3, 1),c(-0.5,-2, -4), pch = "o", col="blue")
points(c(-4,-2.9,-3,-2),c(2.2,0,-3,3), pch = "*", col="green")
model <- DecisionTree.fit(data, Y, 1:2)
DecisionTree.predict(model, list(c(1,0),c(2,1)))  #(0, 1)
DecisionTree.predict(model, list(c(1, 2),c(2, -1))) #(2, -1)
DecisionTree.predict(model, list(c(1,-1),c(2,1))) #(-1, 1)
DecisionTree.predict(model, list(c(3,-1))) #Non-exist feature


#Draw line
o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- DecisionTree.predict(model, list(c(1,i),c(2,j)))
    if(r == 0){
      x_x <- c(x_x, i)
      x_y <- c(x_y, j)
    }else if(r == 1){
      o_x <- c(o_x, i)
      o_y <- c(o_y, j)
    }else{
      n_x <- c(n_x, i)
      n_y <- c(n_y, j)
    }
  }
}
plot(x_x,x_y, xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(o_x,o_y, xlim = c(-5,5), ylim = c(-5,5), pch = "o", col="blue")
points(n_x,n_y, xlim = c(-5,5), ylim = c(-5,5), pch = "*", col="green")


n          <- 8
Y          <- c(-1,-1,-1, -1,1,1,1, 1)
data <- matrix(c(-4.6, -3, -2, 1, 0, 0, 1, 2, 2, 0, 3, -2, -4, 2, 1, -4),2,n)

#plot data point
plot(c(-2,0,1, -4.6),c(1,0,2, -3), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3, -4, 1),c(0,-2, 2, -4), pch = "o", col="blue")
model <- DecisionTree.fit(data, Y, 1:2)
DecisionTree.predict(model, list(c(1,0),c(2,0.1)))  #(0, 0.1)


#Draw line
o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- DecisionTree.predict(model, list(c(1,i),c(2,j)))
    if(r == -1){
      x_x <- c(x_x, i)
      x_y <- c(x_y, j)
    }else if(r == 1){
      o_x <- c(o_x, i)
      o_y <- c(o_y, j)
    }else{
      n_x <- c(n_x, i)
      n_y <- c(n_y, j)
    }
  }
}
plot(x_x,x_y, xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(o_x,o_y, xlim = c(-5,5), ylim = c(-5,5), pch = "o", col="blue")
points(n_x,n_y, xlim = c(-5,5), ylim = c(-5,5), pch = "*", col="green")


RandomForest.fit <- function(data, target, treeCounts, featureCount){
  trees <- c()
  numberOfData <- ncol(data)
  numerOfFeatures <- nrow(data)
  for(i in 1:treeCounts){
    #bootstrap sample
    ind <- unique(sample(1:numberOfData, replace = T))
    sampleData <- data[,ind]
    sampleTarget <- target[ind]
    #Sample featues
    sampleFeatureIndex <- sort(sample(1:numerOfFeatures, featureCount))
    t <- DecisionTree.fit(sampleData, sampleTarget, sampleFeatureIndex)
    trees <- c(trees, t)
  }
  trees
}


RandomForest.predict <- function(model, features){
  #voting
  results <- c()
  for(t in model){
    results <- c(results, DecisionTree.predict(t, features))
  }
  
  counts <- table(results)
  maxFreqLabel <- names(counts)[which.max(counts)]
  return(maxFreqLabel)
}

model <- RandomForest.fit(data, Y, 100, 2)

#Draw line
o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- RandomForest.predict(model, list(c(1,i),c(2,j)))
    if(r == "-1"){
      x_x <- c(x_x, i)
      x_y <- c(x_y, j)
    }else if(r == "1"){
      o_x <- c(o_x, i)
      o_y <- c(o_y, j)
    }else{
      n_x <- c(n_x, i)
      n_y <- c(n_y, j)
    }
  }
}
plot(x_x,x_y, xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(o_x,o_y, xlim = c(-5,5), ylim = c(-5,5), pch = "o", col="blue")
points(n_x,n_y, xlim = c(-5,5), ylim = c(-5,5), pch = "*", col="green")


# Regression Tree Example
library(rpart)

# grow tree 
fit <- rpart(Mileage~Price + Country + Reliability + Type, 
             method="anova", data=cu.summary)

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# create additional plots 
par(mfrow=c(1,2)) # two plots on one page 
rsq.rpart(fit) # visualize cross-validation results    

# plot tree 
plot(fit, uniform=TRUE, 
     main="Regression Tree for Mileage ")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

