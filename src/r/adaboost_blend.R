#Blending

n          <- 7
Y          <- c(-1,-1,-1,1,1,1, 1)
data <- matrix(c(-2, 1, 0, 0, 1, 2, 2, 0, 3, -2, -4, 2, 1, -4),2,n)

#plot data point
plot(c(-2,0,1),c(1,0,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3, -4, 1),c(0,-2, 2, -4), pch = "o", col="blue")

pocket <- function(data, Y){
  train <- rbind(data, rep(1, ncol(data)))
  w <- matrix(c(0, 0, 0))
  for(x in 1:ncol(data)){
    err <- NULL
    err_y <- c()
    for(i in 1:ncol(data)){
      if(sign(t(w) %*% train[,i]) != Y[i]){
        if(is.null(err)){
          err <- matrix(train[,i])
        }else{
          err <- cbind(err, train[,i])
        }
        err_y <- c(err_y, Y[i])
      }
    }
    if(is.null(err)){
      return(w)
    }
    #random error
    ind <- sample(1:ncol(err), 1)
    err_data <- err[,ind]
    err_y <- err_y[ind]
    e1 <- 0
    for(j in 1:ncol(data)){
      if(sign(t(w) %*% train[,j]) != Y[j]){
        e1 <- e1 + 1
      }
    }
    tmp_w <- w + err_y * err_data
    e2 <- 0
    for(j in 1:ncol(data)){
      if(sign(t(tmp_w) %*% train[,j]) != Y[j]){
        e2 <- e2 + 1
      }
    }
    if(e2 < e1){
      w <- tmp_w
    }
  }
  return(w)
}

w <- pocket(data)
f <- function(x) { return(-(w[3] + w[1]*x)/w[2])  }
plot(-5:5, f(-5:5), xlim = c(-5,5), ylim = c(-5,5), type="l")
points(c(-2,0,1),c(1,0,2), pch = "x", col="red")
points(c(2,3, -4, 1),c(0,-2, 2, -4), pch = "o", col="blue")


#curry function 
#http://stackoverflow.com/questions/2228544/higher-level-functions-in-r-is-there-an-official-compose-operator-or-curry-fun
library(functional)
make_g <- function(w){
  local_w <- w
  function(data){
    X <- rbind(data, 1)
    return(sign(t(local_w) %*% X))
  }
}

T <- 25
gs <- c()
for(i in 1:T){
  w <- pocket(data, Y)
  g <- make_g(w)
  gs <- c(gs, g)
}

G <- function(data){
  sum <- 0
  for(i in 1:length(gs)){
    sum <- sum + gs[[i]](data)
  }
  return(sign(sum))
}

o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- G(matrix(c(i, j)))
    if(r > 0){
      o_x <- c(o_x, i)
      o_y <- c(o_y, j)
    }else if(r < 0){
      x_x <- c(x_x, i)
      x_y <- c(x_y, j)
    }else{
      n_x <- c(n_x, i)
      n_y <- c(n_y, j)
    }
  }
}

plot(x_x,x_y, xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(o_x,o_y, xlim = c(-5,5), ylim = c(-5,5), pch = "o", col="blue")


#Bagging + Bootstrap
#http://www.ats.ucla.edu/stat/r/library/bootstrap.htm

#A sample that distructbution is good
n          <- 12
Y          <- c(-1,-1,-1,1,1,1, 1, 1, 1, 1, 1, 1)
data <- matrix(c(-2, 1, 0, 0.5, 1, 2, 2, 4,2, 4, 3, -2, -2, -1, 1, -4, 2.1, 4.2, 1.9, 4, -2.2, -0.9, -2.1, -1.1),2,n)

plot(c(-2,0,1),c(1,0.5,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2, 2,3, -2, -2.2, -2.1, 1, 2.1, 1.9),c(4, 4,-2, -1, -0.9, -1.1, -4, 4.2, 4), pch = "o", col="blue")


#A sample that distructbution is bad (had outlter)
n          <- 7
Y          <- c(-1,-1,-1,1,1,1, 1)
data <- matrix(c(-2, 1, 0, 0.5, 1, 2, 2, 4, 3, -2, -2, -1, 2, -4),2,n)

plot(c(-2,0,1),c(1,0.5,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3, -2, 2),c(4,-2, -1, -4), pch = "o", col="blue")



pocket <- function(data, Y){
  train <- rbind(data, rep(1, ncol(data)))
  w <- matrix(c(0, 0, 0))
  for(x in 1:5){
    err <- NULL
    err_y <- c()
    for(i in 1:ncol(data)){
      if(sign(t(w) %*% train[,i]) != Y[i]){
        if(is.null(err)){
          err <- matrix(train[,i])
        }else{
          err <- cbind(err, train[,i])
        }
        err_y <- c(err_y, Y[i])
      }
    }
    if(is.null(err)){
      return(w)
    }
    #random error
    ind <- sample(1:ncol(err), 1)
    err_data <- err[,ind]
    err_y <- err_y[ind]
    e1 <- 0
    for(j in 1:ncol(data)){
      if(sign(t(w) %*% train[,j]) != Y[j]){
        e1 <- e1 + 1
      }
    }
    alpha <- sample(seq(1e-10,1,length.out = 100), 1)
    tmp_w <- w + alpha * err_y * err_data
    e2 <- 0
    for(j in 1:ncol(data)){
      if(sign(t(tmp_w) %*% train[,j]) != Y[j]){
        e2 <- e2 + 1
      }
    }
    print(e1)
    if(e2 < e1){
      w <- tmp_w
    }
  }
  return(w)
}

make_g <- function(w){
  local_w <- w
  function(data){
    X <- rbind(data, 1)
    return(sign(t(local_w) %*% X))
  }
}

T = 25
gs <- c()
for(i in 1:T){
  ind <- sample(1:ncol(data), ncol(data), replace = T)
  w <- pocket(data[, ind], Y[ind])
  g <- make_g(w)
  gs <- c(gs, g)
  f <- function(x) { return(-(w[3] + w[1]*x)/w[2])  }
  lines(-5:5, f(-5:5), xlim = c(-5,5), ylim = c(-5,5), type="l", col="gray")
}

G <- function(data){
  sum <- 0
  for(i in 1:length(gs)){
    sum <- sum + gs[[i]](data)
  }
  return(sign(sum))
}

o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- G(matrix(c(i, j)))
    if(r > 0){
      o_x <- c(o_x, i)
      o_y <- c(o_y, j)
    }else if(r < 0){
      x_x <- c(x_x, i)
      x_y <- c(x_y, j)
    }else{
      n_x <- c(n_x, i)
      n_y <- c(n_y, j)
    }
  }
}

plot(x_x,x_y, xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(o_x,o_y, xlim = c(-5,5), ylim = c(-5,5), pch = "o", col="blue")


#AdaBoost + stump
decisionStumpR <- function(X, Y, W){
  X.dim <- dim(X)[1]
  act <- c("lt", "gt")
  best.dim <- 0
  best.threshold <- 0
  best.act <- NULL
  cost <- Inf
  n <- ncol(data)
  for(d in 1:X.dim){
    
    temp.data <- unique(sort(data[d,]))
    t.list <- c()
    for(i in 2:length(temp.data)){
      t.list <- c(t.list, (temp.data[i] + temp.data[i - 1]) / 2)
    }
    
    for(a in act){
      for(i in t.list){
        if(a == "lt"){
          yhat <- (data[d,] > i) * 2 - 1
        }else{
          yhat <- (data[d,] < i) * 2 - 1
        }
        temp.cost <- sum((yhat != Y) %*% W)
        if(temp.cost < cost){
          best.dim <- d
          best.threshold <- i
          best.act <- a
          cost <- temp.cost 
        }
      }
    }
  }
  return(list(d=best.dim, t=best.threshold, a=best.act, c=cost))
}
#res <- decisionStumpR(data, Y, rep(1/n, n))
dt.classify <- function(clt, X){
  yhat <- 0
  if(clt$a == "lt"){
    yhat <- (X[clt$d,] > clt$t) * 2 - 1
  }else{
    yhat <- (X[clt$d,] < clt$t) * 2 - 1
  }
  yhat
}

n          <- 14
Y          <- c(-1,-1,-1, -1,-1,-1,-1,1,1,1, 1,1,1,1)
data <- matrix(c(-2, 1,-2, 1,-0.5, -4.3, 1.1, -4.5, 1,1,0,0.5,2,2,2,4,-2.5,-2,1.5,-1.2,2,-2.5,0.2,4.2, -3, -2.5, -2, -1.5),2,n)

plot(c(-2,-0.5, 1.1, 1,0,2),c(1,-4.3, -4.5, 1,0.5,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,-2.5, -3, -2, 1.5, 2, 0.2),c(4,-2, -2.5, -1.5, -1.2, -2.5, 4.2), pch = "o", col="blue")


n          <- 14
Y          <- c(-1,-1,-1, -1,-1,-1,-1,1,1,1, 1,1,1,1)
data <- matrix(c(-2, 1,-2, 1,-0.5, -4.3, 1.1, -4.5, 1,1,0,0.5,2,2,2,4,-2.5,-2,1.5,-1.2,2,-2.5,0.2,4.2, -3, 1.5, -2, -1.5),2,n)

plot(c(-2,-0.5, 1.1, 1,0,2),c(1,-4.3, -4.5, 1,0.5,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,-2.5, -3, -2, 1.5, 2, 0.2),c(4,-2, 1.5, -1.5, -1.2, -2.5, 4.2), pch = "o", col="blue")


n          <- 7
Y          <- c(-1,-1,-1,1,1,1, 1)
data <- matrix(c(-2, 1, 0, 0.5, 1, 2, 2, 4, 3, -2, -2, -1, 2, -4),2,n)

plot(c(-2,0,1),c(1,0.5,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3, -2, 2),c(4,-2, -1, -4), pch = "o", col="blue")


n          <- 7
Y          <- c(-1, -1, -1,1,1,1,1)
data <- matrix(c(0, 0.1, 0.2, 0.2, -0.2, -0.2, 2, 0, 0, 2, -2, 0, 0, -2),2,n)

plot(c(0, 0.2, -0.2),c(0.1, 0.2, -0.2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(0, 0, 2, -2),c(2, -2, 0, 0), pch = "o", col="blue")



make_g <- function(res){
  local_res <- res
  function(data){
    yhat <- dt.classify(res, data)
    return(yhat)
  }
}

#AdaBoost
T = 20  #Setup T
gs <- c()
ut <- rep(1/n, n)  # init weight = 1/N for all poins
alpha <- c()
for(i in 1:T){
  clt <- decisionStumpR(X=data, Y=Y, W=ut) #base algorithm, return c(w, et, err_points)
  err_points <- c()
  yhat <- dt.classify(clt, data)
  for(j in 1:n){
    if(yhat[j] != Y[j]){
      err_points <- c(err_points, j)
    }
  }

  et <- sum(ut[err_points]) / sum(ut)
  factor <- sqrt((1-et) / et)

  ut[err_points] <- ut[err_points] * factor
  #ut[-err_points] <- ut[-err_points] / factor
  #ut[err_points] <- ut[err_points] *  exp(log(factor))
  #print(ut)
  alpha <- c(alpha, log(factor))
  g <- make_g(clt)
  gs <- c(gs, g)
}


G <- function(data){
  sum <- 0
  for(i in 1:length(gs)){
    sum <- sum + gs[[i]](data) * alpha[i]
  }
  return(sign(sum))
}

o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- G(matrix(c(i, j)))
    if(r > 0){
      o_x <- c(o_x, i)
      o_y <- c(o_y, j)
    }else if(r < 0){
      x_x <- c(x_x, i)
      x_y <- c(x_y, j)
    }else{
      n_x <- c(n_x, i)
      n_y <- c(n_y, j)
    }
  }
}

plot(x_x,x_y, xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(o_x,o_y, xlim = c(-5,5), ylim = c(-5,5), pch = "o", col="blue")

#CART Tree
library("rpart")
n          <- 15
Y          <- c(-1,-1,-1, -1,-1,-1,-1,-1,1,1,1, 1,1,1,1)
data <- matrix(c(-2, 1,-2, 1,-0.5, -4.3, 1.1, -4.5, 1,1,0,0.5,2,2, 2.5, -2,2,4,-2.5,1,1.5,-1.2,2,-2.5,0.2,4.2, -3, 2.8, -2, -1.5),2,n)

plot(c(-2,-0.5, 1.1, 1,0,2, 2.5),c(1,-4.3, -4.5, 1,0.5,2, -2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,-2.5, -3, -2, 1.5, 2, 0.2),c(4,1, 2.8, -1.5, -1.2, -2.5, 4.2), pch = "o", col="blue")

m_Y <- (Y == 1) + 1 # (-1, 1) -> (1,2)
m_data <- data.frame(X=data[1,], Y=data[2,], L=m_Y)
model <- rpart(L~., data = m_data, control=rpart.control(minsplit=2))
plot(model)
text(model)

x1 <- seq(min(m_data$X), max(m_data$X), length = 50)
x2 <- seq(min(m_data$Y), max(m_data$Y), length = 50)
Feature_x1_to_x2 <- expand.grid(X = x1, Y = x2)
Feature_x1_to_x2_Class <- predict(model,Feature_x1_to_x2)
plot(m_data[1:2], pch = 21, bg = c("red", "green3")[m_Y])
contour(x1,x2,matrix(Feature_x1_to_x2_Class,length(x1)),add = T, levels = 1.5,labex = 0)

