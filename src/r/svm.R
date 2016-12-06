#primal SVM
library(quadprog)
d          <- 3 # 2d + 1
n          <- 4
Dmat       <- matrix(0, d, d)
diag(Dmat) <- 1
dvec       <- rep(0, d)
Y          <- c(-1,-1,1,1)
Amat       <- Y*t(matrix(c(1, 0, 0, 1, 2, 2, 1, 2, 0, 1, 3, 0),d,n))
bvec       <- rep(1, n)
res <- solve.QP(Dmat,dvec,t(Amat),bvec=bvec)
sol <- res$solution
f <- function(x) { return(-(sol[1] + sol[2]*x)/sol[3])  }

plot(-5:5, f(-5:5), type="l")
points(c(0,2),c(0,2), pch = "x", col="red")
points(c(2,3),c(0,0), pch = "o", col="blue")
print(sol)

#############
# dual SVM
n          <- 5
Y          <- c(-1,-1,-1,1,1)
data <- matrix(c(-2, 1, 0, 0, 2, 2, 2, 0, 3, 0),2,n)

#Dmat <- Y %o% Y * t(data) %*% data
m <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    m[i,j] <- Y[i]*Y[j]*t(data[, i]) %*% data[, j]
  }
}
Dmat <- m
dvec       <- rep(-1, n)
bvec       <- rep(0, n)

#https://stat.ethz.ch/pipermail/r-sig-finance/attachments/20080901/637ba8c6/attachment.pl
#Amat       <- matrix(0, n, n)
#diag(Amat)       <- Y
#A=matrix(1:1,1, n)
#for(i in 1:3){
#  A[,i] <- -1
#}
A <- matrix(Y, 1)

#http://d.hatena.ne.jp/repose/20080917/1221580572
library(kernlab)
res <- ipop(c = dvec, H = Dmat, A=A, b=0, l=rep(0, n),u=rep(10000,length=n), r=0, sigf=7)
sol <- res@primal
SV <- sol > 0.1 #simulate = 0
w <- apply(sol[SV] * Y[SV] * data[, SV], MARGIN = 1, sum)
b <- Y[SV] - t(w) %*% data[,SV]
b <- b[1,1]
f <- function(x) { return(-(b + w[1]*x)/w[2])  }
plot(-5:5, f(-5:5), type="l")
points(c(-2,0,2),c(1,0,2), pch = "x", col="red")
points(c(2,3),c(0,0), pch = "o", col="blue")


#############
# kernel SVM
n          <- 6
Y          <- c(-1,-1,-1,1,1,1)
#data <- matrix(c(-2, 1, 0, 0, 1, 2, 2, 0, 3, -2, -4, 2),2,n)
data <- matrix(c(-4, -1, 0, 3, 4, -1, -3, 3, 1, -1, 3, 3),2,n)
#data <- matrix(c(-4, 4, 4, 4, 4, -4, 1, -1, 0, 0, -1, 1),2,n)

#plot data point
#plot(c(-2,0,1),c(1,0,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
#points(c(2,3, -4),c(0,-2, 2), pch = "o", col="blue")
#plot(c(-4,0,4),c(-1,3,-1), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
#points(c(-3,1, 3),c(3,-1, 3), pch = "o", col="blue")
#plot(c(-4,4,4),c(4,4,-4), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
#points(c(1,0, -1),c(-1,0, 1), pch = "o", col="blue")

polyKernel <- function(Q, r, e){
  kernel <- function(x1, x2){
    return ((e + r * t(x1) %*% x2)^Q)
  }
  return(kernel)
}

RBFKernel <- function(r){
  kernel <- function(x1, x2){
    v <- x1 - x2
    return (exp(-r * ( t(v) %*% v )^2))
  }
  return(kernel)
}
#K <- polyKernel(2,1,1)
K <- RBFKernel(1)

m <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    m[i,j] <- Y[i]*Y[j]*K((data[, i]), data[, j])
  }
}
Dmat <- m
dvec       <- rep(-1, n)
bvec       <- rep(0, n)
A <- matrix(Y, 1)

library(kernlab)
res <- ipop(c = dvec, H = Dmat, A=A, b=0, l=rep(0, n),u=rep(100000,length=n), r=0, sigf=7)
sol <- res@primal
SV <- sol > 0.01 #simulate = 0

ys <- Y[SV][1]
xs <- data[,SV][,1]
tmp <- 0
for(i in 1:length(sol[SV])){
  tmp <- tmp + sol[SV][i] * Y[SV][i] * K(data[,SV][,i], xs)
}
b <- ys - tmp
b = b[1,1]

svm <- function(x){
  r <- 0
  for(i in 1:length(sol[SV])){
    r <- r + sol[SV][i] * Y[SV][i] * K(data[,SV][,i], x)
  }
  return(sign(r))
}

o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- svm(matrix(c(i, j)))
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


#############
# soft-margin SVM
#soft-margin with kernel
n          <- 7
Y          <- c(-1,-1,-1,1,1,1, 1)
data <- matrix(c(-2, 1, 0, 0, 1, 2, 2, 0, 3, -2, -4, 2, 1, -4),2,n)
#plot data point
plot(c(-2,0,1),c(1,0,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3, -4, 1),c(0,-2, 2, -4), pch = "o", col="blue")

RBFKernel <- function(r){
  kernel <- function(x1, x2){
    v <- x1 - x2
    return (exp(-r * ( t(v) %*% v )^2))
  }
  return(kernel)
}
K <- RBFKernel(1)

m <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    m[i,j] <- Y[i]*Y[j]*K((data[, i]), data[, j])
  }
}
Dmat <- m

dvec       <- rep(-1, n)
bvec       <- rep(0, n)
Amat       <- matrix(Y, 1)

#C <- rep(2, n)
C <- rep(0.8, n)

library(LowRankQP)
res <- LowRankQP(Vmat=Dmat, dvec=dvec, Amat=Amat, bvec=0, uvec=C)
sol <- res$alpha
SV <- sol > 0.01 #simulate = 0


ys <- Y[SV][1]
xs <- data[,SV][,1]

tmp <- 0
for(i in 1:length(sol[SV])){
  tmp <- tmp + sol[SV][i] * Y[SV][i] * K(data[,SV][,i], xs)
}
b <- ys - tmp
b = b[1,1]


svm <- function(x){
  r <- 0
  for(i in 1:length(sol[SV])){
    r <- r + sol[SV][i] * Y[SV][i] * K(data[,SV][,i], x)
  }
  return(sign(r))
}



o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- svm(matrix(c(i, j)))
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



###############################
#Probabilistic SVM (Platt's Model)
n          <- 5
Y          <- c(-1,-1,-1,1,1)
data <- matrix(c(-2, 1, 0, 0, 1, 2, 2, 0, 3, -2),2,n)

#plot data point
plot(c(-2,0,1),c(1,0,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3),c(0,-2), pch = "o", col="blue")


#Kernel Function
RBFKernel <- function(r){
  kernel <- function(x1, x2){
    v <- x1 - x2
    return (exp(-r * ( t(v) %*% v )^2))
  }
  return(kernel)
}
K <- RBFKernel(0.1)


#Level-1 learning - SVM
m <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    m[i,j] <- Y[i]*Y[j]*K((data[, i]), data[, j])
  }
}
Dmat <- m
dvec       <- rep(-1, n)
bvec       <- rep(0, n)
A <- matrix(Y, 1)

C <- rep(1, n)
library(LowRankQP)
res <- LowRankQP(Vmat=Dmat, dvec=dvec, Amat=A, bvec=0, uvec=C)
sol <- res$alpha
SV <- sol > 0.01 #simulate = 0

ys <- Y[SV][1]
xs <- data[,SV][,1]
tmp <- 0
for(i in 1:length(sol[SV])){
  tmp <- tmp + sol[SV][i] * Y[SV][i] * K(data[,SV][,i], xs)
}
b <- ys - tmp
b = b[1,1]

svm <- function(x){
  r <- 0
  for(i in 1:length(sol[SV])){
    r <- r + sol[SV][i] * Y[SV][i] * K(data[,SV][,i], x)
  }
  return(r)
}

#Level-2 learning - logistic
lc  <- c(0, 0, 0, 1, 1)
#Raw feature
tx <- data[1,]
ty <- data[2,]
rdata <- data.frame(tx, ty, lc)

res <- glm(lc ~ tx + ty, data=rdata, family=binomial)
w <- coefficients(res)
g <-  function(x,y){
  return(w[1] + w[2]*x + w[3] * y )
}

o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- g(i,j)
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
points(o_x,o_y, pch = "o", col="blue")

#z-space feature
tf <- apply(data, MARGIN = 2, svm)
ldata <- data.frame(tf, lc)

res <- glm(lc ~ tf, data=ldata, family=binomial)
w <- coefficients(res)
g <-  function(x,y){
  return(w[1] + w[2]* svm(c(x,y)))
}

o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- g(i,j)
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
points(o_x,o_y, pch = "o", col="blue")


###############################
#Kernel Logistic Regression(KLR)

# ksmooth package
# https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ksmooth.html
# http://stackoverflow.com/questions/27112181/can-you-perform-a-kernel-logistic-regression-in-r

#L2
#http://r.789695.n4.nabble.com/different-L2-regularization-behavior-between-lrm-glmnet-and-penalized-td900625.html
#L1 L2
#https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization

#LiblineaR
library(LiblineaR)

n          <- 5
data <- matrix(c(-2, 1, 0, 0, 1, 2, 2, 0, 3, -2),2,n)
Y  <- c(-1, -1, -1, 1, 1)

#plot data point
plot(c(-2,0,1),c(1,0,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3),c(0,-2), pch = "o", col="blue")


#Raw feature
lambda <- 1
res <- LiblineaR(t(data), Y, type=0, cost=lambda)
w <- res$W
g <-  function(x,y){
  return(w[3] + w[1]*x + w[2] * y )
}

o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- g(i,j)
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
points(o_x,o_y, pch = "o", col="blue")



n          <- 6
data <- matrix(c(-2, 1, 0, 0, 1, 2, 2, 0, 3, -2, -4, 2),2,n)
Y  <- c(-1, -1, -1, 1, 1, 1)

#plot data point
plot(c(-2,0,1),c(1,0,2), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
points(c(2,3, -4),c(0,-2, 2), pch = "o", col="blue")


#Kernel Function
RBFKernel <- function(r){
  kernel <- function(x1, x2){
    v <- x1 - x2
    return (exp(-r * ( t(v) %*% v )^2))
  }
  return(kernel)
}
K <- RBFKernel(0.1)
m <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    m[i,j] <- K((data[, i]), data[, j])
  }
}

beta <- 0
for(i in 1:n){
  for(j in 1:n){
    beta <- beta + K(data[,i],data[,j])
  }
}

#It can be seen as the inverse of a regularization constant. 
lambda <- 0.0000000001
#res <- LiblineaR(m, Y, type=0, cost=lambda * beta) #just a scaling, no need beta
res <- LiblineaR(m, Y, type=0, cost=lambda)
w <- res$W
g <-  function(x,y){
  bias <- w[n + 1]
  tmp <- 0
  for(i in 1:n){
    tmp <- tmp + w[i] * K(data[,i], matrix(c(x,y)))
  }
  return(tmp + bias)
}


o_x <- c()
o_y <- c()
x_x <- c()
x_y <- c()
n_x <- c()
n_y <- c()
for(i in -5:5){
  for(j in -5:5){
    r <- g(i,j)
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
points(o_x,o_y, pch = "o", col="blue")


##########
# Kernel Rigdge Regression = LSSVM (Least-squares SVM)

#glmnet
#http://stats.stackexchange.com/questions/72251/an-example-lasso-regression-using-glmnet-for-binary-outcome
#http://machinelearningmastery.com/penalized-regression-in-r/
#library(glmnet)
library(MASS)
#pseudo inverse
X <- matrix(c(1,2,3,4),2,2)
X_I <- ginv(X)
I <-  X %*% X_I

#LR 
#target function y=x
x <- 2:10
y <- runif(length(x), -2, 2) + x
plot(x, y, xlim = c(0, 10), ylim=c(0, 10))
X <- matrix(x)
X <- cbind(X, rep(1, length(x)))
Y <- matrix(y)


W <- ginv(X) %*% Y
f <- function(x){return(W[1] * x + W[2])}
plot(x, y, xlim = c(0, 10), ylim=c(0, 10))
lines(0:10, f(0:10), col="green")

#Ridge Regression
#http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/9-1.pdf

lambda <- 1
I <- diag(dim(X)[2])
#inverse matrix -> solve
#http://www.statmethods.net/advstats/matrix.html
W <- solve(t(X) %*% X  + lambda * I) %*% t(X)  %*% Y
f <- function(x){return(W[1] * x + W[2])}
plot(x, y, xlim = c(0, 10), ylim=c(0, 10))
lines(0:10, f(0:10), col="green")


#Kernel Ridge regression
#http://stats.stackexchange.com/questions/183074/implementing-kernel-ridge-regression

n <- length(x)
RBFKernel <- function(r){
  kernel <- function(x1, x2){
    v <- x1 - x2
    return (exp(-r * ( t(v) %*% v )^2))
  }
  return(kernel)
}
K <- RBFKernel(0.1)
mK <- matrix(0, n, n)
for(i in 1:n){
  for(j in 1:n){
    mK[i,j] <- K((X[i ,]), X[j, ])
  }
}
lambda <- 0
I <- diag(dim(mK)[2])
B <- solve(lambda * I + mK) %*% Y
g <- function(x){
  tmp <- 0
  for(i in 1:n){
    tmp <- tmp + B[i] * K(X[i,], matrix(c(x, 1)))
  }
  return(tmp)
}
plot(x, y, xlim = c(0, 10), ylim=c(0, 10))
lines(1:10, lapply(1:10,g), col="green")

#SVR
#http://eizoo.hatenablog.com/entry/2013/07/04/033749
library(kernlab)
x <- 2:10
y <- runif(length(x), -2, 2) + x
plot(x, y, xlim = c(0, 10), ylim=c(0, 10))
X <- matrix(x)
#X <- cbind(X, rep(1, length(x)))
Y <- matrix(y)

ep <- 1e-2  # ε-insensitive
SVR1 <- ksvm(x=x,y=y,scaled=F,type="eps-svr", C=100,epsilon=ep ,kernel="rbfdot")  # Gaussian Kernel

plot(x,y,col=4,lwd=2)
lines(SVR1@fitted~x,col=2,lwd=2,ann=F,type="l")

#kpar=degree, scale, offset
SVR2 <- ksvm(x=x,y=y,scaled=F,type="eps-svr",epsilon=1e-10,kernel="polydot",kpar=list(3,3,2))
plot(x,y,col=4,lwd=2)
lines(SVR2@fitted~x,col=2,lwd=2,ann=F,type="l")
