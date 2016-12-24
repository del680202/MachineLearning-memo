#############
# soft-margin SVM
#soft-margin with kernel
n          <- 7
Y          <- c(-1,-1,-1,1,1,1, 1)
data <- matrix(c(-2, 1, 0, 0, 1, 2, 2, 0, 3, -2, -4, 2, 1, -4),2,n)
#plot data point
#plot(c(-4,0,4),c(-1,3,-1), xlim = c(-5,5), ylim = c(-5,5), pch = "x", col="red")
#points(c(-3,1, 3, 1),c(3,-1, 3, -4), pch = "o", col="blue")

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
