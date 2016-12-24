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
