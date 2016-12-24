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

