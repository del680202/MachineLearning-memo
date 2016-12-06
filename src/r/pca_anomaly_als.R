###PCA
#Generate test data
data(iris)
df <- data.frame(h=iris$Sepal.Length, w=iris$Sepal.Width)
df <- t(matrix(c(1,-1,1,1), 2) %*% t(df)) #turn 45 
#df <- data.frame(h=rnorm(100), w=rnorm(100))
plot(df)

#1.Mean-centering
meanVector <- apply(df, 2, mean)
X <- t(t(df) - meanVector)
plot(X)

#2.covariance matrix
sigma = (t(X) %*% X) / nrow(X)

#3.SVD for eign
S <- svd(sigma)

#4/Reduce dim to k
k <- 1
Ureduce <- S$u[1:k,]
plot(X)
arrows(0,0,Ureduce[1],Ureduce[2],col = "red")
#arrows(0,0,Ureduce[2,1],Ureduce[2,2],col = "red")
z <- Ureduce %*% data.matrix(df[1,])
#prcomp(df) test PCA

###Anomaly detection example(using Gaussian Distribution)
df <- data.frame(x=abs(rnorm(20))*2, y=abs(rnorm(20))*3)
#Fetch each feature avg and var

#When probability of a data is too small in Gaussian, it is anomaly
is_anomaly <- function(x,y, threshold){
  x_avg <- mean(df$x)
  x_var <- var(df$x)
  y_avg <- mean(df$x)
  y_var <- var(df$x)
  x_probability <- dnorm(x, x_avg, x_var)
  y_probability <- dnorm(y, y_avg, y_var)
  return(x_probability*y_probability < threshold)
}
anomaly_point <- list(x=10, y=10)
print(is_anomaly(anomaly_point$x, anomaly_point$y, 0.1))

#什麼時候適合用異常檢測
#1.當資料的label非常的偏頗 ex,正樣本資料只有總體的1%
#2. Label無法預測，異常類型有太多的變化
#異常檢測可以使用F1-scope去做評估
#異常轉換=>先對所有特徵進行分析，如果是用高斯分佈，就要想辦法讓資料符合高斯
#1. 若資料是符合二項分佈，對資料取log做特徵轉換就可以貼近高斯分佈
#2. 若單一特徵無法判斷出異常，可以考慮組合特徵 ex: (CPU Load)/(Network traffic)
#進階算法- Multivariate Gaussian Distribution 他有自動做特徵間的兩兩線性組合的效果
#當資料量大於十倍特徵量 且特徵量不算很多的時候可以考慮使用，可以省掉自己組合特徵的麻煩

###ALS + SGD
#http://qiita.com/ysekky/items/c81ff24da0390a74fc6c
# 5 users rating 4 movies
data <- matrix(c(5, 3, 0, 1, 4, 0, 0, 1, 1, 1, 0, 5, 1, 0, 0, 4, 0, 1, 5, 4), 5)
# U = m x k,  V = k x n, R =  U * V
ALS <- function(R, k){
  U <- abs(matrix( rnorm(nrow(R)*k,mean=0,sd=0.5), nrow(R), k))
  V <- abs(matrix( rnorm(ncol(R)*k,mean=0,sd=0.5), ncol(R), k)) 
  alpha=0.01
  iter_num <- 500
  for(s in 1:iter_num){
    for(i in 1:nrow(R)){
      for(j in 1:ncol(R)){
        if(R[i, j] == 0){
          next
        }
        U_old <- U[i,]
        V_old <- V[j,]
        #Using V to update U
        err <- R[i, j] - (t(U[i,]) %*% V[j,])
        U[i,] <- U_old + alpha * err * V_old
        # Using U to update V
        #err <- R[i, j] - (t(U[i,]) %*% V[j,])
        V[j,] <- V_old + alpha * err * U_old
      }
    }
  }
  return(list(U=U, V=V))
}
res <- ALS(data, 2)
print(data)
print(res$U %*% t(res$V))


