train_data <- read.csv(file="kdd99_training_data.csv",head=TRUE,sep=",")
test_data <- read.csv(file="kdd99_testing_data.csv",head=TRUE,sep=",")
TRAIN_NUM <- nrow(train_data)
TEST_NUM <- nrow(test_data)
CLASSNUM <- 5
train.x <- as.matrix(train_data[,1:10])
train.y <- matrix(0L,CLASSNUM,TRAIN_NUM)
for (i in 1:TRAIN_NUM){
  train.y[train_data[i,11]+1,i] <- 1
}
##(a)gradient descent algorithm
learning_rate <- 5
w <- matrix(rnorm(50),10,5)
iterative_index_gra <- 0
y<-matrix(0L,CLASSNUM,TRAIN_NUM)
error <- vector('numeric')
err <- 100
iterative_index_gra <- 0
grad.descent <- function(x, min_err){
  while (err > min_err){
    iterative_index_gra <- iterative_index_gra+1
    a = t(x %*% w)
    for(j in 1:CLASSNUM){
      y[j,] <- 1/colSums(exp(a-t(replicate(CLASSNUM,a[j,]))))
      w[,j] <- w[,j]- learning_rate * t(t((y[j,]-train.y[j,])) %*% train.x);
    }
    error <- c(error,-sum((train.y*log(y))))
    err <- error[iterative_index_gra]
  }
  return(list(iterative_index_gra, err, error))
}
ans = grad.descent(train.x,6);
print(c(ans[1],ans[2]))
plot(seq(1,ans[[1]],1),ans[[3]],main="Gradient descent", sub="", xlab="iteration index", ylab="cross-entropy error")