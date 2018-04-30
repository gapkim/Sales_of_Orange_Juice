##############################
### Initial Setup

knitr::opts_chunk$set(comment=NA, echo=FALSE, warning=TRUE, message=FALSE,
                      fig.align="center")
options(digits=4)
rm(list=ls())

library(ISLR)
data(OJ)

sum(is.na(OJ))

#################################
### EXPLORATORY DATA ANALYSIS
#################################

OJ$StoreID = as.factor(OJ$StoreID)
OJ$SpecialCH = as.factor(OJ$SpecialCH)
OJ$SpecialMM = as.factor(OJ$SpecialMM)
OJ$STORE = as.factor(OJ$STORE)

### Table 1: Total count of factors in categorical variables

library(htmlTable)

tab1 = c(table(OJ$Purchase), table(OJ$SpecialCH), table(OJ$SpecialMM), table(OJ$Store7))
tab2 = c(table(OJ$StoreID), table(OJ$STORE))

htmlTable(tab1,
          caption="Table 1: Total count of factors in categorical variables",
          header = names(tab1),
          cgroup = c("Purchase","SpecialCH","SpecialMM", "Store7"),
          n.cgroup = c(2,2,2,2),
          css.cell = "width: 60px;")

htmlTable(tab2,
          caption="Table 1(continued): Total count of factors in categorical variables",
          header = names(tab2),
          cgroup = c("StoreID","STORE"),
          n.cgroup = c(5,5),
          css.cell = "width: 60px;")

### Figure 1: Boxplots of quantitative variables

library(reshape2)
library(ggplot2)
melt.OJ = melt(OJ)

ggplot(data=melt.OJ) +
   geom_boxplot(aes(x="", y=value)) +
   facet_wrap(~variable, scale="free") +
   labs(x="", y="Values")


library(corrplot)
OJ.cor = cor(OJ[,c(2,4,5,6,7,10,11,12,13,15,16,17)])
corrplot(OJ.cor, method="square", type="upper")

#######################################################
### Sample stratification (Not used in the report)

library(caret)

feature.str = names(OJ)
feature.groups = paste(OJ$Purchase,
                    OJ$SpecialCH,
                    OJ$SpecialMM,
                    OJ$StoreID,
                    OJ$Store7,
                    OJ$STORE)

set.seed(21)
folds.id = createDataPartition(feature.groups, times=1, p=0.732)

########################################
### ANALYSIS WITH DECISION TREE
########################################

### User function definition

my.metric = function(cm){
   accuracy = round((cm[1,1]+cm[2,2])/sum(cm),4)
   error.rate = round((cm[1,2]+cm[2,1])/sum(cm),4)
   type.1 = round(cm[2,1]/sum(cm[,1]),4)
   type.2 = round(cm[1,2]/sum(cm[,2]),4)
   sensitivity = round(cm[2,2]/sum(cm[,2]),4)
   specificity = round(cm[1,1]/sum(cm[,1]),4)
   
   metric = c(accuracy, error.rate, type.1, type.2, sensitivity, specificity)
   return(metric)
}

print.metric = function(cm) {
   accuracy = round((cm[1,1]+cm[2,2])/sum(cm),4)
   error.rate = round((cm[1,2]+cm[2,1])/sum(cm),4)
   type.1 = round(cm[2,1]/sum(cm[,1]),4)
   type.2 = round(cm[1,2]/sum(cm[,2]),4)
   sensitivity = round(cm[2,2]/sum(cm[,2]),4)
   specificity = round(cm[1,1]/sum(cm[,1]),4)
   
   metric = c(accuracy, error.rate, type.1, type.2, sensitivity, specificity)

   cat(paste("Overall accuracy =", metric[1]),
      paste("Error rate =", metric[2]),
      paste("Type I error =", metric[3]),
      paste("Type II error =", metric[4]),
      paste("Sensitivity =", metric[5]),
      paste("Specificity =", metric[6]),
      sep="\n")
}

print.cm = function(cm, lab.name, cap) {
   table.cm = unname(cm)
   colnames(table.cm) = lab.name
   rownames(table.cm) = lab.name
   
   
   out.cm = htmlTable(table.cm,
             caption=paste(cap),
             cgroup = c("Actual"),
             rowlabel = "Predicted",
             css.cell = "width: 100px;")
   return(out.cm)
}

######################################
### Training and test datasets

set.seed(23)
train.id = sample(1:nrow(OJ), 800)
#train.id = folds.id[[1]]     # Goes with stratified sampling (not used)
test.id = -train.id

Purchase = OJ$Purchase
OJ.test = OJ[test.id,]
Purchase.test = Purchase[test.id]

CH.ratio = sum(Purchase.test == "CH") / length(Purchase.test)

library(tree)

tree.OJ = tree(Purchase ~ ., data=OJ, subset=train.id)
summary_tree.OJ = summary(tree.OJ)
train.error.tree = summary_tree.OJ$misclass[1] / summary_tree.OJ$misclass[2]

### Figure 3: Unpruned decision tree resulting from the training set

par(mar=c(2,2,2,2))
plot(tree.OJ)
text(tree.OJ)

summary(tree.OJ)

tree.OJ
tree.pred = predict(tree.OJ, newdata=OJ.test, type="class")
tree1.cm = table(tree.pred, Purchase.test)
test.error.tree = my.metric(tree1.cm)[2]
#test.error.tree

print.cm(tree1.cm, colnames(tree1.cm), "Table 2: Confusion matrix resulting from <br> prediction of unpruned tree using test set")

### Figure 4: Cross validation from the training set to find optimal tree size

library(ggplot2)

set.seed(30)
cv.OJ = cv.tree(tree.OJ, FUN=prune.misclass, K=30)
opt.size = cv.OJ$size[which.min(cv.OJ$dev)]
#opt.size

ggplot() +
   geom_point(aes(x=cv.OJ$size, y=cv.OJ$dev), size=2) +
   geom_line(aes(x=cv.OJ$size, y=cv.OJ$dev)) +
   scale_x_continuous(breaks=c(1:8)) +
   labs(x="Number of terminal leaf", y="Misclassifiation")

prune.OJ = prune.misclass(tree.OJ, best=opt.size)
summary_prune.OJ = summary(prune.OJ)
train.error.pruned = summary_prune.OJ$misclass[1] / summary_prune.OJ$misclass[2]

### Figure 5: Pruned tree with optimal terminal leaf number of 2

par(mar=c(2,2,2,2))
plot(prune.OJ)
text(prune.OJ)

prune.OJ

tree.prune.pred = predict(prune.OJ, newdata=OJ.test, type="class")
tree.prune.cm = table(tree.prune.pred, Purchase.test)
test.error.pruned = my.metric(tree.prune.cm)[2]

### Table 3: Confusion matrix resulting from <br> the pruned tree using the test dataset

print.cm(tree.prune.cm, colnames(tree.prune.cm), "Table 3: Confusion matrix resulting from <br> the pruned tree using the test dataset")

### Figure 5: Comparison of errors from unpruned and pruned decision tree

error.unpruned = c(train.error.tree, test.error.tree)
error.pruned = c(train.error.pruned, test.error.pruned)

tab3 = cbind(error.unpruned, error.pruned)
rownames(tab3) = c("Train.Error","Test.Error")
colnames(tab3) = c("Unpruned", "Pruned")

tab3.melt = melt(tab3)

ggplot(data=tab3.melt) +
   geom_col(aes(x=Var1, y=value)) +
   geom_text(aes(x=Var1, y=value+0.01, label=as.vector(tab3))) +
   facet_wrap(~ Var2) +
   scale_x_discrete(labels=c("Train","Test")) +
   labs(x="", y="Error")

library(randomForest)

set.seed(30)
bag.OJ = randomForest(Purchase ~ ., data=OJ, subset=train.id, 
                      mtry=ncol(OJ)-1, ntree=500, importance=T)
#bag.OJ

bag.OJ.pred = predict(bag.OJ, newdata=OJ.test, type="class")
bag.OJ.cm = table(bag.OJ.pred, Purchase.test)
test.error.bag = my.metric(bag.OJ.cm)[2]
#test.error.bag

### Table 4: Confusion matrix resulting <br/> from Bagging classifier

print.cm(bag.OJ.cm, colnames(bag.OJ.cm), "Table 4: Confusion matrix resulting <br/> from Bagging classifier")

varImpPlot(bag.OJ, sort=T, n.var=8, type=2, main="Bagging classifier")

set.seed(30)
rf.OJ = randomForest(Purchase ~ ., data=OJ, subset=train.id, 
                     mtry=3, ntree=500, importance=T)
#rf.OJ

rf.OJ.pred = predict(rf.OJ, newdata=OJ.test, type="class")
rf.OJ.cm = table(rf.OJ.pred, Purchase.test)
test.error.rf = my.metric(rf.OJ.cm)[2]
#test.error.rf

### Table 5: Confusion matrix resulting <br/> from Random Forest classifier

print.cm(rf.OJ.cm, colnames(rf.OJ.cm), "Table 5: Confusion matrix resulting <br/> from Random Forest classifier")

varImpPlot(rf.OJ, sort=T, n.var=8, type=2, main="Random Forest classifier")

library(gbm)

OJ.boost = OJ
OJ.boost$Purchase = factor(OJ.boost$Purchase, levels=c("CH","MM"), labels=c(0,1))
OJ.boost$Purchase = as.numeric(OJ.boost$Purchase)-1

set.seed(30)
boost.OJ = gbm(Purchase ~ ., data=OJ.boost[train.id, ], distribution="bernoulli",
               n.trees=5000, interaction.depth=2, shrinkage=0.001)

boost.OJ.pred = predict(boost.OJ, newdata=OJ.boost[test.id, ], n.trees=5000, type="response")

#summary(boost.OJ.pred)

boost.pred = rep("CH", length(boost.OJ.pred))
boost.pred[which(boost.OJ.pred > 0.5)] = "MM"

boost.OJ.cm = table(boost.pred, Purchase.test)

test.error.boost = my.metric(boost.OJ.cm)[2]

### Table 6: Confusion matrix resulting <br/> from Gradient Boosting classifier

print.cm(boost.OJ.cm, colnames(boost.OJ.cm), "Table 6: Confusion matrix resulting <br/> from Gradient Boosting classifier")

### Figure 8: Variable importance plot from Gradient Boosting classifier

summary.boost = summary(boost.OJ, plotit=F)[1:8,]
summary.boost$var = factor(summary.boost$var, levels=summary.boost$var)

ggplot(data=summary.boost) +
   geom_col(aes(x=var, y=rel.inf)) +
   labs(x="Variable", y="Relative influence")

### Figure 9: Comparison of test errors of all classifiers used in this study

test.error = c(test.error.tree, test.error.pruned, test.error.bag, test.error.rf, test.error.boost)
names(test.error) = c("Unpruned", "Pruned", "Bagging", "RandomForest", "Boosting")

test.error = melt(test.error)
test.error$classifier = factor(rownames(test.error), levels=rownames(test.error))

ggplot(data=test.error) +
   geom_col(aes(x=classifier, y=value)) +
   labs(x="Classifier", y="Test error")
