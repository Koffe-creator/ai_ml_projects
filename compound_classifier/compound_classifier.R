require(tidyverse)
require(caret)
require(caretEnsemble)
require(e1071)
require(xgboost)
require(pls)
require(nnet)
require(kernlab)
require(klaR)
require(pROC)
require(limma)

# ============================================================
# USER CONFIGURATION — edit this section before running
# ============================================================
seed     <- 123
namesies <- "annot_v5"

# Pick a combination of algorithms to run:
#   "glmnet"     — Elastic net (GLM with L1/L2 regularization)
#   "rf"         — Random forest
#   "pls"        — Partial least squares
#   "gbm"        — Gradient boosting
#   "svmLinear2" — Linear SVM
#   "nnet"       — Neural network
#selected_methods <- c("glmnet", "rf", "pls", "gbm", "svmLinear2", "nnet")
selected_methods <- c("glmnet","rf")
# ============================================================

dir.create("output", showWarnings = FALSE)

# Load features and hold out unclassified compounds for post-hoc prediction
itsa_train  <- read_csv("data/compound_features.csv")
itsa_train  <- itsa_train[, 2:ncol(itsa_train)]
sample_annot <- read_csv("data/compound_names.csv")

itsa_vip   <- as.data.frame(itsa_train[is.na(itsa_train$Class), ])
itsa_train <- as.data.frame(itsa_train[!is.na(itsa_train$Class), ])

itsa_train$Class <- gsub("1", "case", itsa_train$Class, fixed = TRUE)
itsa_train$Class <- gsub("0", "ctrl", itsa_train$Class, fixed = TRUE)

set.seed(seed)
trainIndex <- createDataPartition(itsa_train$Class, p = .8, list = FALSE, times = 1)
training   <- itsa_train[trainIndex, ]
testing    <- itsa_train[-trainIndex, ]

# 10-fold CV repeated 10x; ROC is the optimization metric
my_control <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           classProbs = TRUE,
                           allowParallel = TRUE,
                           summaryFunction = twoClassSummary)

model_list <- caretList(x = training[, colnames(training) != "Class"],
                        y = training[, "Class"],
                        preProcess = c("nzv", "scale", "center"),
                        trControl = my_control,
                        methodList = selected_methods)

print(as.data.frame(predict(model_list, newdata = head(testing))))

resamps  <- resamples(model_list)
sum_val  <- summary(resamps)
write.csv(sum_val$statistics$ROC,
          file = paste0("output/ROC_training_summary_table_", namesies, ".csv"))

for (metric in c("ROC", "Sens", "Spec")) {
  png(paste0("output/all_models_", metric, "_", namesies, ".png"))
  trellis.par.set(caretTheme())
  print(dotplot(resamps, metric = metric))
  dev.off()
}

# caretStack fits a meta-model on top of base learners.
# Use a separate trainControl here — sharing one with model_list causes index misalignment.
glm_ensemble <- caretStack(
  model_list,
  method = "glm",
  metric = "ROC",
  trControl = trainControl(
    method = "boot",
    number = 10,
    savePredictions = "final",
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
)

saveRDS(model_list$rf, file = paste0("output/rf_model_", namesies, ".RDS"))

# Variable importance for all selected models
allImp <- do.call(cbind, lapply(glm_ensemble$models, function(m) {
  varImp(m, scale = TRUE)$importance$Overall
}))
allImp           <- as.data.frame(allImp)
row.names(allImp) <- row.names(varImp(glm_ensemble$models[[1]], scale = TRUE)$importance)
colnames(allImp)  <- paste(names(glm_ensemble$models), namesies, sep = "_")
allImp$probe      <- row.names(allImp)
write.csv(allImp, file = paste0("output/variableImportance_", namesies, ".csv"))

# Predict probabilities on held-out test set
model_preds <- lapply(model_list, predict, newdata = testing, type = "prob")
model_preds <- lapply(model_preds, function(x) x[, "case"])
model_preds <- data.frame(model_preds)
model_preds$ensemble <- predict(glm_ensemble, newdata = testing, type = "prob")

AUCsum  <- caTools::colAUC(model_preds, testing$Class)
bestROC <- AUCsum
itsa_name <- namesies

# ROC plots with Youden-optimal threshold; collect cutoffs and hard calls per model
roc_list   <- list()
cutoff_vec <- numeric()
calls_df   <- data.frame(row.names = seq_len(nrow(testing)))

for (m in colnames(model_preds)) {
  jpeg(paste0("output/", m, "_ROC_", namesies, ".jpeg"))
  roc_obj      <- roc(testing$Class, model_preds[[m]])
  plot(roc_obj, type = "S", best.method = "youden", print.thres = "best", print.auc = TRUE)
  dev.off()

  roc_list[[m]]        <- roc_obj
  best                 <- coords(roc_obj, "b", ret = "t")
  cutoff_vec[m]        <- best[[1]]
  calls_df[[paste0(m, "_call")]] <- ifelse(model_preds[[m]] > best[[1]], "case", "ctrl")
}

# Derive per-model Youden cutoffs and apply them to produce hard calls
AUCsummer <- rbind(AUCsum, cutoff_vec)
row.names(AUCsummer)[2] <- "cutoff"
write.csv(AUCsummer, paste0("output/all_test_ROCs_", namesies, ".csv"))

final_calls <- cbind(calls_df, Class = testing$Class, model_preds, testing)
write.csv(final_calls, file = paste0("output/calls_", namesies, ".csv"))
