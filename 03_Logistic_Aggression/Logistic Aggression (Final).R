###############################################################################
# SIOP ML Competition 2019
# Team Name: Logistic Aggression
###############################################################################

# 1. Import & Combine Data ----------------------------------------------------

# Import raw data
df_train <- read.csv("siop_ml_train_participant.csv", stringsAsFactors = F)
df_dev <- read.csv("siop_ml_dev_participant.csv", stringsAsFactors = F)
df_test <- read.csv("siop_ml_test_participant.csv", stringsAsFactors = F)

# Combine into one dataframe
df <- rbind(
  df_train[, c("Respondent_ID", paste0("open_ended_", 1:5))], df_dev, df_test
)

# Concatenate the 5 text responses into one column of text
paste_wo_NA <- function(x){
  not_NA <- ifelse(!is.na(x), TRUE, FALSE)
  new_str <- x[not_NA]
  return(paste(new_str, collapse = " "))
}
df$pasted <- apply(df[, paste0("open_ended_", 1:5)], 1, paste_wo_NA)

# 2. Topic Modeling -----------------------------------------------------------

library(quanteda)
library(ldatuning)
library(magrittr)
dtms <- lda_metrics <- lda_models <- lda_probs <- list()

# Store names of text variables
text_vars <- c(paste0("open_ended_", 1:5), "pasted")

for(i in text_vars){
  
  # Create document-term matrices
  dtms[[i]] <- dfm(
    corpus(df, text_field = i)
    ,tolower = TRUE
    ,remove = stopwords('english')
    ,remove_punct = TRUE
    ,remove_numbers = TRUE
    ,remove_symbols = TRUE
    ,remove_url = FALSE
    ,stem = FALSE
    ,ngrams = 1:5
  )
  
  # Remove tokens that occur in only 1 document
  dtms[[i]] <- dfm_trim(dtms[[i]], min_docfreq = 2)
  
  # Gather metrics to decide how many topics to extract
  lda_metrics[[i]] <- FindTopicsNumber(
    dtms[[i]]
    ,topics = seq(2, 75, 1)
    ,metrics = c("Griffiths2004", "CaoJuan2009")
    ,method = "Gibbs"
    ,control = list(seed = 77)
    ,mc.cores = parallel::detectCores() - 1
  )
  
}

# Identify optimal number of topics to extract (k)...
metrics <- lapply(lda_metrics, function(x){
  return(c(
    x$topics[which.max(x$Griffiths2004)]
    ,x$topics[which.min(x$CaoJuan2009)]
  ))
}) %>% do.call(rbind, .) %>% set_colnames(c("Griffiths", "CaoJuan"))

# ...and also inspect lda_metrics visually
FindTopicsNumber_plot(lda_metrics$open_ended_1)
FindTopicsNumber_plot(lda_metrics$open_ended_2)
FindTopicsNumber_plot(lda_metrics$open_ended_3)
FindTopicsNumber_plot(lda_metrics$open_ended_4)
FindTopicsNumber_plot(lda_metrics$open_ended_5)
FindTopicsNumber_plot(lda_metrics$pasted)

# Decided on the following number of topics (k) to extract for each SJT item
opt_k <- c(
   45  # open_ended_1
  ,36  # open_ended_2
  ,42  # open_ended_3
  ,51  # open_ended_4
  ,60  # open_ended_5
  ,26  # pasted
) %>% set_names(text_vars)

# Run and store LDA topic models
library(topicmodels)

for(i in text_vars){
  
  lda_models[[i]] <- LDA(
    dtms[[i]]
    ,k = opt_k[i]
    ,method = "Gibbs"
    ,control = list(seed = 77)
  )
  
  # Extract posterior probabilities
  lda_probs[[i]] <- posterior(lda_models[[i]])$topics
  
  # Name topics according to the questions they came from
  if(which(text_vars == i) <= 5){
    question <- paste0("OE", which(text_vars == i))
  } else {
    question <- "pasted"
  }
  colnames(lda_probs[[i]]) <- 
    paste0("topic_", 1:ncol(lda_probs[[i]]), "_", question)
  
}

# Combine all posterior probabilities in one dataframe
lda_probs_agg <- lda_probs %>% do.call(cbind, .)

# 3. Dictionary Lookups -------------------------------------------------------

library(quanteda.dictionaries)

dictionary_list <- list(
  data_dictionary_AFINN
  ,data_dictionary_geninqposneg
  ,data_dictionary_HuLiu
  ,data_dictionary_LaverGarry
  ,data_dictionary_LoughranMcDonald
  ,data_dictionary_LSD2015
  ,data_dictionary_MFD
  ,data_dictionary_NRC
  ,data_dictionary_RID
  ,data_dictionary_sentiws
  ,data_dictionary_uk2us
)

# Only keep these variables for the first dictionary lookup
dont_repeat <- c("docname", "Segment", "WC", "WPS", "Sixltr", "Dic", 
                 "AllPunc", "Period", "Comma", "Colon", "SemiC", "QMark",
                 "Exclam", "Dash", "Quote", "Apostro", "Parenth", "OtherP")

for(i in 1:length(dictionary_list)){
  if(i == 1){  # i.e., keep the 'dont_repeat' variables above
    dictionaries <- liwcalike(df$pasted, dictionary_list[[i]])
    dictionaries <- 
      dictionaries[,-which(colnames(dictionaries) %in% c("docname","Segment"))]
  } else {  # i.e., do not keep the 'dont_repeat' variables above
    tmp <- liwcalike(df$pasted, dictionary_list[[i]])
    dictionaries <- cbind(
      dictionaries
      ,tmp[,-which(colnames(tmp) %in% dont_repeat)]
    )
  }
}

# Name the dictionary variables
colnames(dictionaries) <- 
  paste0("dict_", make.unique(colnames(dictionaries), sep = "."))

# 4. Sentiment Analysis -------------------------------------------------------

# Single sentiment score that accounts for negation, amplification, etc.
sentiment <- apply(df[, text_vars], 2, function(x){
  sentimentr::sentiment_by(x)$ave_sentiment
}) %>% set_colnames(c(paste0("sent", 1:5), "sent_pasted"))

# Emotion specific scores (e.g., joy, fear, surprise, etc.)
nrc_sentiment <- syuzhet::get_nrc_sentiment(df$pasted)
colnames(nrc_sentiment) <- paste0("nrc_", colnames(nrc_sentiment))

# 5. Readability Indices-------------------------------------------------------

# Coleman Liau short
# Formula includes average word length, # of sentences, and # of words
readability_CL <- apply(df[, paste0("open_ended_", 1:5)], 2, function(x){
  quanteda::textstat_readability(x)$Coleman.Liau.short
})
readability_CL <- cbind(readability_CL, rowMeans(readability_CL)) %>% 
  set_colnames(c(paste0("readability_CL", 1:5), "readability_CL_pasted"))

# Danielson Bryan 2
# Forumla includes # of characters, # of blanks, and number of sentences
readability_DB <- apply(df[, paste0("open_ended_", 1:5)], 2, function(x){
  quanteda::textstat_readability(x)$Danielson.Bryan.2
}) 
readability_DB <- cbind(readability_DB, rowMeans(readability_DB)) %>% 
  set_colnames(c(paste0("readability_DB", 1:5), "readability_DB_pasted"))

# 6. Lexical Diversity --------------------------------------------------------

# Create dtms with 'near-zero variance' tokens removed
dtms_nzv <- list()

for(i in 1:5){
  
  dtms_nzv[[i]] <-dfm(
    corpus(df, text_field = paste0("open_ended_", i))
    ,tolower=TRUE
    ,remove_punct=TRUE
    ,remove_numbers=TRUE
    ,remove_symbols=TRUE
    ,ngrams = 1:5
  ) %>% dfm_tfidf(.)
  
  nzv_index <- caret::nearZeroVar(as.matrix(dtms_nzv[[i]]))
  dtms_nzv[[i]] <- dtms_nzv[[i]][,-nzv_index]
  colnames(dtms_nzv[[i]]) <- paste0(colnames(dtms_nzv[[i]]), "_OE", i)
  
}

dtm_agg <- dtms_nzv %>% do.call(cbind, .)

# These indices use # of unique tokens and # of total tokens
lex_div <- quanteda::textstat_lexdiv(dtm_agg)

# 7. Misc. Feature Engineering ------------------------------------------------

# Average number of characters per word
chars <- apply(df[, paste0("open_ended_", 1:5)], 2, nchar) 
words <- apply(df[, paste0("open_ended_", 1:5)], 2, quanteda::ntoken)
avg_token_length <- (chars / words)
avg_token_length <- 
  cbind(avg_token_length, rowMeans(chars) / rowMeans(words)) %>%
  set_colnames(c(paste0("char_per_word", 1:5), "char_per_word_pasted"))

# Spelling errors
errors <- unlist(lapply(hunspell::hunspell(df$pasted), length))

# Profanity
profanity <- sentimentr::profanity_by(df$pasted)$profanity_count

# 8. Combine Predictors -------------------------------------------------------

X <- cbind(
  lda_probs_agg
  ,dictionaries
  ,sentiment
  ,nrc_sentiment
  ,readability_CL
  ,readability_DB
  ,lex_div_R = lex_div$R  # Guirad's Root TTR
  ,lex_div_C = lex_div$C  # Herdan's C
  ,lex_div_D = lex_div$D  # Simpson's D
  ,avg_token_length
  ,errors
  ,profanity
)

###############################################################################
# Tune, Train, & Predict
###############################################################################

# 9. Feature Selection --------------------------------------------------------

# For each DV, eliminate vars w/ correlation p-values above some threshold
dvs <- paste0(c("E","A","C","N","O"), "_Scale_score")
ps <- seq(.01, .2, length.out = 10)
iterations <- 100
output <- matrix(NA, ncol = length(ps), nrow = iterations,
                 dimnames = list(NULL, ps))
cor_list <- rep(list(output), length(dvs)) %>% set_names(dvs)

for(h in dvs){
  
  cors <- psych::corr.test(df_train[, h], X[1:1088,], adjust = "none")
  
  for(i in 1:iterations){
    
    set.seed(i)
    train_ids <- sample(1:1088, 788, replace = F)
    
    for(j in ps){
      
      keep <- colnames(X)[which(cors$p < j)]
      lm_model <- lm(df_train[train_ids, h] ~ .,
                     data = X[1:1088,][train_ids, keep])
      cor_list[[h]][i, as.character(j)] <- cor(
        predict(lm_model, X[1:1088,][-train_ids, keep])
        ,df_train[-train_ids, h]
      )
    }
  }
}

# Print best p-values for each DV
opt_p <- lapply(cor_list, function(x){names(which.max(colMeans(x)))})

# After multiple rounds of tuning, the p-value thesholds used were:
opt_p <- c(
   .01322222222  # E
  ,.00944444444  # A
  ,.03666666666  # C
  ,.10333333333  # N
  ,.11111111111  # O
) %>% set_names(dvs)

# Store all variables to keep in a list
keep_vars <- list()
for(h in dvs){
  cors <- psych::corr.test(df_train[, h], X[1:1088, ], adjust = "none")
  keep_vars[[h]] <- colnames(X)[which(cors$p < opt_p[h])]
}

# 10. Elastic Net -------------------------------------------------------------

library(caret)
glm_tuning <- list()

for(h in dvs){
  
  # Set up repeated CV in caret
  folds <- createMultiFolds(df_train[, h], k = 10, times = 5)
  control <- trainControl(
    method = "repeatedcv"
    ,number = 10
    ,repeats = 5
    ,returnResamp = "final"
    ,index = folds
    ,summaryFunction = defaultSummary
    ,selectionFunction = "oneSE"
  )
  
  # Tune elastic net model
  model <- caret::train(
    x = as.matrix(X[1:1088, keep_vars[[h]] ])
    ,y = df_train[, h]
    ,method = "glmnet"
    ,metric = "Rsquared"
    ,trControl = control
    ,tuneLength = 10
  )
  
  # Store best values of alpha and lambda
  glm_tuning[[h]] <- model$bestTune
  
}

# With caret's default tuning parameters,
# all models performed best when both alpha and lambda were ~.1
elastic_params <- glm_tuning %>% do.call(rbind,.)

# 11. Generate Predictions ----------------------------------------------------

# Import test submission format
test <- read.csv("siop_ml_test_submission_format.csv", stringsAsFactors = F)

library(glmnet)
glm_models <- list()

# Run models and generate predictions
for(h in dvs){
  
  glm_models[[h]] <- glmnet(
    x = as.matrix(X[1:1088, keep_vars[[h]] ])
    ,y= df_train[, h]
    ,alpha = 0.1
    ,lambda = 0.1
  )
  
  test_name <- gsub("_Scale_score", "", h)
  test[,paste0(test_name, "_Pred")] <- 
    predict(glm_models[[h]], as.matrix(X[1389:1688, keep_vars[[h]] ])) %>% 
    as.vector
  
}