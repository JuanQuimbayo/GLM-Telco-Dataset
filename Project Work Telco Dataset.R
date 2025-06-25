## ----setup, include=FALSE-----------------------------------------------------------
knitr::opts_chunk$set(
	echo = FALSE,
	fig.align = "center",
	fig.height = 3,
	fig.width = 5,
	message = FALSE,
	warning = FALSE,
	comment = NA,
	results = "markup",
	tidy = TRUE
)
options(width = 70)



## ----echo=FALSE, message=FALSE, warning=FALSE, paged.print=TRUE---------------------
# caricamento Librerie
#install.packages(c("tidyverse", "caret", "glmnet", "sparklyr", "xgboost", "pROC", "skimr", "corrplot", "janitor"))
library(tidyverse)
library(caret)

# caricamento dataset
telco <- read_csv("telco.csv")
telco %>% head(5)



## ----echo=FALSE---------------------------------------------------------------------
# rinominiamo le variabili

telco_rename <-  telco %>%
  rename("customer_id" = `Customer ID`,
  "under_30" = `Under 30`, "senior_citizen" = `Senior Citizen`, "number_of_dependents" = `Number of Dependents`,
  "zip_code" = `Zip Code`, "referred_a_friend" = `Referred a Friend`, "number of referrals" = `Number of Referrals`,
  "tenure_in_months" = `Tenure in Months`, "phone_service" = `Phone Service`, "avg_monthly_long_distance_charges" = `Avg Monthly Long Distance Charges`,
  "multiple_lines" = `Multiple Lines`, "internet_service" = `Internet Service`, "internet_type" = `Internet Type`, "avg_monthly_gb_download" = `Avg Monthly GB Download`,
  "online_security" = `Online Security`, "online_backup" = `Online Backup`, "device_protection_plan" = `Device Protection Plan`, "premium_tech_support" = `Premium Tech Support`,
  "streaming_tv" = `Streaming TV`, "streaming_movies" = `Streaming Movies`, "streaming_music" = `Streaming Music`, "unlimited_data" = `Unlimited Data`, 
  "paperless_billing" = `Paperless Billing`, "payment_method" = `Payment Method`, "monthly_charge" = `Monthly Charge`, "total_charges" = `Total Charges`, "total_refunds" = `Total Refunds`,
  "total_extra_data_charges" = `Total Extra Data Charges`, "total_long_distance_charges" = `Total Long Distance Charges`, "total_revenue" = `Total Revenue`, 
  "satisfaction_score" = `Satisfaction Score`, "customer_status" = `Customer Status`, "churn_label" = `Churn Label`, "churn_score" = `Churn Score`, "churn_category" = `Churn Category`,
  "churn_reason" = `Churn Reason`, "population" = `Population`
  )

# variabili non utili 
not_useful <- c("customer_id", "under_30", "senior_citizen", "zip_code", 
"population")

# creazione variabile binara (0,1) churn
telco_rename <- telco_rename %>%
  mutate(Churn = ifelse(churn_label == "Yes", 1, 0)) %>%
  janitor::clean_names()

# rimozione variabili non utili
telco_drop <- telco_rename %>% select(-all_of(not_useful))

# cambio tipologia delle variabili 
telco_drop$age <- as.numeric(telco_drop$age)
telco_drop$gender <- as.factor(telco_drop$gender)
telco_drop$married <- as.factor(telco_drop$married)
telco_drop$dependents <- as.factor(telco_drop$dependents)
telco_drop$number_of_dependents <- as.numeric(telco_drop$number_of_dependents)
telco_drop$latitude <- as.numeric(telco_drop$latitude)
telco_drop$longitude <- as.numeric(telco_drop$longitude)
telco_drop$referred_a_friend <- as.factor(telco_drop$referred_a_friend)
telco_drop$offer <- as.factor(telco_drop$offer)
telco_drop$phone_service <- as.factor(telco_drop$phone_service)
telco_drop$multiple_lines <- as.factor(telco_drop$multiple_lines)
telco_drop$internet_service <- as.factor(telco_drop$internet_service)
telco_drop$internet_type <- as.factor(telco_drop$internet_type)
telco_drop$online_security <- as.factor(telco_drop$online_security)
telco_drop$online_backup <- as.factor(telco_drop$online_backup)
telco_drop$device_protection_plan <- as.factor(telco_drop$device_protection_plan)
telco_drop$premium_tech_support <- as.factor(telco_drop$premium_tech_support)
telco_drop$streaming_tv <- as.factor(telco_drop$streaming_tv)
telco_drop$streaming_movies <- as.factor(telco_drop$streaming_movies)
telco_drop$streaming_music <- as.factor(telco_drop$streaming_music)
telco_drop$unlimited_data <- as.factor(telco_drop$unlimited_data)
telco_drop$contract <- as.factor(telco_drop$contract)
telco_drop$paperless_billing <- as.factor(telco_drop$paperless_billing)
telco_drop$payment_method <- as.factor(telco_drop$payment_method)
telco_drop$satisfaction_score <- as.numeric(telco_drop$satisfaction_score)
telco_drop$customer_status <- as.factor(telco_drop$customer_status)
telco_drop$churn_label <- as.factor(telco_drop$churn_label)
telco_drop$churn_category <- as.factor(telco_drop$churn_category)
telco_drop$churn <- as.factor(telco_drop$churn)

# variabili 
telco_drop %>% head(5)


## -----------------------------------------------------------------------------------
ggplot(telco_drop, aes(x = factor(churn), fill = factor(churn))) + 
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "coral")) +
  labs(title = "Distribuzione Churn", x = "Churn", fill = "Churn")


## -----------------------------------------------------------------------------------
# Categoriche vs Churn
plot_categorical <- function(var) {
  telco_drop %>%
    group_by(!!sym(var), churn) %>%
    count() %>%
    ggplot(aes(x = !!sym(var), y = n, fill = factor(churn))) +
    geom_col(position = "fill") +
    scale_fill_manual(values = c("0" = "skyblue", "1" = "coral")) +
    labs(y = "Proporzione", fill = "Churn")
}

plot_categorical("contract")


## ----fig.align='center-left'--------------------------------------------------------
telco_drop %>%
  filter(contract == "Month-to-Month") %>%
  select(phone_service, internet_service, online_backup, online_security,
         streaming_tv, streaming_music, streaming_movies) %>%
  summarise_all(~ mean(. == "Yes", na.rm = TRUE)) %>%
  pivot_longer(cols = everything(), names_to = "Servizio", values_to = "Percentuale") %>%
  ggplot(aes(x = reorder(Servizio, Percentuale), y = Percentuale)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(title = "Servizi più utilizzati (Month-to-Month)",
       x = NULL, y = "Percentuale di utilizzo")



## -----------------------------------------------------------------------------------
plot_categorical("offer")


## ----paged.print=FALSE, fig.align='center-left'-------------------------------------
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(ggplot2)
library(dplyr)

# Mappa base degli stati USA
usa <- ne_states(country = "United States of America", returnclass = "sf")

# Filtra solo la California
california <- usa %>% filter(name == "California")

# Trasforma i dati dei clienti in oggetto sf
churn_points <- telco_drop %>%
  filter(churn == 1) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

# Plot finale
ggplot() +
  geom_sf(data = california, fill = "gray95", color = "black") +
  geom_sf(data = churn_points, color = "coral", alpha = 0.5, size = 2) +
  labs(
    title = "Distribuzione geografica dei clienti 'churn",
    x = "Longitudine", y = "Latitudine"
  ) +
  theme_minimal()



## -----------------------------------------------------------------------------------
telco_drop %>%
  group_by(dependents, churn) %>%
  count() %>%
  ggplot(aes(x = dependents, y = n, fill = factor(churn))) +
  geom_col(position = "fill") +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "coral")) +
  labs(title = "Churn in base alla presenza di persone a carico", fill = "Churn")



## -----------------------------------------------------------------------------------
library(dplyr)
library(ggplot2)

# Soglia per CLTV alto: usa la mediana o un valore che preferisci
cltv_threshold <- median(telco_drop$cltv, na.rm = TRUE)

# Filtra clienti con CLTV alto
clienti_preziosi <- telco_drop %>%
  filter(cltv > cltv_threshold)

# Controlla quanti clienti preziosi hanno basso satisfaction score
clienti_bassi_satisfaction <- clienti_preziosi %>%
  filter(satisfaction_score < 3)

#nrow(clienti_bassi_satisfaction)

# Distribuzione satisfaction score per clienti preziosi
ggplot(clienti_preziosi, aes(x = satisfaction_score)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_histogram(data = clienti_bassi_satisfaction, binwidth = 1, fill = "coral", alpha = 0.9) +
  labs(
    title = "Clienti con CLTV alto: confronto satisfaction score",
    x = "Satisfaction Score",
    y = "Numero di clienti"
  ) +
  theme_minimal()



## -----------------------------------------------------------------------------------
# train, test spli (70-30)
set.seed(123)
train_index <- sample(seq_len(nrow(telco_drop)), size = 0.7 * nrow(telco_drop))
train_set <- telco_drop[train_index, ]
test_set  <- telco_drop[-train_index, ]


## ----warning=TRUE, include=FALSE----------------------------------------------------
model.completo <- glm(churn ~ gender + age + married + dependents + number_of_dependents + latitude + longitude +
                    referred_a_friend + number_of_referrals + tenure_in_months + offer + phone_service + avg_monthly_long_distance_charges + 
                    multiple_lines + internet_service + internet_type + avg_monthly_gb_download + online_security
                    + online_backup + device_protection_plan + premium_tech_support + streaming_tv + streaming_music + streaming_movies + 
                    unlimited_data + contract + payment_method + paperless_billing + monthly_charge + total_charges + total_refunds + 
                    total_extra_data_charges + total_long_distance_charges + total_revenue + satisfaction_score + cltv,
data = train_set,
family = binomial(link = "logit"))
options(scipen = 999)
summary(model.completo)


## ----include=FALSE------------------------------------------------------------------
model.ridotto <- glm(churn ~ age + dependents + referred_a_friend+ 
                    tenure_in_months+ number_of_referrals + offer + 
                    phone_service + internet_service+
                    + avg_monthly_gb_download + online_security + 
                    online_backup + premium_tech_support + streaming_movies + 
                    contract + monthly_charge + payment_method,
data = train_set,
family = binomial(link = "logit"))
summary(model.ridotto)


## ----include=FALSE------------------------------------------------------------------
model.ridotto2 <- glm(churn ~ age + dependents + referred_a_friend+ 
                    tenure_in_months+ number_of_referrals + offer + 
                    phone_service + online_security + 
                    online_backup + premium_tech_support + 
                    contract + monthly_charge + payment_method,
data = train_set,
family = binomial(link = "logit"))
summary(model.ridotto2)


## ----echo=FALSE---------------------------------------------------------------------
anova(model.ridotto,model.ridotto2,test="LRT")


## -----------------------------------------------------------------------------------
results <- parameters::model_parameters(model.ridotto2, exponentiate = TRUE)
selected_results <- results[1:8, 1:6]  # prime 4 colonne
print(selected_results)


## ----fig.align='center-left'--------------------------------------------------------
library(ggplot2)
library(dplyr)

vars_to_plot <- c(
  "dependentsYes",
  "offerOffer D",
  "contractTwo Year",
  "online_securityYes",
  "offerOffer A",
  "offerOffer E",
  "payment_methodMailed Check",
  "monthly_charge"
)

results_subset <- results %>% filter(Parameter %in% vars_to_plot)

plot(results_subset)


## -----------------------------------------------------------------------------------

prob_ridotto <- predict(model.ridotto2, test_set, type = "response")

predicted_class <- ifelse(prob_ridotto >= 0.5, 1, 0)

accuracy <- mean(predicted_class == test_set$churn)
print(paste("Accuracy:", round(accuracy, 4)))


## -----------------------------------------------------------------------------------
library(pROC)
prob_ridotto <- predict(model.ridotto2, test_set, type = "response")
roc_ridotto <- roc(test_set$churn, prob_ridotto)
auc(roc_ridotto)


## -----------------------------------------------------------------------------------
library(ggeffects)

preds <- predict_response(model.ridotto2,terms=c("tenure_in_months[all]","online_backup","payment_method [Credit Card, Mailed Check]"),margin="empirical")

plot(preds)


## -----------------------------------------------------------------------------------
library(ggeffects)
preds2 <- predict_response(model.ridotto2,terms=c("contract[all]","offer [Offer A, Offer B, Offer C, Offer D]"),margin="empirical")

plot(preds2)


## -----------------------------------------------------------------------------------
library(ggeffects)
preds3 <- predict_response(model.ridotto2,terms=c("number_of_referrals[all]","dependents"),margin="empirical")

plot(preds3)


## -----------------------------------------------------------------------------------
# Definizione X e Y (train e test)
library(glmnet)

# Train 
x.train <- model.matrix(churn ~ gender + age + married + dependents + number_of_dependents + latitude + longitude +
                    referred_a_friend + number_of_referrals + tenure_in_months + offer + phone_service + avg_monthly_long_distance_charges + 
                    multiple_lines + internet_service + internet_type + avg_monthly_gb_download + online_security
                    + online_backup + device_protection_plan + premium_tech_support + streaming_tv + streaming_music + streaming_movies + 
                    unlimited_data + contract + payment_method + paperless_billing + monthly_charge + total_charges + total_refunds + 
                    total_extra_data_charges + total_long_distance_charges + total_revenue + cltv, data = train_set)[, -1]

y.train <- train_set$churn

# Test
x.test <- model.matrix(churn ~ gender + age + married + dependents + number_of_dependents + latitude + longitude +
                    referred_a_friend + number_of_referrals + tenure_in_months + offer + phone_service + avg_monthly_long_distance_charges + 
                    multiple_lines + internet_service + internet_type + avg_monthly_gb_download + online_security
                    + online_backup + device_protection_plan + premium_tech_support + streaming_tv + streaming_music + streaming_movies + 
                    unlimited_data + contract + payment_method + paperless_billing + monthly_charge + total_charges + total_refunds + 
                    total_extra_data_charges + total_long_distance_charges + total_revenue + cltv, data = test_set)[, -1]

y.test <- test_set$churn


## -----------------------------------------------------------------------------------
library(glmnet)

# RIDGE (alpha = 0)
cv.ridge <- cv.glmnet(x.train, y.train, alpha = 0, family = "binomial", type.measure = "deviance", nfolds = 10)

# LASSO (alpha = 1)
cv.lasso <- cv.glmnet(x.train, y.train, alpha = 1, family = "binomial", type.measure = "deviance", nfolds = 10)

# ELASTIC NET (alpha = 0.5)
cv.elastic <- cv.glmnet(x.train, y.train, alpha = 0.5, family = "binomial", type.measure = "deviance", nfolds = 10)




## -----------------------------------------------------------------------------------
# Deviance vs Lambda
par(mfrow = c(1,3))
plot(cv.ridge, main = "CV Ridge - Deviance")
plot(cv.lasso, main = "CV Lasso - Deviance")
plot(cv.elastic, main = "CV Elastic Net - Deviance")


## -----------------------------------------------------------------------------------
#Deviance lambda.1se
deviance.ridge.1se <- cv.ridge$cvm[cv.ridge$lambda == cv.ridge$lambda.1se]    
deviance.lasso.1se <- cv.lasso$cvm[cv.lasso$lambda == cv.lasso$lambda.1se]    
deviance.elastic.1se <- cv.elastic$cvm[cv.elastic$lambda == cv.elastic$lambda.1se]

print(paste("Deviance Ridge (lambda.1se):", round(deviance.ridge.1se, 4)))
print(paste("Deviance Lasso (lambda.1se):", round(deviance.lasso.1se, 4)))
print(paste("Deviance Elastic (lambda.1se):", round(deviance.elastic.1se, 4)))

#Deviance lambda.min
deviance.ridge.min <- cv.ridge$cvm[cv.ridge$lambda == cv.ridge$lambda.min]    
deviance.lasso.min <- cv.lasso$cvm[cv.lasso$lambda == cv.lasso$lambda.min]    
deviance.elastic.min <- cv.elastic$cvm[cv.elastic$lambda == cv.elastic$lambda.min]

print(paste("Deviance Ridge (lambda.min):", round(deviance.ridge.min, 4)))
print(paste("Deviance Lasso (lambda.min):", round(deviance.lasso.min, 4)))
print(paste("Deviance Elastic (lambda.min):", round(deviance.elastic.min, 4)))


## -----------------------------------------------------------------------------------
# Codice per il confronto delle performance
pred_ridge <- predict(cv.ridge, newx = x.test, type = "response", 
                      lambda = "lambda.1se")
pred_lasso <- predict(cv.lasso, newx = x.test, type = "response", 
                      lambda = "lambda.1se")
pred_elastic <- predict(cv.elastic, newx = x.test, type = "response",
                        lambda = "lambda.1se")

# Calcolo delle metriche
library(pROC)
auc_ridge <- auc(y.test, pred_ridge)
auc_lasso <- auc(y.test, pred_lasso)
auc_elastic <- auc(y.test, pred_elastic)

print(paste("AUC Ridge:", round(auc_ridge, 4)))
print(paste("AUC Lasso:", round(auc_lasso, 4)))
print(paste("AUC Elastic:", round(auc_elastic, 4)))


## -----------------------------------------------------------------------------------
#Accuracy Ridge
predicted_class_ridge <- ifelse(pred_ridge >= 0.5, 1, 0)
accuracy.ridge <- mean(predicted_class_ridge == y.test)

#Accuracy Lasso
predicted_class_lasso <- ifelse(pred_lasso >= 0.5, 1, 0)
accuracy.lasso <- mean(predicted_class_lasso == y.test)

#Accuracy Elastic
predicted_class_elastic <- ifelse(pred_elastic >= 0.5, 1, 0)
accuracy.elastic <- mean(predicted_class_elastic == y.test)

print(paste("Accuracy Ridge:", round(accuracy.ridge, 4)))
print(paste("Accuracy Lasso:", round(accuracy.lasso, 4)))
print(paste("Accuracy Elastic:", round(accuracy.elastic, 4)))



## -----------------------------------------------------------------------------------
#plot deviance per valori di lambda
plot(cv.lasso, main = "CV Lasso - Deviance")


## -----------------------------------------------------------------------------------
options(scipen = 999)
coefficienti.lasso <- coef(cv.lasso, s = "lambda.1se")

coef_df <- as.data.frame(as.matrix(coefficienti.lasso)) |>
  tibble::rownames_to_column("Variabile") |>
  dplyr::rename(Coefficiente = s0) |>
  dplyr::filter(
    Coefficiente != 0,           # Rimuove coefficienti zero
    Variabile != "(Intercept)"   # Esclude l'intercetta
  ) |>
  dplyr::arrange(desc(abs(Coefficiente)))  # Ordina per importanza

# Visualizza la tabella
print(coef_df[1:10, 1:2])


## -----------------------------------------------------------------------------------
library(sparklyr)
library(tidyverse)

# Connessione a Spark
sc <- spark_connect(master = "local")

# Caricamento dataset Telco in Spark
telco_tbl <- copy_to(sc, telco_drop, overwrite = TRUE)

# Verifica struttura
head(telco_tbl)


## -----------------------------------------------------------------------------------
library(sparklyr)

# Lista delle variabili categoriche da trasformare
categorical_vars <- c("dependents","referred_a_friend","offer", "phone_service", "online_security", "online_backup", "premium_tech_support", 
                      "contract", "payment_method")

# Step 1: Applichiamo ft_string_indexer a tutte le colonne
for (col in categorical_vars) {
  indexed_col <- paste0(col, "_index")
  telco_tbl <- ft_string_indexer(
    telco_tbl,
    input_col = col,
    output_col = indexed_col
  )
}

# Step 2: Applichiamo ft_one_hot_encoder su tutte le colonne indicizzate
indexed_vars <- paste0(categorical_vars, "_index")
encoded_vars <- paste0(categorical_vars, "_onehot")

telco_tbl <- ft_one_hot_encoder(
  telco_tbl,
  input_cols = indexed_vars,
  output_cols = encoded_vars
)


## -----------------------------------------------------------------------------------
library(sparklyr)
library(dplyr)

glm.whole.dataset <- ml_generalized_linear_regression(telco_tbl,churn ~ age + dependents_onehot + referred_a_friend_onehot + tenure_in_months + number_of_referrals + offer_onehot + phone_service_onehot + online_security_onehot + online_backup_onehot + premium_tech_support_onehot + contract_onehot + monthly_charge + payment_method_onehot, 
                                                      
family="binomial",link="logit")



## -----------------------------------------------------------------------------------
ResampleLm <- function(df_tbl, frac = 0.10, B = 100) {
  
  coef_list <- list()
  
  for (b in 1:B) {
    sub_df <- df_tbl %>%
      sdf_sample(fraction = frac, replacement = FALSE)

    glm.subset <- ml_generalized_linear_regression(sub_df,churn ~ age +  dependents_onehot + referred_a_friend_onehot + tenure_in_months + number_of_referrals + offer_onehot + phone_service_onehot + online_security_onehot + online_backup_onehot + premium_tech_support_onehot + contract_onehot + monthly_charge + payment_method_onehot, 
                                                      
family="binomial",link="logit")
    
    coef_list[[b]] <- coef(glm.subset)
  }

  coef_matrix <- do.call(rbind, coef_list)
  colnames(coef_matrix) <- names(coef_list[[1]])

  as_tibble(coef_matrix)
}


## -----------------------------------------------------------------------------------
resampledCoef.glm.subset<- ResampleLm(telco_tbl,frac=0.10, B=100)


## -----------------------------------------------------------------------------------
meanCoef.subset <- apply(resampledCoef.glm.subset, 2, mean)

meanCoef_tbl <- tibble(
  Coefficient = names(meanCoef.subset),
  Coef.subsets = as.numeric(meanCoef.subset), 
  Coef.whole.dataset = as.numeric(coef(glm.whole.dataset))
)

head(meanCoef_tbl,20)


## ----include=FALSE------------------------------------------------------------------
#disconnettiamo da Spark

spark_disconnect_all()

