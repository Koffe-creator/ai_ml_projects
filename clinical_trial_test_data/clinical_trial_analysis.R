# Load necessary libraries
library(readxl)
library(tidyverse)
library(ggplot2)
library(lme4)
library(lmerTest)
library(broom)
library(broom.mixed)
library(emmeans)

# Set working directory to the script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Define file path
file_path <- "stats_R_Python_questions_2.xlsx"

# Load the "data" sheet
data <- read_excel(file_path, sheet = "data")

# Check data structure and summary statistics
str(data)
summary(data)

### 1. Answer Key:

# 1.a) Compute mean and standard deviation
summary_stats <- data %>%
  group_by(marker, sample_time, trt_group) %>%
  summarise(
    mean_digital_count = mean(digital_count, na.rm = TRUE),
    sd_digital_count = sd(digital_count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  as.data.frame()

# View the results
print(summary_stats)

# 1.b) Plot Mean ± SD over time for each marker and treatment

# Convert sample_time to an ordered factor before plotting
summary_stats <- summary_stats %>%
  mutate(sample_time = factor(sample_time, levels = c("DAY1", "DAY8", "DAY15", "DAY22", "DAY29")))

# Create the plot using ggplot2
summary_plots <- ggplot(summary_stats, aes(x = sample_time, y = mean_digital_count, 
                                           color = trt_group, group = trt_group)) +
  geom_line(linewidth = 1) +  
  geom_point(size = 2) +  
  geom_errorbar(aes(ymin = mean_digital_count - sd_digital_count, 
                    ymax = mean_digital_count + sd_digital_count), width = 0.2) +  
  facet_wrap(~ marker, scales = "fixed") +  # Ensuring same y-axis range for all facets
  labs(title = "Marker Readout Over Time",
       x = "Sample Time", 
       y = "Mean Digital Count",
       color = "Treatment Group") +
  scale_color_brewer(palette = "Dark2") +  # New color scheme using Dark2 palette
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5, color = "black"), # Centered title
    axis.title.x = element_text(size = 14, color = "black"),  # X-axis label
    axis.title.y = element_text(size = 14, color = "black"),  # Y-axis label
    axis.text.x = element_text(size = 10, color = "black"),   # X-axis tick labels
    axis.text.y = element_text(size = 12, color = "black"),   # Y-axis tick labels
    legend.title = element_text(size = 12, color = "black"),  # Legend title
    legend.text = element_text(size = 12, color = "black"),   # Legend text
    strip.text = element_text(size = 14, face = "bold", color = "black")  # Facet titles
  )

# View the plot
print(summary_plots)

### 2. Question: For each marker use an appropriate method to test whether DAY8 and DAY1 significantly differs at alpha = 0.05 under each treatment
### Answer:

# We will use t test to test the significance at 0.05. Whenever the subjects are the same in both time points we will use a paired wise t test
# and whenever subjects are missing between the the 2 timepoints we will use a regular t test.

# Filter data for DAY1 and DAY8 only
day1_day8_data <- data %>% 
  filter(sample_time %in% c("DAY1", "DAY8"))

# Perform t-tests for each marker and treatment group
t_test_results <- day1_day8_data %>%
  group_by(marker, trt_group) %>%
  summarise(
    test = list({
      df <- pick(everything())
      # Identify unique subject IDs for each day
      subs_day1 <- unique(df$sbj_id[df$sample_time == "DAY1"])
      subs_day8 <- unique(df$sbj_id[df$sample_time == "DAY8"])
      
      # Check if the same subjects are present (after sorting)
      if (length(subs_day1) == length(subs_day8) && all(sort(subs_day1) == sort(subs_day8))) {
        # For paired t-test: sort by subject ID to ensure proper pairing
        x_vals <- df %>% 
          filter(sample_time == "DAY1") %>% 
          arrange(sbj_id) %>% 
          pull(digital_count)
        y_vals <- df %>% 
          filter(sample_time == "DAY8") %>% 
          arrange(sbj_id) %>% 
          pull(digital_count)
        
        t.test(x = x_vals, y = y_vals, paired = TRUE) # paired t-test
      } else {
        # Otherwise, use an unpaired t-test
        x_vals <- df %>% filter(sample_time == "DAY1") %>% pull(digital_count)
        y_vals <- df %>% filter(sample_time == "DAY8") %>% pull(digital_count)
        
        t.test(x = x_vals, y = y_vals, paired = FALSE) #unpaired t-test
      }
    }),
    .groups = "drop"
  ) %>%
  mutate(
    results = map(test, tidy)
  ) %>%
  unnest(results) %>%
  select(marker, trt_group,p.value,estimate,statistic)

# View the results
print(t_test_results)

# Select only significant results
significant_t_test_results<- filter(t_test_results,p.value<0.05)
selected_markers          <- significant_t_test_results$marker
selected_trt_group        <- significant_t_test_results$trt_group
# Print the results
cat("DAY1 and DAY 8 differs for the following marker(s):",selected_markers,"in the treatment group(s):",selected_trt_group,"\n")

# Write a summary statement.
cat("The difference in treatments between DAY8 and DAY1 is significantly different (P < 0.05) for marker C4 under treatment TA.\n",
    "Note,the evidence is suggestive (p<0.1) for marker TG under treatments TA and TB.\n")


### Question 3

# First let's check the homoskedasticity assumption under the turkey test

# Function to fit the model and append residuals and fitted values for a given marker
analyze_marker_resid <- function(df) {
  model <- lmer(digital_count ~ sample_time + trt_group + (1 | sbj_id), data = df)
  df %>% mutate(fitted = fitted(model), residuals = resid(model))
}

# Apply the function to each marker and combine the results
resid_data <- data %>%
  group_by(marker) %>%
  group_modify(~ analyze_marker_resid(.x)) %>%
  ungroup()

# Create the Residuals vs. Fitted plot for all markers
p <- ggplot(resid_data, aes(x = fitted, y = residuals)) +
  geom_point(color = "blue") +
  geom_hline(yintercept = 0, color = "black",linetype = "dashed") +
  facet_wrap(~ marker, scales = "free") +
  labs(title = "Residuals vs Fitted Values by Marker",
       x = "Fitted Values",
       y = "Residuals") +
  theme_minimal()

# Visualize
print(p)
cat("The constant variance assumption appears to be reasonable for some makers than the others.\nFor instance, i suspect heteroskedascity for maker C4.\n")

cat("That said, I will not use the Turkey test for inference here. \nI will use the unadjusted pvalues and report the corresponding adjusted p-values after multiple testing correction.\n")

## "C4" 

# 
##
marker_results <- data %>%
  group_by(marker) %>%
  nest() %>%
  mutate(
    # Fit the model for each marker
    model = map(data, ~ lmer(digital_count ~ sample_time + trt_group + (1 | sbj_id), data = .x)),
    
    # Extract the ANOVA table and get the p-value for the treatment effect
    anova_out = map(model, ~ anova(.x)),
    trt_p_value = map_dbl(anova_out, ~ .x["trt_group", "Pr(>F)"]),
    
    # Compute estimated marginal means for sample_time
    emms = map(model, ~ emmeans(.x, ~ sample_time)),
    
    # Get unadjusted pairwise comparisons among sample_time levels
    pairwise_unadj = map(emms, ~ contrast(.x, method = "pairwise", adjust = "none")),
    
    # Get Bonferroni-adjusted pairwise comparisons among sample_time levels
    pairwise_bonf = map(emms, ~ contrast(.x, method = "pairwise", adjust = "bonferroni")),
    
    # Extract the unadjusted p-value for the contrast between DAY22 and DAY8
    day22_day8_p_unadj = map_dbl(pairwise_unadj, ~ {
      tmp <- as.data.frame(.x) %>% filter(contrast %in% c("DAY22 - DAY8", "DAY8 - DAY22"))
      if(nrow(tmp) > 0) tmp$p.value[1] else NA_real_
    }),
    
    # Extract the Bonferroni-adjusted p-value for the contrast between DAY22 and DAY8
    day22_day8_p_bonf = map_dbl(pairwise_bonf, ~ {
      tmp <- as.data.frame(.x) %>% filter(contrast %in% c("DAY22 - DAY8", "DAY8 - DAY22"))
      if(nrow(tmp) > 0) tmp$p.value[1] else NA_real_
    })
  ) %>%
  select(marker, trt_p_value, day22_day8_p_unadj, day22_day8_p_bonf)

print(marker_results)
## In Summary
cat("## In Summary\n",
    "Here, report the unadjusted and adjusted p-values. I opted to use Bonferroni correction to control for the family wise error rate.\n",
    "The p-values for the treatment effect for all markers are >0.05; this suggests that there is not enough statistical evidence to conclude that the treatment (trt_group) has an effect on the marker digital_count at the 5% significance level.\n",
    "Similarly, we do not have enough evidence to conclude that there is a significant difference in digital_count between DAY22 and DAY8 for all markers (p>0.05). Note the evidence is suggestive (p-val < 0.1) for markers C4 & C8.\n")

