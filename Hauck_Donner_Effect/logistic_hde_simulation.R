
#02-12-2021

# This script simulate the Hauck Donner Effect that arises in association studies

all.stat <- all.pi.1 <- all.pi.0 <- all.beta0 <- loglike <- waldstat <- permstat <- NULL

N <- 50

for (i in 1:(N - 1)) {
  Nx1 <- Nx0 <- 50
  R0  <- 25;  mypi.0 <- R0 / Nx0   # PI.0 fixed at 1/2
  R1  <- i;   mypi.1 <- i  / Nx1   # PI.1 varies across iterations

  N_R0  <- Nx0 - R0
  N_R1  <- Nx1 - R1
  mymat <- matrix(c(N_R0, N_R1, R0, R1), nrow = 2)

  N_R0 <- mymat[1, 1]; N_R1 <- mymat[2, 1]
  R0   <- mymat[1, 2]; R1   <- mymat[2, 2]

  my.y <- c(rep(1, R0 + R1),   rep(0, N_R0 + N_R1))
  my.x <- c(rep(0, R0), rep(1, R1), rep(0, N_R0), rep(1, N_R1))

  myfit  <- glm(my.y ~ my.x, family = "binomial")
  stats  <- summary(myfit)
  nfit   <- anova(myfit, test = "Chisq")

  loglike  <- c(loglike,  nfit[2, 5])
  waldstat <- c(waldstat, stats$coeff[2, 4])

  # Permutation test: compare observed beta^2 against 20 permuted betas
  allbetas <- stats$coeff[2, 1]
  for (k in 1:20) {
    myfet    <- glm(sample(my.y) ~ my.x, family = "binomial")
    allbetas <- c(allbetas, summary(myfet)$coeff[2, 1])
  }
  gvec     <- allbetas^2
  permstat <- c(permstat, sum(gvec >= gvec[1]) / length(gvec))

  all.pi.1 <- c(all.pi.1, mypi.1 - mypi.0)
  all.pi.0 <- c(all.pi.0, mypi.0)
}

dir.create("figures", showWarnings = FALSE)

png("figures/logistic_regression_HDE.png", width = 600, height = 700)
plot(all.pi.1, -log10(waldstat),
     yaxt = "none", type = "l",
     xlab = "Difference in Proportions",
     ylab = "-log(p-values)",
     ylim = c(0, 6),
     main = "Logistic Regression & HDE")
axis(2, seq(0, 6, 2))
lines(all.pi.1, -log10(loglike), lty = 2)
legend("top", legend = c("Wald Test", "LRT"), lty = c(1, 2), cex = 0.7)
dev.off()
