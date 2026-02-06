functions {
  real calculate_Zi(real beta, real sigma_i, real B) {
    return 1 - pow(1 - beta, sigma_i * B);
  }
}
data {
  // Dimensions of the data matrix, and matrix itself.
   int<lower=1> n;
   array[n, n] int<lower=0> M;
   int<lower=1> C;
}
transformed data {
  // Pre-compute the marginals of M to save computation in the model loop.
  array[n] int M_rows = rep_array(0, n);
  array[n] int M_cols = rep_array(0, n);
  int M_tot = 0;
  for (i in 1:n) {
    for (j in 1:n) {
      M_rows[i] += M[i, j];
      M_cols[j] += M[i, j];
      M_tot += M[i, j];
    }
  }
}
parameters {
  real<lower=0,upper=1> beta;
  array[n] real<lower=0,upper=30> k;
  real<lower=0, upper=1> rho;
  real<lower=0,upper=1> B;
}
transformed parameters {
array[n] real Z_1;

for (i in 1:n) {
  Z_1[i] = calculate_Zi(beta, k[i], B);
}

}
model {
  // Global sums and parameters
    // Global sums and parameters
  target += M_tot * log(C) ;
  // Weighted marginals of the data matrix 
  for (i in 1:n) {
    target += M_rows[i] * log(Z_1[i]);
  }
  for (j in 1:n) {
    target += M_cols[j] * log(Z_1[j]);
  }
  for (i in 1:n) {
    for (j in 1:n) {
      target += -C * Z_1[i] * Z_1[j];
    }
  }

  // Pairwise loop
  for (i in 1:n) {
    for (j in 1:n) {
      real nu_ij_0 = log(1 - rho);
      real nu_ij_1 = log(rho) + M[i,j] * log(1 + beta/Z_1[j]) -  C * beta * Z_1[i];
      if (nu_ij_0 > nu_ij_1)
        target += nu_ij_0 + log1p_exp(nu_ij_1 - nu_ij_0);
      else
        target += nu_ij_1 + log1p_exp(nu_ij_0 - nu_ij_1);
  }
 }
} 
generated quantities {
  // Posterior edge probability matrix
  array[n, n] real<lower=0> Q;
  for (i in 1:n) {
    for (j in 1:n) {
      real nu_ij_0 = log(1 - rho);
      real nu_ij_1 = log(rho) + M[i,j] * log(1 + beta/Z_1[j]) -  C * beta * Z_1[i];
      if (nu_ij_1 > 0) 
        Q[i, j] = 1 / (1+ exp(nu_ij_0 - nu_ij_1));
      else
        Q[i, j] = exp(nu_ij_1) / (exp(nu_ij_0) + exp(nu_ij_1));
    }
  }
}