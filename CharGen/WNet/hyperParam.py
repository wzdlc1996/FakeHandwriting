seed = 999  # Is the global rng seed


# Hyper-parameters for loss
alpha = 3
alpha_GP = 10
beta_d = 1
beta_p = beta_r = 0.2
lamb_l1 = 50
lamb_phi = 75
psi_p = 3
psi_r = 5


# Hyper-parameters for labeling
fake_lab = 0.
real_lab = 1.


# Hyper-parameters for optimization
adamBeta = (0.5, 0.999)
iniLr = 0.0005
lrDecay = 0.99


# Hyper-parameters for modeling
dropR = 0.5


# Hyper-parameters for training script
MAXEPOCH = 20
sepr = 10
batch_size = 1
num_workers = 1