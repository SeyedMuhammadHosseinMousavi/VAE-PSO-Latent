# VAE-PSO-Latent
Playing wiht Latent Space of VAE in PSO
Particle Swarm Optimization (PSO) is utilized to explore and refine the latent space of a trained Variational Autoencoder (VAE). Each particle in the PSO represents a potential latent vector, which is a point in the latent space that the VAE uses to generate an image. PSO adjusts these vectors by iteratively shifting them towards the best-performing particles—those that produce images closest to a target image based on reconstruction loss. Through this process, PSO fine-tunes the latent vectors to minimize the differences between the reconstructed and the original images, effectively navigating the latent space to find the optimal encoding for specific images. The results are the reconstructed images saved during the optimization, which should ideally show high fidelity to the target images, demonstrating the VAE's ability to generate accurate reconstructions from optimized latent representations.
