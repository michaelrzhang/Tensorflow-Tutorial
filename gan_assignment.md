# GAN Assignment
GAN is essentially a general method of training generative models, and it can be combined with VAE and other methods. This method uses a generative model (neural network) to map from some latent space to the feature space, and a discriminative model to try to distinguish between a real input (e.g. a real mnist digit) vs a digit generated by the generative model. Indeed, to use GAN with VAE, you just make the decoder of VAE the generative model of the GAN, and add up your losses.

Follow the code on http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/ . It should be a pretty simple introduction to GAN, and the biggest difference is that now you have a generator and discriminator with different losses. If you have more time, I suggest that you try to read https://arxiv.org/pdf/1701.00160.pdf , which goes more into detail about the probability theory.

**You guys should structure your code somewhat like the VAE assignment** in an object-oriented way so it's not all disorganized. Read about variable scope here and take advantage of the **reuse** option for variables (read https://www.tensorflow.org/programmers_guide/variable_scope carefully)! This means that your code should look something like 
```python
def GAN:
    def __init__(self, ...):
        '''
        DON'T initialize your variables in __init__, init them in the helper methods
        '''
        self.x = tf.placeholder(...)
        self.z = tf.placeholder(...)
        with tf.variable_scope('gen'):
            g_sample = _initialize_generator(self.z)
        with tf.variable_scope('dis'):
            D_real, D_logit_real = _initialize_discriminator(self.x)
        with tf.variable_scope('dis'): # this will error! Look up how to use the "reuse" parameter of the scope
            D_fake, D_logit_fake = _initialize_discriminator(g_sample)
        # follow rest of tutorial
        
    
    def _initialize_generator(z):
        '''
        initialize generator variables here. Return same as the guide's "generator" function
        '''
    
    def _initialize_discriminator(x):
        '''
        initialize discriminator variables here. Return same as the guide's "discriminator" function
        '''
    
    def fit_generate(self, ...):
        '''
        runs one batch iteration of training for generator
        '''
        
    def fit_discriminate(self, ...):
    
    def batch_generate(self, ...):
        '''
        generate one batch of output
        '''
        
    def train(self, ...):
        '''
        call fit_generate and fit_discriminate in a loop here
        '''
  
```

## Notes:
* If you look at the math a little, you should be able to see that the discriminator's loss reflects how often the discriminator classifies wrong between "real" and "fake". On the other hand, the generator's loss reflects how often the discriminator correctly predicts the generator's output to be "fake"
* When you train the GAN, use a `batch_size = 100`, and run it for something like 30 epochs. Running for 1 epoch means shuffling all of the input data `X_train` then iterate over all of the shuffled data (feed `batch_size` number of samples at once into the GAN).
* For every 5 epochs, print out the discriminator and generator loss (they show you how to do this). Also, **figure out how to get your GAN to generate some random samples** (hint: it's similar to what we did with the VAE), and generate and display using matplotlib 5 random samples with your GAN every 5 epochs of training.
* If train for enough iterations, you will see that something weird happens to the sample outputs around the 15th to 20th epoch or so. This is called "mode collapse", instead of generating random samples, your GAN converges one sample that it always generates. This is a huge problem with GAN and there's a lot of research focused on fixing it right now.
