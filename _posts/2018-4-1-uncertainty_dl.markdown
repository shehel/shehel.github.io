---
layout: post
comments: true
title:  "Uncertainty in Deep Neural Networks"
date:   2018-03-30 11:41:31 +0530
categories: Uncertainty Bayesian Neural Networks

---
Although our world is most probably deterministic, humans by nature operate under imperfect information and uncertainty. And it's not surprising that we thrive because our biological machinery is so good at dealing with uncertainty helping generate representations of the world that then guide our actions. Quantifying uncertainty is then a reasonable thing to look for in our computational learning models to which we are increasingly offloading decision makings tasks. See AI safety and related case studies if that's not convincing. 

 This will be an ongoing project where I shall try to explore ways of quantifying uncertainty in deep learning models but pardon me when I cut corners because the breadth of the subject is massive. I shall try to compensate this with a thorough reference section. Deep Neural Networks are the models of interest because they have proven to be the state of the art in almost any learning task one can think of. However, its amazing architectural and algorithmic progress has not been consistent with probabilistic models that comes with uncertainty information and all the other benefits of a mathematically elegant formulation. Nevertheless, Bayesian treatment of deep learning(neural network) models has been making a strong comeback since the 'Golden Era' in the early '90s. 

<h2>Classifying Uncertainty</h2>
There are different classes of uncertainty and I found that there's a lack of consensus on the terminology. The wikipedia page on [uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification#Sources_of_uncertainty) does a good job of describing some of these terms. Kendall and Gal([2017](https://arxiv.org/pdf/1703.04977.pdf)) talk about Epistemic and Aleatoric uncertainty which is sufficient for our purposes although I wish these concepts had less imaginative names.
<ul>
	<li>Aleatoric uncertainty, also referred to as the risk in a task is the uncertainty inherent in the task. For example, sensor data are noisy by nature and this can't be fixed by more data. Ofcourse one could improve the sensor and explain away the uncertainty but then it's also a perfect world.</li>
	<li>Epistemic uncertainty, also referred to as the model uncertainty which is usually the uncertainty in the weights of our models. This can be reduced by enough data(parameter uncertainty) and a model with an adequate structure(layers, non-linearities etc) which is able to capture the complexity of the data generating process.</li></ul> 

 But we can simplify our scope to focus only on the predictive probabilities and its uncertainty which encapsulates both the uncertainties above. This suffices for classification tasks we are about to examine, but may not be the case for reinforcement learning and active learning tasks([Gal](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)). Disentangling the two is a topic I leave for another post, perhaps [this](https://towardsdatascience.com/building-a-bayesian-deep-learning-classifier-ece1845bc09)?. The question this post deals with is to assess ways of making deep learning models provide uncertainty estimates. This is important because they can be the pointers to better models and consequently safe and smarter AI.  

<h2>Methodology</h2>
I'll be sticking to the MNIST dataset for the experiments and they can be reproduced by following the Jupyter notebooks [here](https://github.com/shehel/Bayesian_analysis). The quality of the uncertainty estimates will be evaluated by plotting the predictive distribution using samples from the distribution and out of the distribution using the notMNIST [dataset](https://github.com/davidflanagan/notMNIST-to-MNIST). I will be making use of the entropy of the predictive distribution to evaluate uncertainty represented by a model. We seek a model with low entropy and high accuracy on predictive distribution of MNIST and high entropy on notMNIST. We will be using a very basic CNN as shown in the figure below where outputs of each layer is labelled.  

<figure><center>
<img src="/assets/uncertainty/MNISTarch.png" alt="CNN" title="CNN"><figcaption>CNN Architecture</figcaption>
</center>
</figure>

This project already has a serious problem in that there are no benchmark or metric in my knowledge which  can be used to guide the experiments. Yes, I use two opposing datasets and work towards balancing accuracy and entropy but this approach is akin to walking in the dark. For instance, an adequately complex dataset with the ground truth, in this case like labels of a dataset, we also want some kind of information to verify the 'approximation quality'/uncertainty of the models but I guess a problem, perhaps simpler than MNIST needs to be modelled almost perfectly to get to that. I mention this because a discussion in NIPS 2016 spoke about Yarin and his thesis where he shows an experiment in which an image(a digit in MNIST) is rotated and the predictions not only vary but also with wild uncertainty estimates. This raises some fundamental questions regarding our approach, how far off are we from a good representation of information?. Nevertheless, I'll use Yarin's method for further evaluation and put the results up as model uncertainties get 'better'. Adversarial samples is another way of testing the robustness of models.  

Since this is a young field of research and my lack of expertise means that I won't be making strong conclusions from evaluating the methods, unless there's consensus in the research community.

<h2>Variational Inference with Edward</h2>
Starting with the most Bayesian option, we will be using a probabilistic platform build around Tensorflow called Edward. Edward lets us define probability distributions for variables. Realisations will be samples from the defined distributions and the library also provides algorithms for probabilistic inference. The difficulty with probabilistic models is the several intractable integrals involved in doing inference. Variational Inference(VI) is a method by which Bayesian inference can be recasted as an optimisation problem, so no more impossible integrals. 

In a nutshell, we will be approximating the true posterior $p(\mathbf{w} \vert \mathbf{X, Y})$ with a simpler distribution $q_{\theta}(\mathbf{w})$ parametrised by $\theta$, in our case it will be Gaussians. We will then minimise the Kullback-Leibler(KL), which is a method to calculate 'distance' between two ditributions divergence between the true posterior and the approximate posterior. Edward uses the mean-field VI which assumes the variational distribution over the latent variables factorizes as

$$q(w_1, w_2..w_n) = \prod_{j=1}^{n}q(w_j)$$ 

This is a handicap as now our distributions are unimodal and independent between layers which is not true in the case of neural <u>networks</u> but it helps computation and it scales to an extent. Minimising the KL divergence is still hard, so we need to move the elements around a bit and we obtain an equivalent quantity to maximise, the Evidence Lower Bound(ELBO) which is given by 

$$\text{ELBO} = \underbrace{\mathbb{E}_{q_{\theta}}[\log p(\mathbf{x} \vert \mathbf{z})]}_\text{Expected Log Likelihood} - \underbrace{ \text{KL}( q_{\theta}(\mathbf{z}) \| p(\mathbf{z})] }_\text{Prior KL}$$

Implementing a Bayesian Neural Network(BNN) is straightforward in Edward. We don't need a loss function and regularisation comes for free. However, this doesn't mean that we get back perfect models as we will soon see. To get predictions from a BNN after training, we can draw samples from the weight distributions to get a collection of models and average their predictions to get a point estimate or we could look at the distribution and evaluate the uncertainty. 

<figure><center>
<img src="/assets/uncertainty/50BNN.png" alt="50PredictiveDistributions" title="Distribution of Predictive Accruracies - 50"><figcaption>Distribution of accuracies of 50 models sampled from posterior</figcaption>
</center>
</figure>

Averaging predictions of 50 models gets us to 95% accuracy. Although our model seem to be reasonable with the test set, it's quite confident with the out of distribution samples as well. The title of the plot contains the test dataset and the mean entropy whereby 0 represents absolute certainty. How come we fall short?

<figure><center>
<img src="/assets/uncertainty/MNISTvi.png" alt="EntropyBNN" title="Predictive Distribution Entropy"><figcaption>Entropy of predictive distribution - MNIST and nMNIST</figcaption>
</center>
</figure>

Molchanov et al.([2017]()), Trippe and Turner([2018](http://approximateinference.org/2017/accepted/TrippeTurner2017.pdf)) provide compelling evidence of overpruning in mean field variational family. In short, the variability is squashed and the network becomes unable to model uncertainty, therefore variability is traded for sparsity. Trippe and Turner hypothesise that this is due to an unobvious consequence of the ELBO, mean field approximation and balancing modelling the complexity of the data and retaining the simplicity of the prior under this assumption. Note that our approximating distribution which in our case are Gaussians may also not necessarily be anything similar to the true posterior. 

Training on 55,000 images in batches of 128 for 30 epochs took about 14 mins on a 1080Ti, considering it is a minimal CNN, there will be scalability issues with larger CNNs. 

<h2>Monte Carlo(MC) Dropout</h2>
Dropout is an empirical technique used to avoid overfitting in neural networks. It does this by randomly switching off a hidden unit and their connection with a probability to prevent hidden units from co-adapting too much. Gal and Gahrmani([2015](http://proceedings.mlr.press/v48/gal16.html)) shows that approximating ELBO and neural networks with dropout are identical. Therefore, optimising any neural network with dropout resembles some manifestation of approximate Bayesian inference.   

In practice, this means that any neural network can be made to resemble Bayesian inference and this allows us to get predictive mean and uncertainty. This is done by applying dropout during test time as well as training in all layers of the network. Different forward passes will effectively give samples from different models and we can build a predictive distribution from these samples. Below is a plot representing the quality of uncertainty with a model identical to the one we used for VI. 

<figure><center>
<img src="/assets/uncertainty/MCDrop.png" alt="MCDrop" title="Simple Transfer">
<figcaption>97.06% accuracy with 0.5 $p$ and 1e-5 L2</figcaption></center>
</figure>

The results are slightly better than VI. A possible explanation is the absence of the mean field approximation and we may also be sampling from richer posteriors of weights. But in that case, one can expect much better results. The authors in the MC dropout paper talks about the determinants of predictive uncertainty - model structure, model prior and approximating distribution. They identify that the predictive uncertainty depends heavily on the non-linearity and model prior length-scale(equivalent to weight decay). This was proven true when I used ReLu and adjusted the L2 regularisation and dropout probability $p$ and got a somewhat desirable result with 99.3% accuracy(uncertainty plotted below). However that was more ad hocery than I'm comfortable with as techniques to 'learn' these hyper parameters should in theory only lead to overfitting and over-confidence([Louizos and Welling(2017)](https://arxiv.org/pdf/1703.01961)). The hyperparameters(from trial and error) for the best result also contradict the fact that 0.5 for $p$ provides the most variance in weights which will translate into greater variance in the predictive distribution ([Srivastava et al.(2014)](https://www.cs.toronto.edu/%7Ehinton/absps/JMLRdropout.pdf)). Most of the models in use are ones that are tuned for accuracy and may not perform as well if we ask it for uncertainty estimates. Thus, its another step for the user to search for a model that not only gives good accuracy but also appropriate uncertainty estimates. 

<figure><center>
<img src="/assets/uncertainty/MCDropB.png" alt="MCDropB" title="High accuracy(.99) and uncertainty">
<figcaption>High accuracy(99.2%)/confidence on MNIST and low confidence on nMNIST</figcaption></center>
</figure>

Note that this is in no way a criticism of MC dropout, it's simply pointing out gaps in my individual or our collective knowledge. Nevertheless, its also worth pointing out at an actual criticism of the method in a [note](http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf) by Ian Osband which raises a confusion between risk and model uncertainty and posits that MC dropout is actually approximating risk which if it is the case, shouldn't be the uncertainty we should be putting our money on. I feel that this is a valid question and it requires more research hours especially into the theoretical aspects to come to a conclusion.

<h2>Coming up</h2>
There are several other exciting methods to uncertainty estimation which includes propositions from the Bayesian paradigm such as Normalizing Flows, Hamiltonian Monte Carlo as well as frequentist methods such as bootstrap sampling, ensembles and more. I'll update the post irregularly as I get around to exploring the related papers. Given below is a an empirical CDF of the entropy of predictive distributions as done in Louizos and Welling(2017). This should provide a good comparative picture of the methods implemented so far. Curves that are closer to the bottom right part of the plot are preferable for nMNIST, as it denotes that the probability of observing a high confidence prediction is low and the opposite for MNIST.

<figure><center>
<img src="/assets/uncertainty/cdf.png" alt="CDF" title="Empirical CDF of Entropy">
<figcaption>Empirical CDF of entropy in nMNIST(L) and MNIST(R)</figcaption></center>
</figure>

<h2>References</h2>
-[Tensorflow MNIST tutorial](https://www.tensorflow.org/versions/r1.1/get_started/mnist/pros#train_and_evaluate_the_model)<br>
-[The Bayesian brain: the role of
uncertainty in neural coding and
computation, Knill and Pouget, 2004](http://www.wcas.northwestern.edu/nescan/knill.pdf)<br>
-[MLP in Edward Tutorial](https://www.alpha-i.co/blog/MNIST-for-ML-beginners-The-Bayesian-Way.html)
