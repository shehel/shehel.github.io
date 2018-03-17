---
layout: post
comments: true
title:  "Reconciling terminology"
date:   2017-6-30 15:30:31 +0530
categories: probability

---
<h2>Random Variables</h2>
Variables are used to represent events with varying outcomes. It can take on any outcome from its sample space. For example, a variable $X$ can represent the outcomes of a coin toss $X=H$ and $X=T$. 

<h2>Sample space</h2>
Whenever we ask about how likely an outcome is, we always ask with a set of possible outcomes in mind. Sample space is the set that exhausts all possible outcomes and the outcomes are all mutually exclusive. For example, in most realizations, a whale is not an outcome you get when you do a coin toss and hence, a whale is not in its sample space. 

<h2>Probability</h2>
It is just a way of assigning numbers to a set of mutually exclusive possibilities and they need to satisfy the following conditions 
<ul>
	<li>
Probability value must be non-negative (>= 0)</li>
	<li>Sum of the probabilitie across all events in the entire sample space must be 1, i.e., one of the events in the space must happen, otherwise the space doesn’t exhaust all possibilities.</li>
    <li>For any two mutually exclusive events, the probability that one or the other occurs is the sum of their individual probabilities. For example, the probability that a fair six-sided die comes up 3-dots or 4-dots is 1/6 + 1/6 = 2/6.</li></ul>

<h2>Probability Distributions</h2>
A list of all possible outcomes and their corresponding probabilities. 

<h2>Discrete Distributions (Probability Mass)</h2>
When the sample space consists of discrete outcomes, then we can talk about the probability of each distinct outcome. For example, the sample space of a flipped coin has two discrete outcomes, and we talk about the probability of head or tail. Note that the sum of the probability masses across the discrete outcomes must be 1. Continous distributions can be ‘discretized’ by splitting the distribution into a finite set of mutually exclusive and exhaustive ‘bins’, essentially creating a histogram
 

<h2>Continous Distributions (Probability Density Functions</h2>
Suppose you want to assign probabilities to heights. What’s is the probability of 160cm? What about 160.5, 160.0005, 160.0000000 and we could go on. It’ll soon be evident that the probability of a random variable being equal to any exact value is 0. The problem with 'discretizing' is that the interval widths and edge are arbitrary and can misrepresent information. 

Therefore, what we will do is make the intervals infinitesimally narrow, and
instead of talking about the infinitesimal probability mass of each infinitesimal interval,
we will talk about the ratio of the probability mass to the interval width. That ratio is
called the probability density. It's important to note that probability density is not equal to probabilities as we defined above. Thus, there is also nothing mysterious about probability densities larger than 1.0; it means merely that there is a high
concentration of probability mass relative to the scale. 

For any continuous value that is split up into intervals, the sum of the
probability masses of the intervals must still be 1. These notions will become concrete with the discussion on normal distribution.

<h2>Normal/Gaussian Distribution</h2>


<h2>Bayes rule from conditional probability</h2>

$$p(a \vert b) = \frac{p(a,b)}{p(b)}$$
