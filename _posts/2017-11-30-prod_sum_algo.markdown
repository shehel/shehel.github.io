---
layout: post
comments: true
title:  "Sum Product Algorithm and Graph Factorization"
date:   2017-11-30 09:45:31 +0530
categories: Graphical Modeling

---
This post describes the Sum Product algorithm which is a neat idea that leverages a handful of important theories from across the field and extends towards more complex models. I found it to be a very nice method for accessing more complex models and it is infact a generalization of  Markov Chains, Kalman Filter, Fast Fourier Transform, Forward-Backward algorithm and more. I'm aware that the approach I use may not satisfy everyone and in that case, take a look at the reference section for some excellent resources.  

<h2>Minimum Minimorum</h2>
I assume familiarity with the 2 fundamental rules of probability, the product rule and the sum rule given below and the associated notions of joint distributions and marginals. We will be making extensive use of these.   

$$p(X,Y)=p(Y|X)p(X) \label{product} \tag{1}$$ 

$$p(X) = \sum_y p(X,Y) \label{sum} \tag{2}$$ 

<h2>What and Why - Sum Product Algorithm?</h2>
It is used for Inference, which is a frequently used word in statistics to mean marginalizing a joint distribution so we can be informed of something that was unknown given the other known variables. 
An issue with marginalizing a joint is that it quickly becomes intractable, i.e., computationally impossible due to the size of the numbers involved. For instance, say you have a model with 100 binary variables (each variable is 0 or 1) and you are now faced with marginalizing a joint distribution with $2^{100}$ terms. Here we can make a simplifying assumption that our joint distribution is not exactly a general joint but a factorized distribution where each of the factors just depend on a few local variables that are 'close' to it. This is the underlying assumption that makes fast inference through Sum Product possible.

<h2>Factorization</h2>
I have been mentioning factors and all it means here is that there's a different way of specifying joint distribution wherein the function (joint distribution) $p(x_1,...,x_n)$ factors into a product of several local functions each of which only contain a subset of the variables. 
 
$$p(x_1,...,x_n) = \prod_s f_s(X_s)$$

where 
  $X_s$ is a subset of the variables.

<h2>Factor Graphs</h2>
Factorization can be verbosely represented through factor graphs because it makes it explicit by adding additional nodes for factors. 
<div style="example">
	<pre>
A factor graph is a bipartite graph that expresses the structure<br>of the factorization.A factor graph has a variable node for each variable $x_i$,<br>a factor node for each local function $f_s$, and an edge-connecting variable node<br>$x_i$ to factor node $f_s$ if and only if $x_i$ is an argument of $f_s$.
	</pre>
</div>
 An example factorization and its corresponding graph is given below.

$$p(a, b, c, d, e) = f_1(a, b) f_2(b,c,d) f_3(d, e)$$

<figure><center>
<img src="/assets/factorised.png" alt="factorgraph" title="
Factor Graph"></center>
</figure>

<h2>Factor graphs as Expression Trees</h2>

If the factor graph doesn't contain cycles, then it can be represented as a tree and computation can be simplified using the distributive law of multiplication - \ref{distributive}. We can view this with a 'message-passing' analogy whereby the marginal variable is the 'product' of 'messages'. This idea is made clear in the next section. To convert a function representing $p(x_1,...,x_n)$ to the corresponding expression tree for $p(x_i)$, rearrange the factor graph as a rooted tree with $x_i$ as root.

$$\begin{split}
\sum_x \sum_y xy = x_1y_1+x_2y_1+x_1y_2+x_2y_2 \\
= (x_1+x_2)(y_1+y_2) 
= \sum_x x \sum_y y 
\end{split} \label{distributive} \tag{3}$$

<h2>Algorithm in action</h2>
I found that the best way to learn the algorithm was to see it in execution given the basic ingredients we have gathered so far. There are missing pieces which will be explained as it appears. 

<h3>Problem - Find the marginal</h3>
<figure><center>
<img src="/assets/workfac.png" alt="factorgraph2" title="
Factor Graph"></center>
</figure>
The factor graph describes the factorization given

$$p(a,b,c,d,e) = f_1(a)f_2(b)f_3(a,b,c)f_4(c,d)f_5(c,e) \label{cc} \tag{4}$$ 

And we want to find the marginal $p(c)$. Using \ref{sum}

$$p(c) = \sum_a \sum_b \sum_d \sum_e p(a, b, c, d, e) \label{dd} \tag{5}$$

 We can represent the factor graph as a tree and this will give us the ground to build an intuition of message passing. Notice that the tree is simply rearranged to reflect our problem of finding $p(c)$. 
<figure><center>
<img src="/assets/worktree.png" alt="factorgraphtree" title="Factor Graph as a Tree"></center>
</figure>

<h3>Message Passing</h3>
Using the message passing analogy, picture the marginal as a message comprised of several messages that were gathered along the branches of the tree. Substituting \ref{cc} in \ref{dd}, we get a form that we can start to work on

$$
p(c) = \sum_a \sum_b \sum_d \sum_e f_1(a)f_2(b)f_3(a,b,c)f_4(c,d)f_5(c,e)
$$

<ol><li>
Variable $c$ is composed of 3 messages that it received from each of its neighboring factors. To get the message of a variable node, simply multiply all the incoming messages from neighboring factor nodes.

$$p(c) = m_{f3 \rightarrow c}(c) m_{f4 \rightarrow c}(c) m_{f5 \rightarrow c}(c)$$

where $m_{x \rightarrow y}(z)$ represents a message sent from node $x$ to node $y$ which is a function of variable $z$ because the other variables have been summed out. </li>
<li>A natural question now is what the message in the factor nodes are. Let's go through the first factor. In case you are left wondering how something is the way it is, remember that everything is a combination of \ref{distributive} and \ref{sum}. To initiate messages at leaf nodes, the following rules are used depending on if its a leaf or factor node:

<ul>
	<li>$m_{x \rightarrow f}(x)=1$ a leaf variable node sends an identity function.
	</li>
	<li>$m_{f \rightarrow x}(x)=f(x)$ a leaf factor node sends a description of the function to its parent
	</li>
</ul>

The 'procedure' to evaluate a message send by a factor node:
<ol>
	<li>Take product of incoming messages into factor node. 
	</li>
	<li>Multiply by factor associated with the node</li>
	<li>Marginalize over all variables associated with incoming messages by pulling out the summations. 
	</li>
</ol>
</li>
</ol>
By recursively applying the two rules we have seen, the two incoming messages for $f_3$ 

$$m_{a\rightarrow f3}(b) = f_1(a)$$

$$m_{b \rightarrow f3}(a) = f_2(b)$$

Note that the right hand side of both the above equations can be seen as the message from the factor node since variable node simply multiply the messages of factor nodes.And what we are left with is

$$m_{f3 \rightarrow c}(c) = \sum_a f_1(a) \sum_b f_2(b) \Big[ f3(a,b,c) \Big] \label{3-c} \tag{6}$$ 

In the original paper, the authors propose a different notation for equations like above called 'not-sum' or summary notation which gives \ref{3-c} the form

$$m_{f3 \rightarrow c}(c) = \sum_{\sim c} f_1(a) f_2(b) f3(a,b,c)$$ 

<div class="example">
<pre>
Instead of indicating the variables being summed over, we indicate<br>those variables not being summed over. 
</pre>
</div>

<div style="example">
The Sum-Product Update Rule:
<pre>
The message sent from a node $v$ on an edge $c$ is the product of the<br>local function at $v$(or the unit function if $v$ is a variable node) with<br>all messages received at $v$ on edges other than $e$, summarized for<br>the variable associated with $e$.

Variable to local function

$$m_{x \to f} (x) = \prod_{h \in n(x) \backslash \{f\}} m_{h \to x} (x)$$

Local function to variable

$$m_{f \to x} (x) = \sum_{\sim \{x\}} \Big(
f(X) \prod_{y \in n(f) \backslash \{x\}} m_{y \to f} (y)
\Big)$$

</pre>
</div>

We now know everything required to complete the marginal. Applying the equations above, the final form is 

$$p(c) = \sum_{\sim c} \Big(f_1(a) f_2(b) f3(a,b,c) \Big) \sum_{\sim c}\Big(f_4(d)\Big) \sum_{\sim c}\Big(f_5(e)\Big)$$ 

<h2>Marginal for every node</h2>
Having done the work of finding a marginal for $p(x)$, and going on to think of calculating the marginal for $p(y)$ both variables in the same factor graph, one might notice redundancies because the sub computations for evaluating messages are the same. We can take advantage of this by picking any node and propagating messages from leaf to the root as shown above and from the root back to the leaf so that every node has seen two messages, caching the evaluations all along. A slight increase in computation but now we have all the marginals.    

<h2>Summary</h2>
Hopefully this post introduced Factor Graphs and Sum Product algorithm and provided an intuition of the ideas. It is essentially a method for exact inference of the marginal given a factorized distribution in an acyclical graph. 

<h2>Next Steps</h2>
I strongly recommend reading the <a href="http://www.psi.toronto.edu/pubs/2001/frey2001factor.pdf">original paper</a> as this post introduced less than half of the paper. It's very accessible with familiar examples and contains a lot more information including applications.

I also avoided talking about cyclic graphs which is where most of the interesting problems lie and the paper discusses interesting ways of working around this with algorithms like Junction Tree and Loopy Belief but you already know everything to understand them. 

Another avenue for thought is working with continuous variables, i.e., when the messages are intractable. Interesting techniques like Monte Carlo and Variational inference are used in such cases which are whole books worth of content on its own.

<a href="https://github.com/ilyakava/sumproduct">Here</a> is a nice python implementation of the algorithm for the code savvy learners. 

<h2>References</h2>
<a href="http://www.psi.toronto.edu/pubs/2001/frey2001factor.pdf">Factor Graphs and the Sum-Product Algorithm</a><br>
<a href="https://www.youtube.com/watch?v=c0AWH5UFyOk&t=3516s">Christopher Bishop's presentation video</a><br>
<a href="https://github.com/ilyakava/sumproduct">Ilya's Sum Product Python implementation</a><br>
<a href="https://lipn.univ-paris13.fr/~dovgal/about.html">Sergey Dovgal's post</a>


