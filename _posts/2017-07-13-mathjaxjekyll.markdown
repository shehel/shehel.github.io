---
layout: post
comments: true
title:  "MathJax with Jekyll"
date:   2017-07-13 01:12:31 +0530
categories: MathJax with jekyll
---
I use <a href="https://github.com/hemangsk/Gravity"> this </a>simple jekyll theme by hemangsk. Amongst several methods floating in the internet, here is what worked for me with kramdown to get latex running with jekyll pages.

Add to `_layouts/post.html`

{% highlight html %}
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
{% endhighlight %}

Add to `_includes/head.html` between the `head` tags

{% highlight html %}
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea'],
        inlineMath: [['$','$']]
      }
    });
</script>
{% endhighlight %}

`skipTags` specify blocks where latex will be disabled. Code between `$$` will be inline.