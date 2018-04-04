---
layout: page
title : About
permalink: /about/
---

<h2>Hello!</h2>

Computer Science graduate with an interest in NLP and Bayesian methods. Hoping to apply these to improve knowledge access and transfer some day.  

Trying to understand this Strange Loop  
<div class="example">
    <pre>
while alive:
  prior = posterior(xp, prior)
  xp = sense()
    </pre>
</div>

<script type="text/javascript">
	(() => {
          // Create Canvas
          const myCanvas = document.createElement('canvas');
          myCanvas.width = 400;
          myCanvas.height = 400;
          document.body.appendChild(myCanvas);
          const ctx = myCanvas.getContext('2d');
          // Start drawing
          function checkIfBelongsToMandelbrotSet(x,y) {
            let realComponentOfResult = x;
            let imaginaryComponentOfResult = y;
            // Set max number of iterations
            const maxIterations = 100;
            for (let i = 0; i < maxIterations; i++) {
              const tempRealComponent = realComponentOfResult * realComponentOfResult - imaginaryComponentOfResult * imaginaryComponentOfResult + x;
              const tempImaginaryComponent = 2.0 * realComponentOfResult * imaginaryComponentOfResult + y;
              realComponentOfResult = tempRealComponent;
              imaginaryComponentOfResult = tempImaginaryComponent;
              // Return a number as a percentage
              if (realComponentOfResult * imaginaryComponentOfResult > 5) {
               return (i / maxIterations * 100);
              }
            }
            // Return zero if in set
            return 0;
          }
          // Set appearance settings
          const magnificationFactor = 600;
          const panX = 0.8;
          const panY = 0.4;
          for (let x = 0; x < myCanvas.width; x++) {
            for (let y = 0; y < myCanvas.height; y++) {
              const belongsToSet = checkIfBelongsToMandelbrotSet(x / magnificationFactor - panX, y / magnificationFactor - panY);
              if (belongsToSet === 0) {
                ctx.fillStyle = '#000';
                // Draw a black pixel
                ctx.fillRect(x,y, 1,1);
              } else {
                ctx.fillStyle = `hsl(0, 100%, ${belongsToSet}%)`;
                // Draw a colorful pixel
                ctx.fillRect(x,y, 1,1);
              }
            }
          }
      })();
</script>