# SimpleML
 Machine Learning made Simple  
 Wrapper for [DannJS](https://dannjs.org/) deep learning library
 
 Create -> Train -> Predict
 
# Goals  
* Easy to generate networks.
* Custom weights and biases.
* Distributed ML.

# How To Use
HTML
```html
<script src="simpleML.js"></script>
```
Javascript
```Javascript

var engine = new simpleML(); 

const nn = engine.createNetwork();

nn.input(1)
  .layer(2, 'reLU')
  .layer(2, 'reLU')
  .layer(2, 'reLU')
  .output(1, 'reLU');

data = [
 {input:[5] , output:[true]},
 {input:[-10] , output:[false]},
 ]

nn.train(data, {
  epoch: 10,
  batch: 100,
  learningRate: 0.001,
  loss: 'mse',
  shuffle: true,
  callbacks: {
    onEpochEnd: (epoch, err) => {
      console.log(epoch, err);
    }
  }
});


```
