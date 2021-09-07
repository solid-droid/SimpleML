# SimpleML
 Machine Learning made Simple  
 Wrapper for [DannJS](https://dannjs.org/) deep learning library
 
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

const engine = new simpleML(); //create simpleML engine

input = [1 , 2, 3, 2.5];

engine.createLayer('inputs', 4); 
// length of inputs

engine.createLayer('layer1', 3); 
// layer name: 'layer 1' , number of neurons: 3

engine.createLayer('layer2', 3);

engine.createNetwork('network1', ['inputs', 'layer0', 'relu', 'layer1', 'softmax'] ); 
// create a network

const output = engine.predict(inputs, 'network1'); 
// predict over the network

```
