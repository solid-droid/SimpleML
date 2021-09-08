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

//create simpleML engine
const engine = new simpleML(); 

//create a model network => 'network1'
engine.createNetwork('network1',{
    input    : 3, //input neurons
    hidden   : [[4, 'reLU'], [4, 'reLU']], //[neurons , layer activation]
    output   : 2, //outputs
    activate : 'sigmoid', // output activation
}); 

//data = [{input : [1 , 2, 3] , output: [1 , 0] } , ...]

//train 'network1'
engine.train('network1', data , { 
    lr       : 0.0001,
    loss     : 'bce',
}); 

//predict the output
output = engine.predict('network1',[-5]);

```
