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
    input    : 1,
    hidden   : [[2, 'reLU'],],
    output   : [1,'sigmoid']
}); 

//data = [{input : [1 , 2, 3] , output: [1 , 0] } , ...]

//train 'network1'
engine.train('network1', data1 , { 
    lr         : 0.01,
    loss       : 'mce',  //only readly at the moment (not applied to network as hyper param)
    optimizer  : 'adam', //not implemented
    epoch      : 1,      //not implemented
    batch      : 1,      //not implemented
    getloss    : err => updateChart(err),
    onComplete : _ =>   predict(),
}); 

//predict the output
function predict() {
    //loss func
    myChart.update();
    
    //prediction
    output = engine.predict('network1',[-5]);
    console.log(output);
}

```
