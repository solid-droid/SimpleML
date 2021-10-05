# SimpleML
 Machine Learning made Simple  
 Wrapper for [tensorflowjs](https://www.tensorflow.org/js)
 
 Create -> Train -> Predict
 
# Goals  
* Easy to generate networks.
* Custom weights and biases.
* Distributed ML.

# How To Use
HTML
```html
<script src="./ts.min.js"></script>
<script src="./simpleML.js"></script>
```
Javascript
```Javascript

const nn = new simpleML();

//create network
nn.input(2)
  .layer(2, 'sigmoid')
  .layer(1, 'sigmoid')

//dataset
const xor = [
    { input:[0,0], output:[0]  },
    { input:[0,1], output:[1]  },
    { input:[1,0], output:[1]  },
    { input:[1,1], output:[0]  },  
]

//train network
nn.train(xor, {
  epochs: 1500,
  learningRate: 0.01,
  optimizer: 'adam',
  onEpochEnd : (epoch, logs) => console.log(epoch, logs),
  onTrainEnd : () => predict()
});

//predict
const predict = () => {
  const res = nn.predict([[0,0],[1,1],[0,1],[1,0]], {
    precision: 3,
    round: true
  });

  console.log(res)
  //[0, 0, 1, 1]
};




```
