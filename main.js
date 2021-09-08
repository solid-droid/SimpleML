

const data = [...Array(10000).keys()].map(x => {
    const input =  parseInt(String(Math.random() * 50 - 25),10);
     return  {input : [input]  , output : [input > 0 ? true : false]}
   });

//create simpleML engine
const engine = new simpleML(); 


//create a model network
engine.createNetwork('network1',{
    input    : 1,
    hidden   : [[2, 'reLU']],
    output   : 1,
    activate : 'sigmoid',
}); 


//train it
engine.train('network1', data , { 
    lr       : 0.0001,
    loss     : 'bce',
}); 

//predict the output
output = engine.predict('network1',[-5]);

console.log(output);








