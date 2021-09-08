

const nn = new simpleML();


var engine = new simpleML(); //create simpleML engine

const data = [...Array(10000).keys()].map(x => {
 const input =  parseInt(String(Math.random() * 50 - 25),10);
  return  {input : [input]  , output : [input > 0 ? true : false]}
});

engine.createNetwork('network1',{
    input    : 1,
    hidden   : [[2, 'reLU']],
    output   : 1,
    activate : 'sigmoid',
}); 

engine.train('network1', data , { 
    lr       : 0.0001,
    loss     : 'bce',
}); 

output = engine.predict('network1',[-5]);

console.log(output);








