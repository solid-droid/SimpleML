

  const config = {
    type: 'line',
    data: {
            labels : [],
            datasets:[{
                label:'loss',
                data:[],
                borderColor: 'orange',
                borderWidth: 1,
                tension: 0.5
            }]
        },
    options: {}
  };

  const myChart = new Chart(
    document.getElementById('myChart'),
    config
  );

const loss = [];
  const updateChart = data => {
    myChart.data.labels.push(String(myChart.data.labels.length));
    myChart.data.datasets[0].data.push(data);
}


////////////////////////////////////////////////////////

const data1 = [...Array(1000).keys()].map(x => {
    const input =  parseInt(String(Math.random() * 100 - 50),10)
     return  {input : [input]  , output : [input > 0 ? true : false]}
   });

//create simpleML engine
var engine = new simpleML(); 


//create a model network
engine.createNetwork('network1',{
    input    : 1,
    hidden   : [[2, 'reLU'],],
    output   : [1,'sigmoid']
}); 


//train it
engine.train('network1', data1 , { 
    lr         : 0.01,
    loss       : 'mce',
    optimizer  : 'adam',
    epoch      : 1,
    batch      : 1,
    getloss    : err => updateChart(err),
    onComplete : _ =>   predict(),
}); 

//predict the output

function predict() {
    myChart.update();
    output = engine.predict('network1',[-5]);
    console.log(output);
}







