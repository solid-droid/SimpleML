

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

const data1 = [...Array(1000).keys()].map(x => {
    const input =  parseInt(String(Math.random() * 100 - 50),10)
     return  {input : [input]  , output : [input > 0 ? true : false]}
   });

//create simpleML engine ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////


var engine = new simpleML(); 

const nn = engine.createNetwork();

nn.input(1)
  .layer(2, 'reLU')
  .layer(2, 'reLU')
  .layer(2, 'reLU')
  .output(1, 'reLU');


nn.train(data1, {
  epoch: 10,
  batch: 100,
  learningRate: 0.001,
  loss: 'mse',
  shuffle: true,
  callbacks: {
    onEpochEnd: (epoch, err) => {
      updateChart(err[0]);
    }
  }
});
myChart.update();









