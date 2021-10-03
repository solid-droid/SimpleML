

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
    console.log(data);
    myChart.data.labels.push(String(myChart.data.labels.length));
    myChart.data.datasets[0].data.push(data);
}

const data1 = [...Array(10).keys()].map(x => {
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

const data = [{input:[11], output:[true]}, {input:[-5], output:[false]}];

nn.train(data, {
  epoch: 50,
  batch:1,
  learningRate: 0.001,
  loss: 'mse',
  shuffle: true,
  callbacks: {
    onEpochEnd: (epoch, err) => {
      // loss.push(err[0]);
      // updateChart(err[0]);
    }
  }
});

updateChart(10);
updateChart(15);
updateChart(20);










