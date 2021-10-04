

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


var engine = new simpleML(); 

const nn = engine.createNetwork();

nn.input(2)
  .layer(1, 'reLu')
  .layer(1, 'sigmoid')
  .layer(1, 'reLu')
  .output(1, 'sigmoid');

data = [{
  input: [0, 0],
  output: [0]
},
{
  input: [0, 1],
  output: [1]
},
{
  input: [1, 0],
  output: [1]
},
{
  input: [1, 1],
  output: [0]
}
];

nn.train(data, {
  epoch: 100,
  batch: 4,
  learningRate: 0.05,
  loss: 'mse',
  shuffle: true,
  callbacks: {
    onEpochEnd: (epoch, err) => {
      updateChart(err[0]);
    }
  }
});
myChart.update();









