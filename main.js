///////////////////////--chart--/////////////////////////////
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

  ////////////////////////////////////////////////////////////////

const nn = new simpleML();

nn.input(2)
  .layer(2, 'sigmoid')
  .layer(1, 'sigmoid')

nn.train(xor, {
  epochs: 1500,
  learningRate: 0.01,
  optimizer: 'adam',
  onEpochEnd : (epoch, logs) => console.log(epoch, logs),
  onTrainEnd : () => predict()
});

const predict = () => {
  const res = nn.predict([[0,0],[1,1],[0,1],[1,0]], {
    precision: 3,
    round: true
  });

  console.log(res)
};