

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
var engine = new simpleML2(); 

const NN = engine.createNetwork()
NN.createLayer(100);    //inputs layer
NN.createLayer(50);     //hidden layer

console.log(NN);








