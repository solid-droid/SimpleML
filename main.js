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