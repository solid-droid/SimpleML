////////////////////////---evolutionary---////////////////////////////////////////////

geneticAlgo();
function geneticAlgo(){
  console.log('------evolution/genetic algo------');
  const brain = new simpleML.evolutionaryBrain();

  brain.network
    .input(2)
    .layer(2, 'sigmoid')
    .layer(1, 'sigmoid')

  brain.createGeneration(100);

  for(let i=0; i<100; i++){
  
    const outputs = brain.mutate(0.1)
                         .think([[0,0],[0,1],[1,0],[1,1]])
                         .getOutputs();
  // based on some condition on output setScore(<id> , <score>)
      setScore(outputs, brain);

  //calculate fitness -> evolve -> cleanup memory
      brain.calculateFitness()
           .evolve(0.7)
    
    console.log(brain.predict([[0,0],[1,1],[0,1],[1,0]],{round:true}));
  }
}

function setScore(outputs , brain){
  outputs.forEach(child => {
    let score = 0;
    child.output.forEach((value,i)=>{
      switch(i){
        case 0: value < 0.5 ? score+=value : score-=value; break;
        case 1: value > 0.5 ? score+=value : score-=value; break;
        case 2: value > 0.5 ? score+=value : score-=value; break;
        case 3: value < 0.5 ? score+=value : score-=value; break;
      }       
    });
    brain.setScore(child.id, score);
  }); 
}

//////////////////////---backpropogation---//////////////////////////////////////////

// backPropogate();

function backPropogate(){
  console.log('------BackPropogate------');
  const nn = new simpleML.network();

  nn.input(2)
    .layer(2, 'sigmoid')
    .layer(1, 'sigmoid')
  
  
  nn.train(xor, {
    epochs: 10,
    learningRate: 0.1,
    optimizer: 'adam',
    onEpochEnd : (epoch, logs) => console.log(epoch, logs),
    onTrainEnd : () => predict()
  });
  
  const predict = () => {
    const res = nn.predict([[0,0],[1,1],[0,1],[1,0]],{
      round: true
    });
    console.log(res);
  };
}



