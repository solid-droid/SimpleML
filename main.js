

const input = [[1 , 2, 3, 2.5],
         [0.5 , -1.1, 3.3, -4.5],
         [-2.2 , 4, -4.6, 5.5],]

const target = [1, 2 , 2];

const network = new simpleML();
network.createLayer('layer0', 4, 3); //name , inputs, neurons
network.createLayer('layer1', 3, 3);
network.connect(['layer0', 'relu', 'layer1', 'softmax']);

output = network.feedForward(input );

network.train(input, target);

acc    = network.runBatch_accuracy(output, target);
loss   = network.runBatch_loss(output, target);

console.log(acc);






