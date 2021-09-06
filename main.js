
let network = new simpleML();


input = [[1 , 2, 3, 2.5],
         [0.5 , -1.1, 3.3, -4.5],
         [-2.2 , 4, -4.6, 5.5],]

target = [1, 2 , 2];

network.createLayer('layer0', 4, 3 ,
// {weights: 0.7}
);
network.createLayer('layer1', 3, 3, 
// {weights: 0.5}
);

output = network.feedForward(input, ['layer0', 'relu', 'layer1', 'softmax'] );

acc    = network.runBatch_accuracy(output, target);
loss   = network.runBatch_loss(output, target);

console.log(acc, loss);






