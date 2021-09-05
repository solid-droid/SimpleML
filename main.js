
let network = new simpleML();

network.createLayer('layer0', 4, 3);
network.createLayer('layer1', 3, 3);


input = [[1 , 2, 3, 2.5],
         [0.5 , -1.1, 3.3, -4.5],
         [-2.2 , 4, -4.6, 5.5],]

output = network.runBatch_Layer('layer0', input);
output = network.runBatch_Relu(output);
output = network.runBatch_Layer('layer1', output);
output = network.runBatch_SoftMax(output);

console.log(output);