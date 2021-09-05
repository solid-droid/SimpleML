
let network = new simpleML();

network.createLayer('layer0', 4, 3);
network.createLayer('layer1', 3, 3);

input = [[1 , 2, 3, 2.5],
         [0.5 , 1.1, 3.3, 4.5],
         [2.2 , 4, 4.6, 5.5],]

output = network.runBatchLayer('layer0', input);
output = network.runBatchLayer('layer1', output);
console.log(output);