class simpleML2 {
    
    constructor(options = {}){

    }
    createNetwork(options = {}){
        return new simpleML2.Network(options);
    }
}

simpleML2.Network = class Network {
    layers = [];
    constructor(options = {}){
    }

    createLayer(neurons){
        const layer = new simpleML2.Layer(neurons);
        this.layers.push(layer);
        return layer;
    }

}

simpleML2.Layer = class Layer {
    neurons = [];
    constructor(neurons){
        [...Array(neurons).keys()].map( _ => {
            this.neurons.push(new simpleML2.Neuron());
        }); 
    }
}

simpleML2.Neuron = class Neuron {
            inputs = 0;
            ouputs = 0;
            bias = 0;
            activation = 'relu'
            constructor(options = {}){
    
            }
    }


