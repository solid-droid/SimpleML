

class simpleML {

    neurons = [];
    layers = {};
    classNeuron;

    constructor(){
        this.initNeuron();
    }
    initNeuron(){
        this.classNeuron = class {
            inputCount;
            name;
            weights = [];
            output;
            bias = 0;
            error = false;
            constructor(name, inputs, weights, bias){
                this.inputCount = inputs;
                 if(weights.length!==inputs){
                    this.error = true;
                    console.warn('weights should have the same number of inputs')
                }
                this.weights = weights;
                this.name = name;
                this.bias = bias;
            }

            dot = (a, b) => a.map((x, i) => a[i] * b[i]).reduce((m, n) => m + n);
            execute(inputs){
                if(!this.error && inputs.length == this.inputCount){
                    const output = this.dot(this.weights, inputs)+this.bias;
                    return parseFloat(output.toFixed(3));

                } else {
                    console.error('shape mismatch of inputs and weights')
                }
            }
        }
    }

    createNeuron(name , {inputs = 1, layer='layer0', weights = parseFloat((Math.random()).toFixed(3)), bias = 1} = 
        {inputs: 1, layer: 'layer0', weights: parseFloat((Math.random()).toFixed(3)), bias : 1}, skip = false){
        if(typeof weights == 'number'){
            weights = Array.from({length: inputs}, () => weights);
        }
        const neuron = new this.classNeuron(name, inputs, weights, bias);
        this.neurons.push({name, layer, inputs, neuron } );
        this.buildLayer(layer, skip);
        if(this.layers[layer]){
            this.layers[layer].push({type:'neuron', neuron , output: null});
        }
    }

    buildLayer(name, skip = false) {
        if(!this.layers[name]){
            this.layers[name]=[];
        } else {
            if(!skip){
                console.error('layer already exists');
            }
        }
    }
    
    createLayer(name, inputs, neurons){
        if(this.layers[name]){
            console.error('layer already exists');
            return null;
        }
        this.buildLayer(name, true);
        for(let i =0; i<neurons; ++i){
            this.createNeuron(`${name}_neuron${i}`,{inputs, layer:name},true);
        }
    }

    getNetwork = () => this.layers;

    runLayer(layer, inputs){
        const outputs = [];
        if(this.layers[layer]){
            this.layers[layer].forEach((item , i) => {
                if(item.type == 'neuron'){
                    this.layers[layer][i].output = item.neuron.execute(inputs);
                    outputs.push(this.layers[layer][i].output);
                }
            });;
            return outputs;
        } else {
            console.error(`${layer} not found`);
        }
    }

    runBatchLayer(layer, inputs){
        const outputs = [];
        inputs.forEach(x => {
            outputs.push(this.runLayer(layer, x));
        });
        return outputs;
    }

    ////activation functions
    relu = x => x.map(y => Math.max(0, y)); 
    sigmoid = x => x.map(y => 1 / (1 + Math.exp(-y)));

}

