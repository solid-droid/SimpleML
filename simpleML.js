

class simpleML {

    layers = {};
    network;
    Neuron;

    constructor(){
        this.enableNeuron();
    }

    mean = arr => arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;
    clamp = (num, min, max) => Math.min(Math.max(num, min), max);

    enableNeuron(){
        this.Neuron = class {
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
            forward(inputs){
                if(!this.error && inputs.length == this.inputCount){
                    const output = this.dot(this.weights, inputs)+this.bias;
                    return parseFloat(output.toFixed(4));

                } else {
                    console.error('shape mismatch of inputs and weights')
                }
            }

            getWeights = () => this.weights;
            setWeights = weights => this.weights = weights;
        }
    }

    createNeuron(name , {inputs = 1, layer='layer0', weights = this.getRandom() , bias = 1}, skip = false){
        weights = this.checkWeights(weights, inputs);
        const neuron = new this.Neuron(name, inputs, weights, bias);
        this.buildLayer(layer, skip);

        if(this.layers[layer]){
            this.layers[layer].push({neuron});
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
    
    createLayer(name, inputs, neurons, {weights} = {}){
        if(this.layers[name]){
            console.error('layer already exists');
            return null;
        }
        this.buildLayer(name, true);
        for(let i =0; i<neurons; ++i){
            this.createNeuron(`${name}_neuron${i}`,{inputs, layer:name, weights},true);
        }
    }

    getNetwork = _ => this.layers;
    connect = network => this.network = network;

    getRandom = _ => parseFloat((Math.random()+0.1).toFixed(3));

    checkWeights = ( weights, lengthArr ) => typeof weights == 'number' ? Array.from({length: lengthArr}, () => weights) : weights;

    runLayer(layer, inputs){
        if(this.layers[layer]){
            return this.layers[layer].map( item => item.neuron.forward(inputs))
        } else {
            console.error(`${layer} not found`);
        }
    }
//Batch methods
    runBatch_Layer = (layer, inputs) =>  inputs.map(x => this.runLayer(layer, x));
    runBatch_Relu = x => x.map(y => this.relu(y));
    runBatch_Sigmoid = (x , prec = 4) => x.map(y => this.sigmoid(y, prec));
    runBatch_SoftMax = (x , prec = 4) => x.map(y => this.softmax(y , prec));
    runBatch_loss = (input , target, lossFunc = this.loss, prec = 4) => {
        target = Array.isArray(target[0]) ? target.map(x => x.indexOf(1)) : target;
        return input.map((y,i)=> lossFunc(y[target[i]], target[i] , prec))
    }
    runBatch_accuracy = (inputs, target, prec = 4) => {
        target = Array.isArray(target[0]) ? target.map(x => x.indexOf(1)) : target;
        return parseFloat((this.mean(target.map((node, index) => inputs[index][node]))).toFixed(prec));
    }

//activation methods


    relu = x => x.map(y => Math.max(0, y)); 
    sigmoid =(x, prec = 4) => x.map(y => parseFloat((1 / (1 + Math.exp(-y))).toFixed(prec)));
    softmax = (arr, prec = 4) => {
        const maxLogit = Math.max(...arr);
        const scores = arr.map(l => Math.exp(l - maxLogit));
        const denom = scores.reduce((a, b) => a + b);
        return scores.map(s => parseFloat((s / denom).toFixed(prec)));
    }

//loss methods

    //catagorical loss entropy
    loss = (input, target , prec = 4) =>{
        if(!input && input!==0){
            console.error('Need extra output neurons')
        }
        return parseFloat((-Math.log(this.clamp(input,1e-7,1-1e-7)) * target).toFixed(prec));
    } 
//////////////////////////////////////////////////////////////////////////////////////////
// derivative methods
    derv_sigmoid =(x , prec = 4) => parseFloat((x * (1- x)).toFixed(prec));
    derv_relu = x => x > 0 ? 1 : 0;

// higher order methods

 methods = {
    'sigmoid':  this.runBatch_Sigmoid,
    'relu': this.runBatch_Relu,
    'softmax': this.runBatch_SoftMax
  };

    feedForward(input, network= this.network){
        let output = input;
        network.forEach(stage => {
            output = ['sigmoid','relu','softmax'].includes(stage) ?
                this.methods[stage](output) :  this.runBatch_Layer(stage, output)
        });
        return output;
   }

    train(input, target , network = this.network){

    }

}

