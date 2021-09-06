

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
            forward(inputs){
                if(!this.error && inputs.length == this.inputCount){
                    const output = this.dot(this.weights, inputs)+this.bias;
                    return parseFloat(output.toFixed(3));

                } else {
                    console.error('shape mismatch of inputs and weights')
                }
            }

            getWeights = () => this.weights;
            setWeights = weights => this.weights = weights;
        }
    }

    createNeuron(name , {inputs = 1, layer='layer0', weights = parseFloat((Math.random()+0.1).toFixed(3)), bias = 1}, skip = false){
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

    getNetwork = () => this.layers;

    runLayer(layer, inputs){
        const outputs = [];
        if(this.layers[layer]){
            this.layers[layer].forEach((item , i) => {
                if(item.type == 'neuron'){
                    this.layers[layer][i].output = item.neuron.forward(inputs);
                    outputs.push(this.layers[layer][i].output);
                }
            });;
            return outputs;
        } else {
            console.error(`${layer} not found`);
        }
    }
//Batch methods
    runBatch_Layer(layer, inputs){
        const outputs = [];
        inputs.forEach(x => {
            outputs.push(this.runLayer(layer, x));
        });
        return outputs;
    }
    runBatch_Relu = x => x.map(y => this.relu(y));
    runBatch_Sigmoid = (x , prec = 4) => x.map(y => this.sigmoid(y, prec));
    runBatch_SoftMax = (x , prec = 4) => x.map(y => this.softmax(y , prec));
    runBatch_loss = (input , target, lossFunc = this.loss, prec = 4) => {
        target = Array.isArray(target[0]) ? target.map(x => x.indexOf(1)) : target;
        return input.map((y,i)=> lossFunc(y[target[i]], target[i] , prec))
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
    clamp = (num, min, max) => Math.min(Math.max(num, min), max);
    //catagorical loss entropy
    loss = (input, target , prec = 4) =>{
        if(!input && input!==0){
            console.error('Need extra output neurons')
        }
        return parseFloat((-Math.log(this.clamp(input,1e-7,1-1e-7)) * target).toFixed(prec));
    } 
//////////////////////////////////////////////////////////////////////////////////////////
    mean = arr => arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;

    runBatch_accuracy = (inputs, target, prec = 4) => {
        target = Array.isArray(target[0]) ? target.map(x => x.indexOf(1)) : target;
        return parseFloat((this.mean(target.map((node, index) => inputs[index][node]))).toFixed(prec));
    }


///////////////////////////////////////////////////////////////////////////////////////////
// derivative methods
    derv_sigmoid =(x , prec = 4) => parseFloat((x * (1- x)).toFixed(prec));
    derv_relu = x => x > 0 ? 1 : 0;

// higher order methods

 methods = {
    'sigmoid':  this.runBatch_Sigmoid,
    'relu': this.runBatch_Relu,
    'softmax': this.runBatch_SoftMax
  };

    feedForward(input, network){
        let output = input;
        network.forEach(stage => {
            output = ['sigmoid','relu','softmax'].includes(stage) ?
                this.methods[stage](output) :  this.runBatch_Layer(stage, output)
        });
        return output;
   }


}

