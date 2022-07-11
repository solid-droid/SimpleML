class simpleML {

    optimizers = {
        adam : (lr) => tf.train.adam(lr),
        sgd : (lr) => tf.train.sgd(lr),
        rmsprop : (lr) => tf.train.rmsprop(lr),
        adagrad : (lr) => tf.train.adagrad(lr),
        adadelta : (lr) => tf.train.adadelta(lr),
        adamax : (lr) => tf.train.adamax(lr),
        gradientdescent : (lr) => tf.train.gradientDescent(lr),
        momentum : (lr) => tf.train.momentum(lr),
        nesterov : (lr) => tf.train.nesterov(lr),
    }

    predict(model, input , options = {}){
        const prec = options.precision || 3;
        const round = options.round || false;
        return Array.from(tf.tidy(() => model.predict(tf.tensor2d(input)).dataSync()))
                    .map(item => round ? Math.round(item) : parseFloat(item.toFixed(prec)));
    }

    cloneNetwork(network, randomWeights = false) {
       return tf.tidy(() => {
            const nn = new simpleML.network(network.networkArch);
            if(!randomWeights){
                const weights = network.model.getWeights();
                const weightCopies = [];
                weights.forEach(weight => weightCopies.push(weight.clone()));
                nn.model.setWeights(weightCopies);
            }
            return nn;
        });

    }

    mutate(model, mutationRate = 0.1){
        return tf.tidy(()=>{
             const weights = model.getWeights();
             const mutatedWeights = [];
             weights.forEach(weight => {
                 let tensor = weight.clone();
                 let shape = tensor.shape;
                 let values = tensor.dataSync().slice();
                 for(let i = 0; i < values.length; i++){
                    if(Math.random() < mutationRate){
                         values[i] = values[i] + (Math.random() - 0.5) * 0.1;
                    }
                 }
                 mutatedWeights.push(tf.tensor(values, shape));
             });
             model.setWeights(mutatedWeights);
             return model;
         })
     }

     crossover(networks){
        return tf.tidy(()=>{
            const _networks = networks.slice();
            const mutatedWeights = [];
            const shapeList = [];
            _networks.forEach(parent => {
                const model = parent.network.model;
                const fitness = parent.fitness;
                const weights = model.getWeights();
                weights.forEach((layer,k) => {
                        let tensor = layer.clone();
                        shapeList[k] ??= tensor.shape;
                        let values = tensor.dataSync().slice();
                        for(let i = 0; i < values.length; i++){
                            values[i] = values[i] * fitness;
                            mutatedWeights[k] ??= [];
                            mutatedWeights[k][i] ??= 0;
                            mutatedWeights[k][i] += values[i];
                        }
                });
            });
            return mutatedWeights.map((layerWeights,k) =>tf.tensor(layerWeights, shapeList[k]));
        });
     }

    static showMemory(){
        console.log(tf.memory());
        return tf.memory();
    }

    static summary(network){
        console.log(network.model.summary());
        return network.model.summary();
    }

    constructor() {}

}

simpleML.network = class network extends simpleML{
    type = 'network';
    constructor(networkArch = null) {
        super();
        this.model = tf.sequential();
        this.networkArch = [];
        networkArch?.forEach((layer,i) => {
            if(i==0){
                this.input(layer.nodes);
            } else {
                this.layer(layer.nodes, layer.activation);
            }
        });
    }

    input(nodes) {
        if(this.networkArch.length == 0){
            this.networkArch.push({nodes});
        }
        return this;
    }
    
    layer(nodes, activation = 'sigmoid') {
        this.model.add(tf.layers.dense({
            inputShape: this.networkArch[this.networkArch.length-1].nodes,
            units: nodes,
            activation,
          }));
        this.networkArch.push({nodes, activation});
        return this;
    }

    train(data,options = {}) {
        const epoch = options.epochs || 1;
        const batch = options.batch || 1;
        const optimizer = options.optimizer || 'adam';
        const learningRate = options.learningRate || 0.01;
        const loss = options.loss || 'meanSquaredError';
        const metrics = options.metrics || 'accuracy';
        const onTrainEnd = options.onTrainEnd || function() {};
        const onEpochEnd = options.onEpochEnd || function() {};
        this.model.compile({
            loss: loss,
            optimizer: this.optimizers[optimizer](learningRate),
            metrics: [metrics]
          });

        this.model.fit(
            tf.tensor2d(data.map(item => item.input)),
            tf.tensor2d(data.map(item => item.output)) , 
            {
                    epochs: epoch,
                    batchSize: batch,
                    shuffle: true,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => onEpochEnd(epoch, logs),
                        onTrainEnd: (logs) => onTrainEnd(logs),
                    },
            }
        );

    }

    predict(input , options = {}) {
        return super.predict(this.model, input , options);
    }

    getWeights(){
        return this.model.getWeights().map(weight => weight.dataSync().slice());
    }
    summary(){
        console.log(this.model.summary());
        return this.model.summary();
    }

    dispose(){
        this.model.layers.forEach(l => l.dispose())
    }


}

simpleML.evolutionaryBrain = class evolutionaryBrain extends simpleML.network {
    generations = [];
    generationCount = 0;
    type = 'evolutionaryBrain';
    constructor(options = {}) {
        super();
        this.childCount = options.childCount || 1;
        this.mutateRate = options.mutateRate || 0.1;
        this.randomWeights = options.randomWeights || true;
        if(options.network){
            this.network =  options.network;
        } else {
            this.network = new simpleML.network();
        }
    }

    createGeneration(childCount = this.childCount, options = {}){
        this.childCount = childCount || this.childCount;
        this.mutateRate = options.mutateRate || this.mutateRate;
        this.randomWeights = options.randomWeights || this.randomWeights;
        this.generations.push({
            generationId: ++this.generationCount,
            children: this.createChildren(this.childCount),
            mutateRate: this.mutateRate,
            childCount: this.childCount,
            disposed:false
        });
        return this;
    }

    createChildren(count) {
            const children = [];
            for(let i = 0; i < count; i++){
                children.push({
                    network:this.cloneNetwork(this.network , this.randomWeights),
                    output:[],
                    id:i,
                    fitness:0,
                    score:0,
                });
            }
            if(this.randomWeights) this.randomWeights = false;
            return children;
    }

    calculateFitness() {
        const children = this.generations[this.generations.length - 1].children;
        const totalScore = children.reduce((total, child) => total+child.score, 0) || 1;
        children.forEach(child => child.fitness = child.score / totalScore);
        return this;
    }
    
    think(input) {
        this.generations[this.generations.length - 1].children.forEach(child => {
            child.output = child.network.predict(input);
        });
        return this;
    }

    mutate(mutationRate){
        this.mutateRate = mutationRate || this.mutateRate;
        this.generations[this.generations.length - 1].children.forEach(child => {
            child.network.model = super.mutate(child.network.model, this.mutateRate);
        });
        return this;
    }

    getOutputs(){
        return this.generations[this.generations.length - 1].children.map(child => ({id:child.id , output:child.output}));
    }
    getLastGeneration(){
        return this.generations[this.generations.length - 1];
    }
    showMemory(){
        simpleML.showMemory();
        return this;
    }
    cleanup(force = true){
        this.generations.slice(0, -1).forEach(generation => {
            if(!generation.disposed){
                generation.children.forEach((child,i) => {
                    child.network.dispose();
                    generation.disposed = true;
                });
            }
        });
        if(force){
            this.generations = this.generations.splice(-1);
        }
        return this;
    }

    predict(input , options = {}) {
        return this.network.predict(input , options);
    }

    evolve() {
        tf.tidy(()=>{
            const children = this.generations[this.generations.length - 1]?.children;
            if(children){
               const childWeights = super.crossover(children);
               this.network.model.setWeights(childWeights);
            }
        });
        return this;
    }

    setFitness(id , fitness){
        this.generations[this.generations.length - 1].children[id].fitness = fitness;
        return this;
    }
    setScore(id , score){
        this.generations[this.generations.length - 1].children[id].score = score;
        return this;
    }
}