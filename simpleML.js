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

    constructor() {}

}

simpleML.sequential = class sequential extends simpleML{
    constructor() {
        super();
        this.model = tf.sequential();
    }
    input(nodes) {
        this.inputNodes = nodes;
        return this;
    }
    #getInputNodes(){
        if(this.inputNodes){
            const nodes = this.inputNodes;
            this.inputNodes = undefined;
            return nodes;
        }
    }
    layer(nodes, activation = 'sigmoid') {
        this.model.add(tf.layers.dense({
            inputShape: this.#getInputNodes(),
            units: nodes,
            activation,
            units: nodes,
          }));
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
        const prec = options.precision || 3;
        const round = options.round || false;
        let result;
        tf.tidy(() => {
        result = this.model.predict(tf.tensor2d(input))
                            .dataSync()
        });
        return Array.from(result).map(item => round ? Math.round(item) : parseFloat(item.toFixed(prec)));
    }

    tensors(){
        return tf.memory().numTensors;
    }
}