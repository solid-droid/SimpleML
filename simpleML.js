
isBrowser = false;
class simpleML {


    networks = {}
    constructor(){}

    createNetwork(name , {input = 0 , hidden = [], output = 0 , weights = false}){
      const network = this.generateNN(input, hidden, output[0] , output[1] , weights);
      this.networks[name] = {lr: 0.001, loss: 'bse' ,network}
      return network;
    }
    
    generateNN(inputs, hidden, output , activate, weights){
     const network = new simpleML.Network(inputs, output);
     hidden.forEach(([nodes, activation]) => network.addHiddenLayer(nodes , activation));
     network.outputActivation(activate);
      if(!weights){
        network.makeWeights();
      }
      return network;
    }

    async train(network,data,{lr=0.0001, loss='bce', getloss , onComplete }={}){
      this.networks[network].network.lr = lr;
      this.networks[network].network.setLossFunction(loss);
      data.forEach(x => {
        this.networks[network].network.backpropagate(x.input, x.output);
        if(getloss){
          getloss(this.networks[network].network.loss);
        }
      });
      if(onComplete){
        onComplete();
      }

    };

    predict(network, input){
      const [output] = this.networks[network].network.feedForward(input);
     return parseFloat(output.toFixed(4));
    }
}

  //////////////////////////////////
  //Shortening Mathjs functions:
  random = (a, b) => Math.random(1) * (b - a) + a;
  exp = (x) => Math.exp(x);
  abs = (x) => Math.abs(x);
  log = (x) => Math.log(x);
  pow = (x, e) => Math.pow(x, e);
  round = (x) => Math.round(x);
  sqrt = (x) => Math.sqrt(x);
  
  //Other math functions:
  cosh = (x) => (exp(x) + exp(-x)) / 2;
  
  // Pooling functions:
  poolfuncs = {
    max: function (arr) {
      let record = 0;
      let len = arr.length;
      for (let i = 0; i < len; i++) {
        if (arr[i] > record) {
          record = arr[i];
        }
      }
      return record;
    },
    min: function (arr) {
      let record = Infinity;
      let len = arr.length;
      for (let i = 0; i < len; i++) {
        if (arr[i] < record) {
          record = arr[i];
        }
      }
      return record;
    },
    avg: function (arr) {
      let sum = 0;
      let len = arr.length;
      for (let i = 0; i < len; i++) {
        sum += arr[i];
      }
      return sum / len;
    },
  };

simpleML.Methods = class Methods {

  /////////////////////
   static   bitLength(x) {
    if (x < 1) {
      return 1;
    } else {
      return Math.floor(Math.log(x) / Math.log(2)) + 1;
    }
  }

  static  numberToBinary(x, size) {
    let value = x;
    let sample = value.toString(2);
    let arr = [];
    let k = bitLength(x) - 1;
    for (let i = size - 1; i >= 0; i--) {
      let char = sample.charAt(k);
      if (char === '') {
        arr[i] = 0;
      } else {
        arr[i] = JSON.parse(char);
      }
      k--;
    }
    return arr;
  }

  static makeBinary(size, func) {
    let f;
    if (func !== undefined) {
      f = func;
    } else {
      f = function (x) {
        return x + 1;
      };
    }
    let data = [];
    for (let i = 0; i < Math.pow(2, size) - 1; i++) {
      let targetNum = f(i);
      if (bitLength(targetNum) <= size) {
        let obj = {
          input: numberToBinary(i, size),
          output: numberToBinary(targetNum, size),
        };
        data.push(obj);
      }
    }
    return data;
  }
  
//Activation s:

static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
static   sigmoid_d(x) {
    let x1 = simpleML.Methods.sigmoid(x);
    return x1 * (1 - x1);
  }
static   leakySigmoid(x) {
    return 1 / (1 + Math.exp(-x)) + x / 100;
  }
static leakySigmoid_d(x) {
   let x1 = simpleML.Methods.leakySigmoid(x);
    return x1 * (1 - x1);
  }
static   siLU(x) {
    return x / (1 + Math.exp(-x));
  }
static   siLU_d(x) {
    let top = 1 + Math.exp(-x) + x * Math.exp(-x);
    let down = Math.pow(1 + Math.exp(-x), 2);
    return top / down;
  }
static   tanH(x) {
    let top = Math.exp(x) - Math.exp(-x);
    let down = Math.exp(x) + Math.exp(-x);
    return top / down;
  }
static   tanH_d(x) {
    return 1 - Math.pow(simpleML.Methods.tanH(x), 2);
  }
  static   leakyReLUCapped(x) {
    if (x >= 0 && x <= 6) {
      return x;
    } else if (x < 0) {
      return 0.1 * x;
    } else {
      return 6;
    }
  }
  static   leakyReLUCapped_d(x) {
    if (x >= 0 && x <= 6) {
      return 1;
    } else if (x < 0) {
      return 0.1;
    } else {
      return 0;
    }
  }
  static   leakyReLU(x) {
    if (x >= 0) {
      return 1 * x;
    } else {
      return 0.01 * x;
    }
  }
  static   leakyReLU_d(x) {
    if (x >= 0) {
      return 1;
    } else {
      return 0.01;
    }
  }
  static   reLU(x) {
    if (x >= 0) {
      return 1 * x;
    } else {
      return 0;
    }
  }
  static   reLU_d(x) {
    if (x >= 0) {
      return 1;
    } else {
      return 0;
    }
  }
  static   sinc(x) {
    if (x === 0) {
      return 1;
    } else {
      return Math.sin(x) / x;
    }
  }
  static  sinc_d(x) {
    if (x === 0) {
      return 0;
    } else {
      return Math.cos(x) / x - Math.sin(x) / (x * x);
    }
  }
  static   softsign(x) {
    return x / (1 + Math.abs(x));
  }
  static  softsign_d(x) {
    let down = 1 + Math.abs(x);
    return 1 / (down * down);
  }
  static  binary(x) {
    if (x <= 0) {
      return 0;
    } else {
      return 1;
    }
  }
  static  binary_d(x) {
    return 0;
  }
  static  softplus(x) {
    return Math.log(1 + Math.exp(x));
  }
  static softplus_d(x) {
    return simpleML.Methods.sigmoid(x);
  }
  

  
  // loss s:
  static  mae(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      sum += abs(y - yHat);
    }
    ans = sum / n;
    return ans;
  }
  static  bce(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      sum += y * log(yHat) + (1 - y) * log(1 - yHat);
    }
    ans = -sum / n;
    return ans;
  }
  static  lcl(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      sum += log(cosh(yHat - y));
    }
    ans = sum / n;
    return ans;
  }
  static  mbe(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      sum += y - yHat;
    }
    ans = sum / n;
    return ans;
  }
  //New experimental : Mean absolute exponential loss
  static  mael(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      let x = y - yHat;
  
      //Mean absolute exponential 
      let top = -x * (exp(-x) - 1);
      let down = exp(-x) + 1;
      sum += top / down;
    }
    ans = sum / n;
    return ans;
  }
  static  rmse(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      sum += pow(y - yHat, 2);
    }
    ans = sqrt(sum / n);
    return ans;
  }
  static  mce(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      sum += pow(abs(y - yHat), 3);
    }
    ans = sum / n;
    return ans;
  }
  static  mse(predictions, target) {
    let sum = 0;
    let ans = 0;
    let n = target.length;
    for (let i = 0; i < n; i++) {
      let y = target[i];
      let yHat = predictions[i];
      sum += pow(y - yHat, 2);
    }
    ans = sum / n;
    return ans;
  }
  static  quantile(predictions, target, percentile) {
    let q = percentile;
    let sum = 0;
    for (let i = 0; i < target.length; i++) {
      if (target[i] - predictions[i] >= 0) {
        sum += q * (target[i] - predictions[i]);
      } else {
        sum += (q - 1) * (target[i] - predictions[i]);
      }
    }
    return sum / target.length;
  }

}

const activations = {
    //Basic:
    sigmoid: simpleML.Methods.sigmoid,
    sigmoid_d: simpleML.Methods.sigmoid_d,
    tanH: simpleML.Methods.tanH,
    tanH_d: simpleML.Methods.tanH_d,
    siLU: simpleML.Methods.siLU,
    siLU_d: simpleML.Methods.siLU_d,
    reLU: simpleML.Methods.reLU,
    reLU_d: simpleML.Methods.reLU_d,
    leakyReLU: simpleML.Methods.leakyReLU,
    leakyReLU_d: simpleML.Methods.leakyReLU_d,
    sinc: simpleML.Methods.sinc,
    sinc_d: simpleML.Methods.sinc_d,
    softsign: simpleML.Methods.softsign,
    softsign_d: simpleML.Methods.softsign_d,
    binary: simpleML.Methods.binary,
    binary_d: simpleML.Methods.binary_d,
    softplus: simpleML.Methods.softplus,
    softplus_d: simpleML.Methods.softplus_d,
    //Experimental:
    leakySigmoid: simpleML.Methods.leakySigmoid,
    leakySigmoid_d: simpleML.Methods.leakySigmoid_d,
    leakyReLUCapped: simpleML.Methods.leakyReLUCapped,
    leakyReLUCapped_d: simpleML.Methods.leakyReLUCapped_d,
  }

const lossfuncs = {
    //Basic
    mae: simpleML.Methods.mae,
    bce: simpleML.Methods.bce,
    lcl: simpleML.Methods.lcl,
    mbe: simpleML.Methods.mbe,
    mce: simpleML.Methods.mce,
    mse: simpleML.Methods.mse,
    rmse: simpleML.Methods.rmse,
    //Experimental:
    mael: simpleML.Methods.mael,
    quantile: simpleML.Methods.quantile,
};

simpleML.Logger = class Logger {

    static warn (warning, method) {
        if (isBrowser) {
          console.warn('DannWarning: ' + warning);
          console.warn('> ' + method);
        } else {
          console.warn(warning);
          console.warn(method);
        }
        console.trace();
      };

      static error (error, method) {
        if (isBrowser) {
          console.error('DannError: ' + error);
          console.error('> ' + method);
        } else {
          console.error('\x1b[31m' + 'DannError: ' + error + '\x1b[0m');
          console.error('\x1b[31m' + '> ' + method + '\x1b[0m');
        }
        console.trace();
      };
}
simpleML.Add = class Add {
    activation  (name, activation, derivative) {
        if (typeof name !== 'string') {
          DannError.error('The name argument is not a string.', 'Add.activation');
          return;
        }
        if (activation.length !== 1 || derivative.length !== 1) {
           simpleML.Logger.error(
            'One of the functions specified does not have only 1 argument.',
            'Add.activation'
          );
          return;
        } else {
           activations[name] = activation;
           activations[name + '_d'] = derivative;
          return;
        }
      };

     loss (name, loss) {
        if (typeof name !== 'string') {
           simpleML.Logger.error('The name argument is not a string.', 'Add.loss');
          return;
        }
        if (loss.length === 2) {
          lossfuncs[name] = loss;
        } else {
           simpleML.Logger.error(
            'The loss function specified can only have 2 argument.',
            'newActivation'
          );
          return;
        }
      };
}

simpleML.Matrix = class Matrix {
    cols;
    rows;
    matrix;
    constructor(rows = 0 , cols = 0) {
            this.rows = rows;
            this.cols = cols;
            let m = [[]];
            for (let i = 0; i < rows; i++) {
              m[i] = [];
              for (let j = 0; j < cols; j++) {
                m[i][j] = 0;
              }
            }
            this.matrix = m;
    }


    add(n) {
        if (n instanceof Matrix) {
          if (this.rows !== n.rows || this.cols !== n.cols) {
             simpleML.Logger.error('Matrix dimensions should match', 'Matrix.prototype.add');
            return;
          } else {
            for (let i = 0; i < this.rows; i++) {
              for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] += n.matrix[i][j];
              }
            }
            return this;
          }
        } else {
          for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
              this.matrix[i][j] += n;
            }
          }
          return this;
        }
      };

    static add(a, b) {
        let ans = new simpleML.Matrix(a.rows, a.cols);
        if (a.rows !== b.rows || a.cols !== b.cols) {
           simpleML.Logger.error('Matrix dimensions should match', 'Matrix.add');
          return;
        } else {
          for (let i = 0; i < ans.rows; i++) {
            for (let j = 0; j < ans.cols; j++) {
              ans.matrix[i][j] = a.matrix[i][j] + b.matrix[i][j];
            }
          }
        }
        return ans;
      };

      set(matrix) {
        if (
          typeof matrix.length === 'number' &&
          typeof matrix[0].length === 'number' &&
          typeof matrix === 'object'
        ) {
          this.matrix = matrix;
          this.rows = matrix.length;
          this.cols = matrix[0].length;
        } else {
           simpleML.Logger.error(
            'the argument of set(); must be an array within an array. Here is an example: [[1,0],[0,1]]',
            'Matrix.prototype.set'
          );
          return;
        }
      };

      log(options) {
        let table = false;
        if (options !== undefined) {
          if (options.table) {
            table = options.table;
          }
        }
        if (table) {
          console.table(this.matrix);
        } else {
          console.log(this);
        }
      };

      addPrecent(scalar) {
        for (let i = 0; i < this.rows; i++) {
          for (let j = 0; j < this.cols; j++) {
            let w = this.matrix[i][j];
            this.matrix[i][j] += w * scalar;
          }
        }
      };

      addRandom(magnitude, prob) {
        let newMatrix = simpleML.Matrix.make(this.rows, this.cols);
        if (prob <= 0 || prob > 1) {
           simpleML.Logger.error(
            'Probability argument must be between 0 and 1',
            'Matrix.prototype.addRandom'
          );
        } else {
          for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
              let w = this.matrix[i][j];
              let ran = random(0, 1);
              if (ran < prob) {
                newMatrix[i][j] = w + w * random(-magnitude, magnitude);
              }
            }
          }
        }
        this.set(newMatrix);
      };

      fillCol(col, num) {
        if (col >= this.cols) {
           simpleML.Logger.error(
            'The column index specified is too large for this matrix.',
            'Matrix.prototype.fillCol'
          );
          return;
        }
        for (let i = 0; i < this.rows; i++) {
          this.matrix[i][col] = num;
        }
        return this;
      };

      fillRow(row, num) {
        if (row >= this.rows) {
           simpleML.Logger.error(
            'The row index specified is too large for this matrix.',
            'Matrix.prototype.fillRow'
          );
          return;
        }
        this.matrix[row].fill(num);
        return this;
      };

     static fromArray(array) {
        let m = new simpleML.Matrix(array.length, 1);
        for (let i = 0; i < array.length; i++) {
          m.matrix[i][0] = array[i];
        }
        return m;
      };

      initiate(value = 0) {
        if (value !== undefined) {
          if (typeof value === 'number') {
            for (let i = 0; i < this.matrix.length; i++) {
              for (let j = 0; j < this.matrix[i].length; j++) {
                this.matrix[i][j] = value;
              }
            }
            return this;
          } else {
             simpleML.Logger.error(
              'The value entered as an argument is not a number',
              'Matrix.prototype.initiate'
            );
            return;
          }
        }
      };

      insert(value, x, y) {
        if (typeof value !== 'number') {
           simpleML.Logger.error(
            'Expected Number for "value" argument',
            'Matrix.prototype.insert'
          );
          return;
        }
        if (typeof x !== 'number') {
           simpleML.Logger.error(
            'Expected Number for "x" argument',
            'Matrix.prototype.insert'
          );
          return;
        }
        if (typeof y !== 'number') {
           simpleML.Logger.error(
            'Expected Number for "y" argument',
            'Matrix.prototype.insert'
          );
          return;
        }
        if (x < this.rows && y < this.cols) {
          this.matrix[x][y] = value;
          return this;
        } else {
           simpleML.Logger.error(
            ' x, y arguments exceed the matrix dimensions.',
            'Matrix.prototype.insert'
          );
        }
      };

    static  make(rows = 0, cols = 0) {
        let m = [[]];
        for (let i = 0; i < rows; i++) {
          m[i] = [];
          for (let j = 0; j < cols; j++) {
            m[i][j] = 0;
          }
        }
        return m;
      };

      map(f) {
        for (let i = 0; i < this.rows; i++) {
          for (let j = 0; j < this.cols; j++) {
            let v = this.matrix[i][j];
            this.matrix[i][j] = f(v);
          }
        }
        return this;
      };

     static map(m, f) {
        if (m instanceof Matrix) {
          for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
              let v = m.matrix[i][j];
              m.matrix[i][j] = f(v);
            }
          }
          return m;
        } else {
           simpleML.Logger.error(
            'First argument must be an instance of Matrix',
            'Matrix.map'
          );
          return;
        }
      };

     mult(n) {
        if (n instanceof Matrix) {
          if (this.rows !== n.rows || this.cols !== n.cols) {
             simpleML.Logger.error(
              'The matrix dimensions should match in order to multiply their values. If you are looking for dot product, try Matrix.mult',
              'Matrix.prototype.mult'
            );
            return;
          } else {
            for (let i = 0; i < this.rows; i++) {
              for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] *= n.matrix[i][j];
              }
            }
            return this;
          }
        } else {
          for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
              this.matrix[i][j] *= n;
            }
          }
          return this;
        }
      };

      static mult(a, b, options = { mode: 'cpu' }) {
        let mode = '';
        if (options && options.mode) {
            mode = options.mode;
          }
        if (mode === 'cpu') {
          let ans = new simpleML.Matrix(a.rows, b.cols);
          if (a instanceof Matrix && b instanceof Matrix) {
            if (a.cols !== b.rows) {
               simpleML.Logger.error(
                'The rows of B must match the columns of A',
                'Matrix.mult'
              );
              return;
            } else {
              for (let i = 0; i < ans.rows; i++) {
                for (let j = 0; j < ans.cols; j++) {
                  let sum = 0;
                  for (let k = 0; k < a.cols; k++) {
                    sum += a.matrix[i][k] * b.matrix[k][j];
                  }
                  ans.matrix[i][j] = sum;
                }
              }
            }
            return ans;
          }
        } else {
           simpleML.Logger.error('mode specified is not valid', 'Matrix.prototype.mult');
          return;
        }
      };

      randomize(min, max) {
        for (let i = 0; i < this.matrix.length; i++) {
          for (let j = 0; j < this.matrix[i].length; j++) {
            this.matrix[i][j] = random(min, max);
          }
        }
        return this;
      };

     sub(n) {
        if (n instanceof Matrix) {
          if (this.rows !== n.rows || this.cols !== n.cols) {
             simpleML.Logger.error('Matrix dimensions should match', 'Matrix.prototype.sub');
            return;
          } else {
            for (let i = 0; i < this.rows; i++) {
              for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] -= n.matrix[i][j];
              }
            }
            return this;
          }
        } else {
          for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
              this.matrix[i][j] -= n;
            }
          }
          return this;
        }
      };

    static  sub(a, b) {
        if (a instanceof Matrix && b instanceof Matrix) {
          if (a.rows !== b.rows || a.cols !== b.cols) {
             simpleML.Logger.error('The matrix dimensions should match', 'Matrix.sub');
            return undefined;
          } else {
            let result = new simpleML.Matrix(a.rows, a.cols);
            for (let i = 0; i < result.rows; i++) {
              for (let j = 0; j < result.cols; j++) {
                result.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j];
              }
            }
            return result;
          }
        } else {
           simpleML.Logger.error('The arguments should be p5.MatrixTensors', 'Matrix.sub');
          return undefined;
        }
      };


      toArray() {
        let ans = [];
        if (this.cols === 1) {
          for (let i = 0; i < this.rows; i++) {
            ans[i] = this.matrix[i][0];
          }
          return ans;
        } else if (this.rows === 1) {
          ans = this.matrix[0];
          return ans;
        } else {
           simpleML.Logger.error(
            'None of the lengths of the matrix equal 1',
            'Matrix.prototype.toArray'
          );
          return undefined;
        }
      };

     static toArray(m) {
        let ans = [];
        if (m.cols === 1) {
          for (let i = 0; i < m.rows; i++) {
            ans[i] = m.matrix[i][0];
          }
          return ans;
        } else if (m.rows === 1) {
          ans = m.matrix[0];
          return ans;
        } else {
           simpleML.Logger.error(
            'None of the lengths of the matrix equal 1',
            'Matrix.toArray'
          );
          return undefined;
        }
      };

     static transpose(m) {
        let result = new simpleML.Matrix(m.cols, m.rows);
        for (let i = 0; i < m.rows; i++) {
          for (let j = 0; j < m.cols; j++) {
            result.matrix[j][i] = m.matrix[i][j];
          }
        }
        return result;
      };
      
}

simpleML.Layer = class Layer  {
  constructor (type, arg1, arg2, arg3, arg4, arg5) {
    
    this.type = type;
    this.subtype = this.getSubtype();
    if (this.subtype !== 'pool') {
      // Neuron Layers
      if (this.type === 'hidden' || this.type === 'output') {
        this.size = arg1;
        this.setFunc(arg2);
        this.layer = new simpleML.Matrix(this.size, 1);
      } else if (this.type === 'input') {
        this.size = arg1;
        this.layer = new simpleML.Matrix(this.size, 1);
      }
    } else if (this.subtype === 'pool') {
      // Pooling Layers
      this.stride = arg3;
      this.sampleSize = arg2;
      this.inputSize = arg1;
  
      //Optional X&Y size parameters
      if (arg4 !== undefined && arg5 !== undefined) {
        this.sizeX = arg4;
        this.sizeY = arg5;
      } else {
        this.sizeX = Math.sqrt(this.inputSize);
        this.sizeY = this.sizeX;
        if (this.sizeX !== Math.floor(this.sizeX)) {
          console.error(
            'Dann Error: the array can not be set in a square matrix'
          );
          console.trace();
          return;
        }
      }
      //get the size of output.
      this.size = simpleML.Layer.getPoolOutputLength(arg2, arg3, this.sizeX, this.sizeY);
      let divx = this.inputSize / this.sizeX;
      let divy = this.inputSize / this.sizeY;
  
      // Handle Unvalid Layer Formats
      if (divx !== Math.floor(divx) && divy !== Math.floor(divy)) {
        console.error(
          'Dann Error: the width & height value specified to arrange the inputted array as a matrix are not valid. (The array length must be divisible by the width & height values.)'
        );
        console.trace();
        return;
      }
      if (this.size !== Math.floor(this.size)) {
        console.error(
          "Dann Error: the Width must be divisible by the stride (jumps size). Width is the root of the array's length."
        );
        console.trace();
        return;
      }
  
      //Input values.
      this.input = new simpleML.Matrix(this.inputSize, 1);
      //Output values.
      this.layer = new simpleML.Matrix(this.size, 1);
  
      // picking the pooling function:
      this.prefix = this.getPrefix();
      this.poolfunc = poolfuncs[this.prefix];
  
      //Downsampling function, appling the pool function to the segmented arrays.
      this.downsample = function (data, f, s) {
        this.input = simpleML.Matrix.fromArray(data);
        //Split inputs in smaller pool arrays.
        let samples = simpleML.Layer.selectPools(data, f, s, this.sizeX, this.sizeY);
        let output = [];
        for (let i = 0; i < samples.length; i++) {
          output[i] = this.poolfunc(samples[i]);
        }
        this.layer = simpleML.Matrix.fromArray(output);
        return output;
      };
    } else {
      // Handle Unvalid Layer types.
      if (typeof this.type === 'string') {
        console.error(
          "Dann Error: The Layer type '" + this.type + "' is not valid."
        );
        console.trace();
      } else {
        console.error('Dann Error: You need to specify a valid type of Layer');
        console.trace();
      }
    }
  }

  feed(data, options) {
    if (this.subtype !== 'pool') {
       simpleML.Logger.error(
        "This function can only be used by Layers with 'pool' subtype",
        'Layer.prototype.feed'
      );
    } else {
      let showLog = false;
      let table = false;
      let f = this.sampleSize;
      let s = this.stride;
      if (options !== undefined) {
        if (options.log) {
          showLog = options.log;
        }
        if (options.table) {
          table = options.table;
        }
      }
      if (data.length !== this.inputSize) {
         simpleML.Logger.error(
          'Dann Error: The data you are trying to feed to this ' +
            this.type +
            ' layer is not the same length as the number of input this layer has.',
          'Layer.prototype.feed'
        );
        return;
      } else {
        let downsampled = this.downsample(data, f, s);
        if (showLog) {
          if (table) {
            console.table(downsampled);
          } else {
            console.log(downsampled);
          }
        }
        return downsampled;
      }
    }
  };

  static getPoolOutputLength(f, s, w, h) {
    return ((w - f) / s + 1) * ((h - f) / s + 1);
  };

  static getSqIndex(w, i, j) {
    return w * j + i;
  };

  log() {
    console.log(this);
  };

  static selectPools(arr, f, s, w, h) {
    let len = arr.length;
    if (w !== Math.floor(w)) {
      return;
    } else if (w / s !== Math.floor(w / s)) {
      return;
    }
    let samples = [];
    for (let y = 0; y + f <= h; y += s) {
      for (let x = 0; x + f <= w; x += s) {
        let sample = [];
        for (let j = 0; j < f; j++) {
          for (let i = 0; i < f; i++) {
            sample.push(arr[Layer.getSqIndex(w, i + x, j + y)]);
          }
        }
        samples.push(sample);
      }
    }
    return samples;
  };

  setFunc(act) {
    let obj = simpleML.Layer.stringTofunc(act);
    if (obj !== undefined) {
      this.actname = obj.name;
      this.actname_d = obj.name_d;
      this.actfunc = obj.func;
      this.actfunc_d = obj.func_d;
    } else {
       simpleML.Logger.error('Bad activation information', 'Layer.prototype.setFunc');
      return;
    }
  };

  static stringTofunc(str) {
    let act = str;
    let der = act + '_d';
    let func;
    let func_d;
    func =  activations[act];
    func_d =  activations[der];
  
    if (func !== undefined) {
      if (func_d !== undefined) {
        return { name: act, name_d: der, func: func, func_d: func_d };
      } else {
         simpleML.Logger.error(
          "Dann Error: You need to create the derivative of your custom function. The activation function specified '" +
            str +
            "' does not have a derivative assigned. The activation function was set to the default 'sigmoid'.",
          'Layer.stringTofunc'
        );
        return;
      }
    } else {
       simpleML.Logger.error(
        "Dann Error: the activation function '" +
          str +
          "' is not a valid activation function. The activation function was set to the default 'sigmoid'.",
        'Layer.stringTofunc'
      );
      return;
    }
  };

    getPrefix() {
        let str = this.type;
        let len = str.length;
        let prefix = str.slice(0, len - 4);
        return prefix;
    };

    getSubtype() {
    let str = this.type;
    let len = str.length;
    let subtype = str.slice(len - 4, len);
    if (subtype === 'pool') {
        return subtype;
    } else {
        return str;
    }
    };

}

simpleML.Network = class Network  {

    constructor (i = 1, o = 1) {
        
        this.i = i;
        this.inputs = new simpleML.Layer('input', i);
      
        this.o = o;
        this.outputs = new simpleML.Layer('output', o, 'sigmoid');
      
        this.Layers = [this.inputs, this.outputs];
        this.weights = [];
        this.biases = [];
        this.errors = [];
        this.gradients = [];
        this.dropout = [];
      
        this.outs = [];
        this.loss = 0;
        this.losses = [];
        this.lr = 0.001;
        this.arch = [i, o];
      
        this.epoch = 0;
        this.recordLoss = false;
      
        this.lossfunc = lossfuncs['mse'];
        this.lossfunc_s = this.lossfunc.name;
        this.percentile = 0.5;
    }

    addDropout(rate) {
        // if weights do not exist, cancel.
        if (this.weights.length === 0) {
           simpleML.Logger.error(
            'You need to initialize weights before using this function, use Dann.prototype.makeWeights();',
            'Dann.prototype.addDropout'
          );
          return;
        }
      
        // Set the map function argument 'rate'
        let func = ((v) => {
          let a = 1 - rate;
          return Math.floor(Math.random() + a);
        })
          .toString()
          .replace(/rate/gm, rate);
        let randomMap = eval(func);
      
        // Determine randomly based on the rate which neuron is inactive
        let inactive = [];
        for (let i = 0; i < this.Layers.length; i++) {
          let neuronList = new Array(this.Layers[i].size).fill(1).map(randomMap);
          inactive.push(neuronList);
        }
      
        // Create the dropout matrices, which are the same dimensions as a weight matrix
        this.dropout = [];
        for (let i = 0; i < this.weights.length; i++) {
          this.dropout.push(
            new simpleML.Matrix(this.weights[i].rows, this.weights[i].cols).initiate(1)
          );
        }
      
        // Iterate through dropout matrices and add a 0 value to every row or column affected by an idle neuron
        for (let i = 0; i < inactive.length; i++) {
          if (i === 0) {
            // Input layer, affects the matrix in front of the layer.
            for (let j = 0; j < inactive[i].length; j++) {
              if (inactive[i][j] === 0) {
                this.dropout[i].fillCol(j, 0);
              }
            }
          } else if (i === inactive.length - 1) {
            // Output layers, affects the matrix before the layer.
            for (let j = 0; j < inactive[i].length; j++) {
              if (inactive[i][j] === 0) {
                this.dropout[i - 1].fillRow(j, 0);
              }
            }
          } else {
            // Hidden layers, affects two matrices.
            for (let j = 0; j < inactive[i].length; j++) {
              if (inactive[i][j] === 0) {
                this.dropout[i - 1].fillRow(j, 0);
                this.dropout[i].fillCol(j, 0);
              }
            }
          }
        }
      };
      
    addHiddenLayer(size, act) {
        if (act !== undefined) {
          if ( activations[act] === undefined) {
            if (typeof act === 'string') {
               simpleML.Logger.error(
                "'" +
                  act +
                  "' is not a valid activation function, as a result, the activation function was set to 'sigmoid'.",
                'Dann.prototype.addHiddenLayer'
              );
            }
            act = 'sigmoid';
          }
        } else {
          act = 'sigmoid';
        }
        this.arch.splice(this.arch.length - 1, 0, size);
        let layer = new simpleML.Layer('hidden', size, act);
        this.Layers.splice(this.Layers.length - 1, 0, layer);
      };

    backpropagate(
        inputs,
        target,
        options = {}
      ) {
        //optional parameter values:
        let showLog = options.log || false;
        let mode = options.mode || 'cpu';
        let recordLoss = options.saveLoss || false;
        let table = options.table || false;
        let dropout = options.dropout || undefined;
      
        let targets = new simpleML.Matrix(0, 0);
        if (target.length === this.o) {
          targets = simpleML.Matrix.fromArray(target);
        } else {
           simpleML.Logger.error(
            'The target array length does not match the number of ouputs the dannjs model has.',
            'Dann.prototype.backpropagate'
          );
          return;
        }
        if (typeof this.lr !== 'number') {
           simpleML.Logger.error(
            'The learning rate specified (Dann.lr property) is not a number.',
            'Dann.prototype.backpropagate'
          );
          return;
        }
      
        this.outs = this.feedForward(inputs, { log: false, mode: mode });
        this.errors[this.errors.length - 1] = simpleML.Matrix.sub(
          targets,
          this.Layers[this.Layers.length - 1].layer
        );
        this.gradients[this.gradients.length - 1] = simpleML.Matrix.map(
          this.Layers[this.Layers.length - 1].layer,
          this.Layers[this.Layers.length - 1].actfunc_d
        );
        this.gradients[this.gradients.length - 1].mult(
          this.errors[this.errors.length - 1]
        );
        this.gradients[this.gradients.length - 1].mult(this.lr);
      
        if (dropout !== undefined) {
          if (dropout >= 1) {
             simpleML.Logger.error(
              'The probability value can not be bigger or equal to 1',
              'Dann.prototype.backpropagate'
            );
            return;
          } else if (dropout <= 0) {
             simpleML.Logger.error(
              'The probability value can not be smaller or equal to 0',
              'Dann.prototype.backpropagate'
            );
            return;
          }
          // init Dropout here.
          this.addDropout(dropout);
        }
      
        for (let i = this.weights.length - 1; i > 0; i--) {
          let h_t = simpleML.Matrix.transpose(this.Layers[i].layer);
          let weights_deltas = simpleML.Matrix.mult(this.gradients[i], h_t);
      
          if (dropout !== undefined) {
            // Compute dropout
            weights_deltas = weights_deltas.mult(this.dropout[i]);
          }
      
          this.weights[i].add(weights_deltas);
          this.biases[i].add(this.gradients[i]);
      
          let weights_t = simpleML.Matrix.transpose(this.weights[i]);
          this.errors[i - 1] = simpleML.Matrix.mult(weights_t, this.errors[i]);
          this.gradients[i - 1] = simpleML.Matrix.map(
            this.Layers[i].layer,
            this.Layers[i].actfunc_d
          );
          this.gradients[i - 1].mult(this.errors[i - 1]);
          this.gradients[i - 1].mult(this.lr);
        }
      
        let i_t = simpleML.Matrix.transpose(this.Layers[0].layer);
        let weights_deltas = simpleML.Matrix.mult(this.gradients[0], i_t);
      
        if (dropout !== undefined) {
          // Add dropout here
          weights_deltas = weights_deltas.mult(this.dropout[0]);
        }
      
        this.weights[0].add(weights_deltas);
        this.biases[0].add(this.gradients[0]);
      
        this.loss = this.lossfunc(this.outs, target, this.percentile);
        if (recordLoss === true) {
          this.losses.push(this.loss);
        }
        if (showLog === true) {
          console.log('Prediction: ');
          if (table) {
            console.table(this.outs);
          } else {
            console.log(this.outs);
          }
          console.log('target: ');
          if (table) {
            console.table(target);
          } else {
            console.log(target);
          }
          console.log('Loss: ', this.loss);
        }
      };

    train(inputs, target, options) {
        return this.backpropagate(inputs, target, options);
    };

    createFromJSON(data) {
        const model = new Dann();
        model.fromJSON(data);
        return model;
      };

      feedForward(inputs, options = {}) {
        //optional parameter values:
        let showLog = options.log || false;
        let table = options.table || false;
        let roundData = false;
        let dec = pow(10, options.decimals) || 1000;
        if (options.decimals !== undefined) {
          roundData = true;
        }
      
        if (inputs.length === this.i) {
          this.Layers[0].layer = simpleML.Matrix.fromArray(inputs);
        } else {
          for (let i = 0; i < this.o; i++) {
            this.outs[i] = 0;
          }
           simpleML.Logger.error(
            'The input array length does not match the number of inputs the dannjs model has.',
            'Dann.prototype.feedForward'
          );
          return this.outs;
        }
        if (this.weights.length === 0) {
           simpleML.Logger.warn(
            'The weights were not initiated. Please use the Dann.makeWeights(); function after the initialization of the layers.',
            'Dann.prototype.feedForward'
          );
          this.makeWeights();
        }
      
        for (let i = 0; i < this.weights.length; i++) {
          let pLayer = this.Layers[i];
      
          let layerObj = this.Layers[i + 1];
      
          layerObj.layer = simpleML.Matrix.mult(this.weights[i], pLayer.layer);
          layerObj.layer.add(this.biases[i]);
          layerObj.layer.map(layerObj.actfunc);
        }
      
        this.outs = simpleML.Matrix.toArray(this.Layers[this.Layers.length - 1].layer);
        let out = this.outs;
        if (showLog === true) {
          if (roundData === true) {
            out = out.map((x) => round(x * dec) / dec);
          }
          if (table === true) {
            console.log('Prediction: ');
            console.table(out);
          } else {
            console.log('Prediction: ');
            console.log(out);
          }
        }
        return out;
      };
      
      feed(inputs, options) {
        return this.feedForward(inputs, options);
      };

      fromJSON(data) {
        this.i = data.arch[0];
        this.inputs = new simpleML.Matrix(this.i, 1);
        this.o = data.arch[data.arch.length - 1];
        this.outputs = new simpleML.Matrix(this.o, 1);
      
        let slayers = JSON.parse(data.lstr);
        for (let i = 0; i < slayers.length; i++) {
          let layerdata = JSON.parse(slayers[i]);
          let layerObj = new simpleML.Layer(layerdata.type, layerdata.size, layerdata.actname);
          this.Layers[i] = layerObj;
        }
        this.makeWeights();
        let sweights = JSON.parse(data.wstr);
        for (let i = 0; i < sweights.length; i++) {
          this.weights[i].set(JSON.parse(sweights[i]));
        }
        let sbiases = JSON.parse(data.bstr);
        for (let i = 0; i < sbiases.length; i++) {
          this.biases[i].set(JSON.parse(sbiases[i]));
        }
        let serrors = JSON.parse(data.estr);
        for (let i = 0; i < serrors.length; i++) {
          this.errors[i].set(JSON.parse(serrors[i]));
        }
        let sgradients = JSON.parse(data.gstr);
        for (let i = 0; i < sgradients.length; i++) {
          this.gradients[i].set(JSON.parse(sgradients[i]));
        }
      
        this.lossfunc_s = data.lf;
        if (isBrowser) {
          this.lossfunc = window[data.lf];
        } else {
          this.lossfunc = lossfuncs[data.lf];
        }
        this.outs = simpleML.Matrix.toArray(this.Layers[this.Layers.length - 1].layer);
        this.loss = data.loss;
        this.losses = [];
        this.lr = data.lrate;
        this.arch = data.arch;
        this.epoch = data.e;
        this.percentile = data.per;
        return this;
      };

      log(
        options = {
          struct: true,
          misc: true,
        }
      ) {
        //Optional parameters values:
        let showWeights = options.weights || false;
        let showGradients = options.gradients || false;
        let showErrors = options.errors || false;
        let showBiases = options.biases || false;
        let showBaseSettings = options.struct || false;
        let showOther = options.misc || false;
        let showDetailedLayers = options.layers || false;
        let table = options.table || false;
        let decimals = 1000;
      
        // Limit decimals to maximum of 21
        if (options.decimals > 21) {
           simpleML.Logger.error('Maximum number of decimals is 21.', 'Dann.prototype.log');
          decimals = pow(10, 21);
        } else {
          decimals = pow(10, options.decimals) || decimals;
        }
      
        // Details sets all values to true.
        if (options.details) {
          let v = options.details;
          showGradients = v;
          showWeights = v;
          showErrors = v;
          showBiases = v;
          showBaseSettings = v;
          showOther = v;
          showDetailedLayers = v;
        }
      
        // Initiate weights if they weren't initiated allready.
        if (this.weights.length === 0) {
          this.makeWeights();
        }
        if (showBaseSettings === true) {
          console.log('Dann Model:');
        }
        if (showBaseSettings) {
          console.log('Layers:');
          for (let i = 0; i < this.Layers.length; i++) {
            let layerObj = this.Layers[i];
            let str = layerObj.type + ' Layer: ';
            let afunc = '';
            if (i === 0) {
              str = 'input Layer:   ';
              afunc = '       ';
            } else if (i === layerObj.length - 1) {
              str = 'output Layer:  ';
              afunc = '  (' + layerObj.actname + ')';
            } else {
              afunc = '  (' + layerObj.actname + ')';
            }
            console.log('\t' + str + layerObj.size + afunc);
            if (showDetailedLayers) {
              console.log(this.Layers[i]);
            }
          }
        }
        if (showErrors) {
          console.log('Errors:');
          for (let i = 0; i < this.errors.length; i++) {
            let e = simpleML.Matrix.toArray(this.errors[i]);
            let er = [];
            for (let j = 0; j < e.length; j++) {
              er[j] = round(e[j] * decimals) / decimals;
            }
            console.log(er);
          }
        }
        if (showGradients) {
          console.log('Gradients:');
          for (let i = 0; i < this.gradients.length; i++) {
            let g = simpleML.Matrix.toArray(this.gradients[i]);
            let gr = [];
            for (let j = 0; j < g.length; j++) {
              gr[j] = round(g[j] * decimals) / decimals;
            }
            console.log(gr);
          }
        }
        if (showWeights) {
          console.log('Weights:');
          for (let i = 0; i < this.weights.length; i++) {
            let w = this.weights[i];
            w.log({ decimals: options.decimals, table: table });
          }
        }
        if (showBiases) {
          console.log('Biases:');
          for (let i = 0; i < this.biases.length; i++) {
            let b = simpleML.Matrix.toArray(this.biases[i]);
            let br = [];
            for (let j = 0; j < b.length; j++) {
              br[j] = round(b[j] * decimals) / decimals;
            }
            console.log(br);
          }
        }
        if (showOther) {
          console.log('Other Values: ');
      
          console.log('\t' + 'Learning rate: ' + this.lr);
          console.log('\t' + 'Loss Function: ' + this.lossfunc_s);
          console.log('\t' + 'Current Epoch: ' + this.epoch);
          console.log('\t' + 'Latest Loss: ' + this.loss);
        }
        console.log(' ');
        return;
      }

      makeWeights(arg1, arg2) {
        let min = -1;
        let max = 1;
        if (arg1 !== undefined && arg2 !== undefined) {
          min = arg1;
          max = arg2;
        }
        for (let i = 0; i < this.Layers.length - 1; i++) {
          let previousLayerObj = this.Layers[i];
          let layerObj = this.Layers[i + 1];
      
          let weights = new simpleML.Matrix(layerObj.layer.rows, previousLayerObj.layer.rows);
          let biases = new simpleML.Matrix(layerObj.layer.rows, 1);
      
          weights.randomize(min, max);
          biases.randomize(1, -1);
          this.weights[i] = weights;
          this.biases[i] = biases;
      
          this.errors[i] = new simpleML.Matrix(layerObj.layer.rows, 1);
          this.gradients[i] = new simpleML.Matrix(layerObj.layer.rows, 1);
      
          if (layerObj.actfunc === undefined) {
            layerObj.setFunc('sigmoid');
          }
        }
        for (let i = 0; i < this.Layers.length; i++) {
          let layerObj = this.Layers[i];
          this.arch[i] = layerObj.layer.rows;
        }
      };

      mapWeights(f) {
        if (typeof f === 'function') {
          for (let i = 0; i < this.weights.length; i++) {
            this.weights[i].map(f);
          }
        } else {
           simpleML.Logger.error('Argument must be a function', 'Dann.prototype.mapWeights');
        }
      };

      mutateAdd(randomFactor) {
        if (typeof randomFactor !== 'number') {
           simpleML.Logger.error(
            'randomFactor argument must be a number.',
            'Dann.prototype.mutateAdd'
          );
          return;
        } else {
          for (let i = 0; i < this.weights.length; i++) {
            this.weights[i].addPercent(randomFactor);
          }
        }
      };
      
      mutateRandom(range, probability) {
        if (typeof range !== 'number') {
           simpleML.Logger.error(
            'Range argument must be a number.',
            'Dann.prototype.mutateRandom'
          );
          return;
        }
        if (probability !== undefined) {
          if (typeof probability !== 'number') {
             simpleML.Logger.error(
              'Probability argument must be a number.',
              'Dann.prototype.mutateRandom'
            );
            return;
          }
        } else {
          probability = 1;
        }
        for (let i = 0; i < this.weights.length; i++) {
          this.weights[i].addRandom(range, probability);
        }
      };

      outputActivation(act) {
        if ( activations[act] === undefined && !isBrowser) {
          if (typeof act === 'string') {
             simpleML.Logger.error(
              "'" +
                act +
                "' is not a valid activation function, as a result, the activation function is set to 'sigmoid' by default.",
              'Dann.prototype.outputActivation'
            );
            return;
          } else {
             simpleML.Logger.error(
              "Did not detect a string value, as a result, the activation function is set to 'sigmoid' by default.",
              'Dann.prototype.outputActivation'
            );
            return;
          }
        }
        this.Layers[this.Layers.length - 1].setFunc(act);
      };

      setLossFunction(
        name,
        percentile = 0.5
      ) {
        this.percentile = percentile;
        let func = lossfuncs[name];
        if (func === undefined) {
          if (typeof name === 'string') {
             simpleML.Logger.error(
              "'" +
                name +
                "' is not a valid loss function, as a result, the model's loss function is set to 'mse' by default.",
              'Dann.prototype.setLossFunction'
            );
            return;
          } else {
             simpleML.Logger.error(
              "Did not detect string value, as a result, the loss function is set to 'mse' by default.",
              'Dann.prototype.setLossFunction'
            );
            return;
          }
        }
        this.lossfunc_s = name;
        this.lossfunc = func;
      };

      toJSON() {
        //weights
        let wdata = [];
        for (let i = 0; i < this.weights.length; i++) {
          wdata[i] = JSON.stringify(this.weights[i].matrix);
        }
        let w_str = JSON.stringify(wdata);
        //layers
        let ldata = [];
        for (let i = 0; i < this.Layers.length; i++) {
          ldata[i] = JSON.stringify(this.Layers[i]);
        }
        let l_str = JSON.stringify(ldata);
        //biases
        let bdata = [];
        for (let i = 0; i < this.biases.length; i++) {
          bdata[i] = JSON.stringify(this.biases[i].matrix);
        }
        let b_str = JSON.stringify(bdata);
        //errors
        let edata = [];
        for (let i = 0; i < this.errors.length; i++) {
          edata[i] = JSON.stringify(this.errors[i].matrix);
        }
        let e_str = JSON.stringify(edata);
        //gradients
        let gdata = [];
        for (let i = 0; i < this.gradients.length; i++) {
          gdata[i] = JSON.stringify(this.gradients[i].matrix);
        }
        let g_str = JSON.stringify(gdata);
        const data = {
          wstr: w_str,
          lstr: l_str,
          bstr: b_str,
          estr: e_str,
          gstr: g_str,
          arch: this.arch,
          lrate: this.lr,
          lf: this.lossfunc_s,
          loss: this.loss,
          e: this.epoch,
          per: this.percentile,
        };
        return data;
      };

      toEs6(func) {
        let str = func.toString();
        let index = str.indexOf('(');
        let rstr = slicestring(str, index)[1];
        index = rstr.indexOf(')');
        let funcarr = slicestring(rstr, index - 1);
        let args = funcarr[0];
        index = funcarr[1].indexOf(')');
        let content = slicestring(funcarr[1], index)[1];
        let funcstr = '(' + args + ')=>' + content;
        let minfuncstr = minify(funcstr);
        return minfuncstr;
      }

      slicestring(str, index) {
        return [str.slice(0, index + 1), str.slice(index + 1, str.length)];
      }

      minify(string) {
        string = string.replace(/ = /g, '=');
        string = string.replace(/ \+ /g, '+');
        string = string.replace(/ - /g, '-');
        string = string.replace(/ \* /g, '*');
        string = string.replace(/ \/ /g, '/');
        string = string.replace(/for \(/g, 'for(');
        string = string.replace(/; /g, ';');
        string = string.replace(/\) {/g, '){');
        string = string.replace(/ < /g, '<');
        string = string.replace(/ > /g, '>');
        string = string.replace(/ \+= /g, '+=');
        string = string.replace(/;\}/g, '}');
        for (let i = 0; i < 5; i++) {
          string = string.replace(/\{ /g, '{');
          string = string.replace(/ \{/g, '{');
          string = string.replace(/\} /g, '}');
          string = string.replace(/\t/g, '');
          string = string.replace(/\n/g, '');
        }
        for (let i = 0; i < 5; i++) {
          string = string.replace(/; /g, ';');
        }
        return string;
      }

      toFunction(name = 'myDannFunction') {
        let stringfunc = 'function ' + name + '(input) {';
        // Setting weights
        stringfunc += 'let w = [];';
        for (let i = 0; i < this.weights.length; i++) {
          stringfunc +=
            'w[' + i + '] = ' + JSON.stringify(this.weights[i].matrix) + ';';
        }
        // Setting biases
        stringfunc += 'let b = [];';
        for (let i = 0; i < this.biases.length; i++) {
          stringfunc +=
            'b[' + i + '] = ' + JSON.stringify(this.biases[i].matrix) + ';';
        }
        stringfunc += 'let c = ' + JSON.stringify(this.arch) + ';';
        // Setting 
        stringfunc += 'let a = [];';
        for (let i = 1; i < this.Layers.length; i++) {
          let actname = this.Layers[i].actname;
          if (i !== 0) {
            let actfunc = toEs6(  activations[actname]).toString().split('\n');
            let minfunction = '';
            for (let u = 0; u < actfunc.length; u++) {
              minfunction += actfunc[u];
            }
            actfunc = minfunction.split('\t');
            let minfunction_notabs = '';
            for (let u = 0; u < actfunc.length; u++) {
              minfunction_notabs += actfunc[u];
            }
            stringfunc += 'a[' + i + '] = ' + minfunction_notabs + ';';
          } else {
            stringfunc += 'a[' + i + '] = undefined;';
          }
        }
        // Setting layers
        stringfunc += 'let l = [];';
        stringfunc +=
          'l[0] = [];' +
          'for (let i = 0; i < ' +
          this.i +
          '; i++) {' +
          'l[0][i] = [input[i]];' +
          '};';
        stringfunc +=
          'for (let i = 1; i < ' +
          this.Layers.length +
          '; i++) {' +
          'l[i] = [];' +
          'for (let j = 0; j < c[i]; j++) {' +
          'l[i][j] = [0];' +
          '}' +
          '};';
        // ffw
        stringfunc +=
          'for (let m = 0; m < ' +
          this.weights.length +
          '; m++) {' +
          // mult
          'for (let i = 0; i < w[m].length; i++) {' +
          'for (let j = 0; j < l[m][0].length; j++) {' +
          'let sum = 0;' +
          'for (let k = 0; k < w[m][0].length; k++) {' +
          'sum += w[m][i][k] * l[m][k][j];' +
          '};' +
          'l[m+1][i][j] = sum;' +
          '}' +
          '};' +
          // add biases
          'for (let i = 0; i < l[m+1].length; i++) {' +
          'for (let j = 0; j < l[m+1][0].length; j++) {' +
          'l[m+1][i][j] = l[m+1][i][j] + b[m][i][j];' +
          '}' +
          '};' +
          // map layer to activation function
          'for (let i = 0; i < l[m+1].length; i++) {' +
          'for (let j = 0; j < l[m+1][0].length; j++) {' +
          'l[m+1][i][j] = a[m+1](l[m+1][i][j]);' +
          '}' +
          '}' +
          '};' +
          // return output
          'let o = [];' +
          'for (let i = 0; i < ' +
          this.o +
          '; i++) {' +
          'o[i] = l[' +
          (this.Layers.length - 1) +
          '][i][0];' +
          '};' +
          'return o' +
          '}';
        //minify
        return minify(stringfunc);
      };
}
