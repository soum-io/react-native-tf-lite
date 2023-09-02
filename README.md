# react-native-tf-lite

Run TensorFlow Lite models in React Native Android apps.

## Example

Please see [the example app](https://github.com/soum-io/react-native-tf-lite/tree/main/example) to get started!

<img src="demo.gif" width="300">

## Installation

```sh
npm install react-native-tf-lite
```

## Usage

```js
const RNFS = require('react-native-fs');
import { Tflite } from 'react-native-tf-lite';
let tflite = new Tflite()

// load model
tflite.loadModel({
    // internal storage path to model. This library supports both regular and quanitzed TensorFlow Lite models.
    modelPath: '/Model/graph.lite',
    // internal storage path to labels. Labels file must have each label on new line.
    labelsPath: '/Model/labels.txt'
},
    (err, res) => {
        if (err)
            console.log(err);
        else
            console.log(res);
    }
);
);

// run an image against loaded model
tflite.runModelOnImage({
    // image path must be a uri and the image must be square
    path: 'file://' + RNFS.DocumentDirectoryPath + '/example.jpg', 
    // maximum number of results to return from model. Results are ordered by confidence
    numResults: 10,
    // only return results that pass a cretain threshold of confidence
    threshold: 0
},
    (err, res) => {
        if (err)
            console.log(err + '\n' + res);
        else {
            console.log(results);
        }
    });

// Releases all resources used by the model on the native side
tflite.close()



```



## License

MIT
