import { NativeModules, Platform } from 'react-native';

const { TfLite } = NativeModules;

export class Tflite {
  loadModel(args, callback) {
    TfLite.loadModel(
      args['modelPath'],
      args['labelsPath'] || '',
      (error, response) => {
        callback && callback(error, response);
      });
  }

  runModelOnImage(args, callback) {
    TfLite.runModelOnImage(
      args['path'],
      args['numResults'] || 5,
      args['threshold'] != null ? args['threshold'] : 0.1,
      (error, response) => {
        callback && callback(error, response);
      });
  }

  close() {
    TfLite.close();
  }
}
