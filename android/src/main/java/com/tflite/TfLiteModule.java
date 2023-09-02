package com.tflite;

import androidx.annotation.NonNull;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.util.Log;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.module.annotations.ReactModule;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


@ReactModule(name = TfLiteModule.NAME)
public class TfLiteModule extends ReactContextBaseJavaModule {
  public static final String NAME = "TfLite";

  // presets for rgb conversion
  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 128.0f;

  // options for model interpreter
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();
  // tflite graph
  private Interpreter tflite;
  // holds all the possible labels for model
  private List<String> labelList;
  // holds the selected image data as bytes
  private ByteBuffer imgData = null;
  // holds the probabilities of each label for non-quantized graphs
  private float[][] labelProbArray = null;
  // holds the probabilities of each label for quantized graphs
  private byte[][] labelProbArrayB = null;


  // Dictates whether or not the model passed in is quantized
  private boolean quant;

  // input image dimensions for the Inception Model
  private int DIM_IMG_SIZE_X;
  private int DIM_IMG_SIZE_Y;
  private int DIM_PIXEL_SIZE;

  // int array to hold image data
  private int[] intValues;

  // priority queue that will hold the top results from the CNN
  PriorityQueue<WritableMap> sortedLabels =
    new PriorityQueue<>(
      1,
      new Comparator<WritableMap>() {
        @Override
        public int compare(WritableMap lhs, WritableMap rhs) {
          return Double.compare(rhs.getDouble("confidence"), lhs.getDouble("confidence"));
        }
      });

  private final ReactApplicationContext reactContext;

  public TfLiteModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.reactContext = reactContext;
  }

  @ReactMethod
  public void close() {
    tflite.close();
    labelList = null;
    labelProbArray = null;
    labelProbArrayB = null;
    imgData = null;
  }

  @ReactMethod
  public void loadModel(final String modelPath, final String labelsPath, final Callback callback)
    throws IOException {
    //initialize graph and labels
    try{
      tflite = new Interpreter(loadModelFile(modelPath), tfliteOptions);
      labelList = loadLabelList(labelsPath);
    } catch (Exception ex){
      Log.d(NAME, ex.toString());
      return;
    }

    // model is quantized if the weights are bytes and not floats
    Tensor tensor = tflite.getInputTensor(0);
    quant = tensor.dataType() == DataType.UINT8;

    // get input size of image
    DIM_IMG_SIZE_X = tensor.shape()[1];
    DIM_IMG_SIZE_Y = tensor.shape()[2];
    DIM_PIXEL_SIZE = tensor.shape()[3];

    // initialize array that holds image data
    intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

    // initialize byte array. The size depends if the input data needs to be quantized or not
    if(quant){
      imgData =
        ByteBuffer.allocateDirect(
          DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    } else {
      imgData =
        ByteBuffer.allocateDirect(
          4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    }
    imgData.order(ByteOrder.nativeOrder());

    // initialize probabilities array. The datatypes that array holds depends if the input data needs to be quantized or not
    if(quant){
      labelProbArrayB= new byte[1][labelList.size()];
    } else {
      labelProbArray = new float[1][labelList.size()];
    }

    callback.invoke(null, "successfully loaded model");
  }

  @ReactMethod
  public void runModelOnImage(final String path, final int numResults, final float threshold,
                               final Callback callback) throws IOException {
    // create bitmap from given input image
    InputStream inputStream = new FileInputStream(path.replace("file://", ""));
    Bitmap bitmap_orig = BitmapFactory.decodeStream(inputStream);

    // resize the bitmap to the required input size to the CNN
    Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
    // convert bitmap to byte array
    convertBitmapToByteBuffer(bitmap);
    // pass byte data to the graph
    if(quant){
      tflite.run(imgData, labelProbArrayB);
    } else {
      tflite.run(imgData, labelProbArray);
    }
    // display the results
    callback.invoke(null, GetTopN(numResults, threshold));
  }

  private WritableArray GetTopN(int numResults, float threshold) {

    for (int i = 0; i < labelList.size(); ++i) {
      float confidence;
      if(quant){
        // convert byte to the same float value
        confidence = (labelProbArrayB[0][i] & 0xff) / 255.0f;
      } else {
        confidence = labelProbArray[0][i];
      }
      if (confidence > threshold) {
        WritableMap res = Arguments.createMap();
        res.putInt("index", i);
        res.putString("label", labelList.size() > i ? labelList.get(i) : "unknown");
        res.putDouble("confidence", confidence);
        sortedLabels.add(res);
      }
    }

    WritableArray results = Arguments.createArray();
    int recognitionsSize = Math.min(sortedLabels.size(), numResults);
    for (int i = 0; i < recognitionsSize; ++i) {
      results.pushMap(sortedLabels.poll());
    }
    return results;
  }

  // loads the labels from the label txt file in assets into a string array
  private List<String> loadLabelList(String labelsPath) throws IOException {
    File labelsFile = new File(reactContext.getFilesDir() + labelsPath);
    List<String> labelList = new ArrayList<String>();
    BufferedReader reader =
      new BufferedReader(new InputStreamReader(new FileInputStream(labelsFile)));
    String line;
    while ((line = reader.readLine()) != null) {
      labelList.add(line);
    }
    reader.close();
    return labelList;
  }

  // loads tflite graph from file
  private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
    File modelFile = new File(reactContext.getFilesDir() + modelPath);
    FileChannel channel = new FileInputStream(modelFile).getChannel();
    return channel.map(FileChannel.MapMode.READ_ONLY, 0, modelFile.length());
  }

  // converts bitmap to byte array which is passed in the tflite graph
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // loop through all pixels
    int pixel = 0;
    for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
      for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
        final int val = intValues[pixel++];
        // get rgb values from intValues where each int holds the rgb values for a pixel.
        // if quantized, convert each rgb value to a byte, otherwise to a float
        if(quant){
          imgData.put((byte) ((val >> 16) & 0xFF));
          imgData.put((byte) ((val >> 8) & 0xFF));
          imgData.put((byte) (val & 0xFF));
        } else {
          imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
          imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
          imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        }

      }
    }
  }

  // resizes bitmap to given dimensions
  public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;
    Matrix matrix = new Matrix();
    matrix.postScale(scaleWidth, scaleHeight);
    Bitmap resizedBitmap = Bitmap.createBitmap(
      bm, 0, 0, width, height, matrix, false);
    return resizedBitmap;
  }


  @Override
  @NonNull
  public String getName() {
    return NAME;
  }
}
