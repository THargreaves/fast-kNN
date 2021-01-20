
import java.lang.Math;

public class kNN {

  private double[][] Xtr; // training data
  private int[] Ytr; // training labels

  private int N; // number of training data points
  private int d; // dimension of training data points

  // Constructors

  public kNN() {
  }

  public kNN(double[][] Xtr, int[] Ytr) {
    this.Xtr = Xtr;
    this.Ytr = Ytr;

    this.N = this.Xtr.length;
    this.d = this.Xtr[0].length;
  }

  // Fit function in case training data was not pre-specified

  public void fit(double[][] Xtr, int[] Ytr) {
    this.Xtr = Xtr;
    this.Ytr = Ytr;
  }

  // getters

  public double[][] getTrain() {
    return this.Xtr;
  }

  public int[] getTest() {
    return this.Ytr;
  }

  // Euclidean distance function - private, since for internal use only

  private static double euc_dist(double[] x, double[] y) {
    double dist = 0;

    for(int i = 0; i < x.length; i++) {
      dist += ((x[i] - y[i])*(x[i] - y[i]));
    }

    return Math.sqrt(dist);
  }

  // Predict method, which takes a matrix of training data and returns an array
  // of predicted labels for a given k-nearest neighbours

  public int[] predict(int k, double[][] Xtt) {

    // KNN GOES HERE

    int[] Ytt = new int[Xtt.length];
    return this.Ytr;
  }
}
