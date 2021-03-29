// An abstract base class for other kNN implementations.

public abstract class Base {

  protected double[][] X_train; // training data
  protected int[] y_train; // training labels

  protected int N; // number of training data points
  protected int P; // dimension of training data points

  // constructors

  protected Base(double[][] X_train, int[] y_train) {
    this.X_train = X_train;
    this.y_train = y_train;

    this.N = this.X_train.length;
    this.P = this.X_train[0].length;
  }

  // getters

  public double[][] getTrain() {
    return this.X_train;
  }

  public int[] getTest() {
    return this.y_train;
  }

  // distance function

  protected static double dist(double[] x1, double[] x2) {
    double dist = 0;
    for(int i=0; i<x1.length; i++) {
      dist += ((x1[i] - x2[i])*(x1[i] - x2[i]));
    }
    return Math.sqrt(dist);
  }

  // predict method

  public abstract double[] predict(double[][] X_test, int k);

}
