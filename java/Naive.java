import java.lang.Math;

public class Naive extends Base {

  public Naive(double[][] X_train, int[] y_train) {
    super(X_train, y_train);
  }

  private static double euc_dist(double[] x1, double[] x2) {
    double dist = 0;
    for(int i=0; i<x1.length; i++) {
      dist += ((x1[i] - x2[i])*(x1[i] - x2[i]));
    }
    return Math.sqrt(dist);
  }

  protected double distance(double[] x1, double[] x2) {
    // implemented distance as euclidean for now but can be changed easily
    return this.euc_dist(x1, x2);
  }

  public int[] predict(double[][] X_test, int k) {
    return this.y_train;
  }

}
