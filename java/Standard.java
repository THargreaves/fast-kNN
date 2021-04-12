// A naive approach to kNN, running in O(mndk) time.

import java.lang.Math;
import java.util.Arrays;
import java.util.Comparator;

public class Standard extends Base {

  public Standard(double[][] X_train, double[] y_train) {
    super(X_train, y_train);
  }

  public double[] predict(double[][] X_test, int K) {

    double[] pred = new double[0];

    for(int j=0; j<X_test.length; j++) {
      double[][] neighbour_values = new double[0][2];

      for(int i=0; i<this.X_train.length; i++) {

        double d = this.dist(X_test[j], this.X_train[i]);

        // this is the irritating way to append values to an array in Java :(

        double[][] temp = new double[(neighbour_values.length + 1)][2];
        for(int t=0; t<neighbour_values.length; t++) {
          temp[t][0] = neighbour_values[t][0];
          temp[t][1] = neighbour_values[t][1];
        }
        temp[neighbour_values.length][0] = d;
        temp[neighbour_values.length][1] = this.y_train[i];
        neighbour_values = temp;
      }
      Arrays.sort(neighbour_values, Comparator.comparingDouble(o -> o[0]));

      double neighbour_sum = 0;
      for(int i=0; i<K; i++) {
        neighbour_sum += neighbour_values[i][1];
      }

      neighbour_sum /= K;
      double[] temp2 = new double[(pred.length + 1)];
      for(int t=0; t<pred.length; t++) {
        temp2[t] = pred[t];
      }
      temp2[pred.length] = neighbour_sum;
      pred = temp2;
    }
    return pred;
  }
}
