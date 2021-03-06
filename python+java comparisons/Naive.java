// A naive approach to kNN, running in O(mndk) time.

import java.lang.Math;
import java.util.Arrays;

public class Naive extends Base {

  public Naive(double[][] X_train, double[] y_train) {
    super(X_train, y_train);
  }

  public double[] predict(double[][] X_test, int K) {

    double[] pred = new double[0];

    for(int j=0; j<X_test.length; j++) {
      boolean[] selected = new boolean[this.N];
      for(int e=0; e<selected.length; e++) {selected[e] = false;}
      double[] neighbour_values = new double[0];

      for(int k=0; k<K; k++) {
        int nearest_i = 0;
        double[] nearest_x = null;
        double nearest_y = 0;
        double nearest_d = 0;

        for(int i=0; i<this.X_train.length; i++) {
          if (selected[i]) {continue;}

          double d = this.dist(X_test[j], this.X_train[i]);

          if(((nearest_i == 0) && (nearest_y == 0) &&
              (nearest_x == null) && (nearest_d == 0)) || d < nearest_d) {

               nearest_i = i;
               nearest_x = this.X_train[i];
               nearest_y = this.y_train[i];
               nearest_d = d;
          }
        }
        selected[nearest_i] = true;

        // this is the irritating way to append values to an array in Java :(

        double[] temp = new double[(neighbour_values.length + 1)];
        for(int t=0; t<neighbour_values.length; t++) {
          temp[t] = neighbour_values[t];
        }
        temp[neighbour_values.length] = nearest_y;
        neighbour_values = temp;
      }

      double neighbour_sum = 0;
      for(int i=0; i<neighbour_values.length; i++) {
        neighbour_sum += neighbour_values[i];
      }

      //System.out.print(Arrays.toString(neighbour_values) + "\n");

      neighbour_sum = (double) neighbour_sum / K;
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
