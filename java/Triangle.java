// A naive approach to kNN, running in O(mndk) time.

import java.lang.Math;
import java.util.Arrays;
import java.util.Comparator;

public class Triangle extends Base {

  private double[][] train_dist;

  public Triangle(double[][] X_train, int[] y_train) {
    super(X_train, y_train);

    for(int i=0; i<(X_train.length-1); i++) {
      for(int j=0; j<i+1; j++) {
        this.train_dist[i][j] = this.dist(X_train[i], X_train[j]);
      }
    }
  }

  public double[] predict(double[][] X_test, int K) {

    double[] pred = new double[0];

    for(int j=0; j<X_test.length; j++) {
      boolean[] possible = new boolean[this.N];
      for(int e=0; e<possible.length; e++) {possible[e] = true;}
      int[] curr_i = new int[K];
      double[][] curr_x = new double[K][this.P];
      double[] curr_y = new double[K];
      double[] curr_d = new double[K];

      for(int i=0; i<this.X_train.length; i++) {
        if (!possible[i]) {continue;}
        double d = this.dist(X_test[j], this.X_train[i]);

        if(i>=K) {
          for(int l=0; l<(N-(i+1)); l++) {
            if (!possible[l]) {continue;}
            if (Math.abs(d - this.train_dist[i][l]) > curr_d[K-1]) {
              possible[i + l + 1] = false;
            }
          }
        }

        for(int k=0; k<K; k++) {
          int[] test_i = new int[K];
          double[][] test_x = new double[K][this.P];
          double[] test_y = new double[K];
          double[] test_d = new double[K];
          if(((curr_i[k] == test_i[k]) &&
              Arrays.equals(curr_x[k], test_x[k]) &&
              (curr_y[k] == test_y[k]) &&
              (curr_d[k] == test_d[k])) ||
             curr_d[k] > d) {
               curr_i[k] = i;
               curr_x[k] = this.X_train[i];
               curr_y[k] = this.y_train[i];
               curr_d[k] = d;

               // this is the irritating way to delete elements of an array in Java :(

               int[] temp_i = new int[K-1];
               double[][] temp_x = new double[K-1][this.P];
               double[] temp_y = new double[K-1];
               double[] temp_d = new double[K-1];
               for(int t=0; t<curr_i.length-1; t++) {
                 temp_i[t] = curr_i[t];
                 temp_x[t] = curr_x[t];
                 temp_y[t] = curr_y[t];
                 temp_d[t] = curr_d[t];
               }
               curr_i = temp_i;
               curr_x = temp_x;
               curr_y = temp_y;
               curr_d = temp_d;
               break;
             }
           }
        }

        double sum = 0;
        for (int e=0; e<curr_y.length; e++) {
          sum += curr_y[e];
        }
        sum /= K;

        // more appending incoming

        double[] temp = new double[(pred.length + 1)];
        for(int t=0; t<pred.length; t++) {
          temp[t] = pred[t];
        }
        temp[pred.length] = sum;
        pred = temp;
      }
      return pred;
  }
}
